/*****************************************************************************
 * Differentiable Gaussian Mixture Renderer
 * Copyright (C) 2023 Adam Celarek
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <stroke/gaussian.h>
#include <stroke/linalg.h>
#include <whack/Tensor.h>
#include <whack/kernel.h>
#include <whack/torch_interop.h>

#include "constants.h"
#include "grad/math.h"
#include "marching_steps.h"
#include "math.h"
#include "vol_marcher_backward.h"

namespace {
using namespace dgmr;
using namespace dgmr::vol_marcher;
namespace gaussian = stroke::gaussian;

// my own:
STROKE_DEVICES glm::vec3 clamp_cov_scales(const glm::vec3& cov_scales)
{
    const auto max_value = stroke::min(50.f, glm::compMax(cov_scales));
    const auto min_value = max_value * 0.01f;
    return glm::clamp(cov_scales, min_value, max_value);
}
} // namespace

dgmr::vol_marcher::Gradients dgmr::vol_marcher::backward(const dgmr::vol_marcher::ForwardData& data, const dgmr::vol_marcher::ForwardCache& cache, const torch::Tensor& incoming_grad)
{
    const auto fb_width = data.framebuffer.size<2>();
    const auto fb_height = data.framebuffer.size<1>();
    const auto n_gaussians = data.gm_weights.size<0>();
    const float focal_y = fb_height / (2.0f * data.tan_fovy);
    const float focal_x = fb_width / (2.0f * data.tan_fovx);
    const auto aa_distance_multiplier = (config::filter_kernel_SD * data.tan_fovx * 2) / fb_width;

    constexpr dim3 render_block_dim = { render_block_width, render_block_height };
    constexpr auto render_block_size = render_block_width * render_block_height;
    constexpr auto render_n_warps = render_block_size / 32;
    static_assert(render_n_warps * 32 == render_block_size);
    const dim3 render_grid_dim = whack::grid_dim_from_total_size({ data.framebuffer.size<2>(), data.framebuffer.size<1>() }, render_block_dim);

    // geometry buffers, filled by the forward preprocess pass
    auto g_rects = whack::make_tensor_view<const glm::uvec2>(cache.rects, n_gaussians);
    auto g_rgb = whack::make_tensor_view<const glm::vec3>(cache.rgb, n_gaussians);
    auto g_rgb_sh_clamped = whack::make_tensor_view<const glm::vec<3, bool>>(cache.rgb_sh_clamped, n_gaussians);
    auto g_depths = whack::make_tensor_view<const float>(cache.depths, n_gaussians);
    auto g_points_xy_image = whack::make_tensor_view<const glm::vec2>(cache.points_xy_image, n_gaussians);
    auto g_inverse_filtered_cov3d = whack::make_tensor_view<const stroke::Cov3_f>(cache.inverse_filtered_cov3d, n_gaussians);
    auto g_filtered_masses = whack::make_tensor_view<const float>(cache.filtered_masses, n_gaussians);
    auto g_tiles_touched = whack::make_tensor_view<const uint32_t>(cache.tiles_touched, n_gaussians);
    auto g_point_offsets = whack::make_tensor_view<const uint32_t>(cache.point_offsets, n_gaussians);
    auto v_remaining_transparency = whack::make_tensor_view<const float>(cache.remaining_transparency, fb_height, fb_width);

    Gradients grads;
    grads.gm_sh_params = torch::zeros({ n_gaussians, 16 * 3 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    grads.gm_weights = torch::zeros({ n_gaussians, 1 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    grads.gm_centroids = torch::zeros({ n_gaussians, 3 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    grads.gm_cov_scales = torch::zeros({ n_gaussians, 3 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    grads.gm_cov_rotations = torch::zeros({ n_gaussians, 4 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    whack::TensorView<const SHs<3>, 1> grad_gm_sh_params_view = whack::make_tensor_view<dgmr::SHs<3>>(grads.gm_sh_params, n_gaussians);
    whack::TensorView<const float, 1> grad_gm_weights_view = whack::make_tensor_view<float>(grads.gm_weights, n_gaussians);
    whack::TensorView<const glm::vec3, 1> grad_gm_centroids_view = whack::make_tensor_view<glm::vec3>(grads.gm_centroids, n_gaussians);
    whack::TensorView<const glm::vec3, 1> grad_gm_cov_scales_view = whack::make_tensor_view<glm::vec3>(grads.gm_cov_scales, n_gaussians);
    whack::TensorView<const glm::quat, 1> grad_gm_cov_rotations_view = whack::make_tensor_view<glm::quat>(grads.gm_cov_rotations, n_gaussians);

    const auto n_render_gaussians = unsigned(cache.point_offsets[n_gaussians - 1].item<int>());
    if (n_render_gaussians == 0)
        return grads;

    const auto v_incoming_grad = whack::make_tensor_view<const float>(incoming_grad, 3, fb_height, fb_width);

    // render backward
    // Let each tile blend its range of Gaussians independently in parallel
    {
        // Main rasterization method. Collaboratively works on one tile per
        // block, each thread treats one pixel. Alternates between fetching
        // and rasterizing data.

        const auto inversed_projectin_matrix = glm::inverse(data.proj_matrix);
        auto i_ranges = whack::make_tensor_view<const glm::uvec2>(cache.i_ranges, render_grid_dim.y, render_grid_dim.x);
        auto b_point_list = whack::make_tensor_view<const uint32_t>(cache.b_point_list, n_render_gaussians);
        whack::start_parallel(
            whack::Location::Device, render_grid_dim, render_block_dim, WHACK_DEVICE_KERNEL(=) {
                WHACK_UNUSED(whack_gridDim);
                // Identify current tile and associated min/max pixel range.
                const glm::uvec2 pix_min = { whack_blockIdx.x * whack_blockDim.x, whack_blockIdx.y * whack_blockDim.y };
                const glm::uvec2 pix_max = min(pix_min + glm::uvec2(whack_blockDim.x, whack_blockDim.y), glm::uvec2(fb_width, fb_height));
                const glm::uvec2 pix = pix_min + glm::uvec2(whack_threadIdx.x, whack_threadIdx.y);
                const glm::vec2 pix_ndc = glm::vec2(pix * glm::uvec2(2)) / glm::vec2(fb_width, fb_height) - glm::vec2(1);
                auto view_at_world = inversed_projectin_matrix * glm::vec4(pix_ndc, -1, 1.0);
                view_at_world /= view_at_world.w;

                const auto ray = stroke::Ray<3, float> { data.cam_poition, glm::normalize(glm::vec3(view_at_world) - data.cam_poition) };
                const unsigned thread_rank = whack_blockDim.x * whack_threadIdx.y + whack_threadIdx.x;

                // Check if this thread is associated with a valid pixel or outside.
                bool inside = pix.x < fb_width && pix.y < fb_height;
                // Done threads can help with fetching, but don't rasterize
                bool done = !inside;

                glm::vec3 grad_current_colour = {};
                float final_transparency = 1.f;
                float grad_current_transparency = {};
                glm::vec3 current_colour = {};
                if (inside) {
                    final_transparency = v_remaining_transparency(pix.y, pix.x);
                    const glm::vec3 final_colour = { data.framebuffer(0, pix.y, pix.x), data.framebuffer(1, pix.y, pix.x), data.framebuffer(2, pix.y, pix.x) };

                    // const auto final_colour = current_colour + current_transparency * data.background;
                    current_colour = final_colour - final_transparency * data.background;
                    grad_current_colour = { v_incoming_grad(0, pix.y, pix.x), v_incoming_grad(1, pix.y, pix.x), v_incoming_grad(2, pix.y, pix.x) };
                    grad_current_transparency = dot(grad_current_colour, data.background);
                }

                const auto render_g_range = i_ranges(whack_blockIdx.y, whack_blockIdx.x);
                const auto n_rounds = ((render_g_range.y - render_g_range.x + render_block_size - 1) / render_block_size);
                auto n_toDo = render_g_range.y - render_g_range.x;

                // Allocate storage for batches of collectively fetched data.
                __shared__ int collected_id[render_block_size];
                __shared__ float collected_3d_masses[render_block_size];
                __shared__ glm::vec3 collected_centroid[render_block_size];
                __shared__ stroke::Cov3<float> collected_inv_cov3[render_block_size];

                bool large_stepping_ongoing = true;
                float current_large_step_start = 0.f;
                float current_transparency = 1;

                while (large_stepping_ongoing) {
                    // Iterate over all gaussians and compute sample_sections
                    marching_steps::DensityArray<config::n_large_steps> sample_sections(current_large_step_start);
                    n_toDo = render_g_range.y - render_g_range.x;
                    bool done_1 = !inside;
                    for (unsigned i = 0; i < n_rounds; i++, n_toDo -= render_block_size) {
                        // End if entire block votes that it is done rasterizing
                        const int num_done = __syncthreads_count(done || done_1);
                        if (num_done == render_block_size)
                            break;

                        // Collectively fetch per-Gaussian data from global to shared
                        const int progress = i * render_block_size + thread_rank;
                        if (render_g_range.x + progress < render_g_range.y) {
                            unsigned coll_id = b_point_list(render_g_range.x + progress);
                            assert(coll_id < n_gaussians);
                            collected_id[thread_rank] = coll_id;
                            collected_centroid[thread_rank] = data.gm_centroids(coll_id);
                            collected_inv_cov3[thread_rank] = g_inverse_filtered_cov3d(coll_id);
                            collected_3d_masses[thread_rank] = g_filtered_masses(coll_id);
                        }
                        __syncthreads();

                        if (done || done_1)
                            continue;

                        // Iterate over current batch
                        for (unsigned j = 0; j < min(render_block_size, n_toDo); j++) {
                            const auto gaussian1d = gaussian::intersect_with_ray_inv_C(collected_centroid[j], collected_inv_cov3[j], ray);
                            const auto sd = stroke::sqrt(gaussian1d.C);

                            if (sample_sections.end() < g_depths(collected_id[j])) {
                                done_1 = true;
                                break;
                            }

                            auto mass_on_ray = gaussian1d.weight * collected_3d_masses[j];
                            if (mass_on_ray <= 1.1f / 255.f || mass_on_ray > 1'000)
                                continue;
                            if (gaussian1d.C + vol_marcher::config::workaround_variance_add_along_ray <= 0)
                                continue;
                            if (stroke::isnan(gaussian1d.centre))
                                continue;

                            const float start = gaussian1d.centre - sd * config::gaussian_relevance_sigma;
                            const float end = gaussian1d.centre + sd * config::gaussian_relevance_sigma;
                            const float delta_t = (sd * config::gaussian_relevance_sigma * 2) / (config::n_steps_per_gaussian - 1);

                            sample_sections.put({ start, end, delta_t });
                        }
                    }

                    // compute sampling
                    const auto bin_borders = marching_steps::sample<config::n_small_steps>(sample_sections);
                    whack::Array<glm::vec4, config::n_small_steps> bin_eval = {};

                    float dbg_mass_in_bins_closeed = 0;
                    float dbg_mass_in_bins_numerik_1 = 0;

                    // Iterate over batches again, and compute samples
                    n_toDo = render_g_range.y - render_g_range.x;
                    for (unsigned i = 0; i < n_rounds; i++, n_toDo -= render_block_size) {
                        // End if entire block votes that it is done rasterizing
                        const int num_done = __syncthreads_count(done);
                        if (num_done == render_block_size)
                            break;

                        // Collectively fetch per-Gaussian data from global to shared
                        const int progress = i * render_block_size + thread_rank;
                        if (render_g_range.x + progress < render_g_range.y) {
                            unsigned coll_id = b_point_list(render_g_range.x + progress);
                            assert(coll_id < n_gaussians);
                            collected_id[thread_rank] = coll_id;
                            collected_centroid[thread_rank] = data.gm_centroids(coll_id);
                            collected_inv_cov3[thread_rank] = g_inverse_filtered_cov3d(coll_id);
                            collected_3d_masses[thread_rank] = g_filtered_masses(coll_id);
                        }
                        __syncthreads();

                        if (done)
                            continue;

                        // Iterate over current batch
                        for (unsigned j = 0; j < min(render_block_size, n_toDo); j++) {
                            const auto inv_cov = collected_inv_cov3[j];
                            const auto gaussian1d = gaussian::intersect_with_ray_inv_C(collected_centroid[j], inv_cov, ray);
                            const auto centroid = gaussian1d.centre;
                            const auto variance = gaussian1d.C + vol_marcher::config::workaround_variance_add_along_ray;
                            const auto sd = stroke::sqrt(variance);
                            const auto inv_sd = 1 / sd;
                            const auto mass_on_ray = gaussian1d.weight * collected_3d_masses[j];

                            if (stroke::isnan(gaussian1d.centre))
                                continue;
                            if (mass_on_ray < 1.1f / 255.f || mass_on_ray > 1'000)
                                continue;
                            if (variance <= 0 || stroke::isnan(variance) || stroke::isnan(mass_on_ray) || mass_on_ray > 100'000)
                                continue; // todo: shouldn't happen any more after implementing AA?

                            const auto mass_in_bins = mass_on_ray * gaussian::integrate_normalised_inv_SD(centroid, inv_sd, { bin_borders[0], bin_borders[bin_borders.size() - 1] });

                            if (mass_in_bins < 0.0001f) { // performance critical
                                continue;
                            }
                            dbg_mass_in_bins_closeed += mass_in_bins;

                            auto cdf_start = gaussian::cdf_inv_SD(centroid, inv_sd, current_large_step_start);
                            for (auto k = 0u; k < bin_borders.size() - 1; ++k) {
                                const auto right = bin_borders[k + 1];
                                const auto cdf_end = gaussian::cdf_inv_SD(centroid, inv_sd, right);
                                const auto mass = stroke::max(0.f, (cdf_end - cdf_start) * mass_on_ray);
                                cdf_start = cdf_end;

                                if (mass < 0.00001f)
                                    continue;

                                dbg_mass_in_bins_numerik_1 += mass;
                                bin_eval[k] += glm::vec4(g_rgb(collected_id[j]) * mass, mass);
                            }
                        }
                    }

                    // blend //todo gradient backwards
                    whack::Array<glm::vec4, config::n_small_steps> grad_bin_eval;
                    // STROKE_DEVICES_INLINE stroke::grad::ThreeGrads<glm::vec<3, scalar_t>, scalar_t, whack::Array<glm::vec<4, scalar_t>, N>>
                    //  integrate_bins(glm::vec<3, scalar_t> current_colour, scalar_t current_transparency, scalar_t final_transparency, const whack::Array<glm::vec<4, scalar_t>, N>& bins,
                    // const glm::vec<3, scalar_t>& grad_colour, scalar_t grad_transparency)
                    cuda::std::tie(current_colour, current_transparency, grad_bin_eval) = math::grad::integrate_bins(current_colour, current_transparency, final_transparency, bin_eval, grad_current_colour, grad_current_transparency);

                    // todo gradient for #compute samples and write back to individual gaussians

                    done = done || sample_sections.size() == 0 || current_transparency < 1.f / 255.f;
                    const int num_done = __syncthreads_count(done);
                    if (num_done == render_block_size)
                        break;
                    current_large_step_start = bin_borders[bin_borders.size() - 1];
                }
            });
    }

    // // preprocess backward, run per Gaussian
    // {
    //     math::Camera<float> camera {
    //         data.view_matrix, data.proj_matrix, focal_x, focal_y, data.tan_fovx, data.tan_fovy, fb_width, fb_height
    //     };

    //     const dim3 block_dim = { 128 };
    //     const dim3 grid_dim = whack::grid_dim_from_total_size({ data.gm_weights.size<0>() }, block_dim);
    //     whack::start_parallel(
    //         whack::Location::Device, grid_dim, block_dim, WHACK_KERNEL(=) {
    //             WHACK_UNUSED(whack_gridDim);
    //             const auto idx = whack_blockIdx.x * whack_blockDim.x + whack_threadIdx.x;
    //             if (idx >= n_gaussians)
    //                 return;

    //             // Initialize touched tiles to 0. If this isn't changed,
    //             // this Gaussian will not be processed further.
    //             g_tiles_touched(idx) = 0;

    //             const auto centroid = data.gm_centroids(idx);
    //             if ((data.view_matrix * glm::vec4(centroid, 1.f)).z < 0.2) // adam doesn't understand, why projection matrix > 0 isn't enough.
    //                 return;

    //             const auto weights = data.gm_weights(idx);
    //             const auto scales = data.gm_cov_scales(idx) * data.cov_scale_multiplier;
    //             const auto rotations = data.gm_cov_rotations(idx);

    //             const auto screen_space_gaussian = math::splat<vol_marcher::config::gaussian_mixture_formulation>(weights, centroid, scales, rotations, camera, 0.3f);

    //             const auto cov3d = math::compute_cov(clamp_cov_scales(data.gm_cov_scales(idx)), data.gm_cov_rotations(idx));

    //             // low pass filter to combat aliasing
    //             const auto filter_kernel_size = glm::distance(centroid, data.cam_poition) * aa_distance_multiplier;
    //             const auto filtered_cov_3d = cov3d + stroke::Cov3_f(filter_kernel_size * filter_kernel_size);
    //             const auto mass = math::weight_to_mass<vol_marcher::config::gaussian_mixture_formulation>(weights, scales + glm::vec3(filter_kernel_size * filter_kernel_size));
    //             if (mass <= 0)
    //                 return; // clipped

    //             // using the more aggressive computation for calculating overlapping tiles:
    //             {
    //                 const glm::uvec2 my_rect = { (int)ceil(3.f * sqrt(screen_space_gaussian.cov[0])), (int)ceil(3.f * sqrt(screen_space_gaussian.cov[2])) };
    //                 g_rects(idx) = my_rect;
    //                 glm::uvec2 rect_min, rect_max;
    //                 getRect(screen_space_gaussian.centroid, my_rect, &rect_min, &rect_max, render_grid_dim);

    //                 const auto tiles_touched = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
    //                 if (tiles_touched == 0)
    //                     return; // clipped
    //                 g_tiles_touched(idx) = tiles_touched;
    //                 g_points_xy_image(idx) = screen_space_gaussian.centroid;
    //             }

    //             const auto inverse_filtered_cov = stroke::inverse(filtered_cov_3d);

    //             // g_depths(idx) = glm::length(data.cam_poition - centroid);
    //             g_depths(idx) = (glm::length(data.cam_poition - centroid) - math::max(scales) * config::gaussian_relevance_sigma / 2);

    //             // convert spherical harmonics coefficients to RGB color.
    //             g_rgb(idx) = computeColorFromSH(data.sh_degree, centroid, data.cam_poition, data.gm_sh_params(idx), &g_rgb_sh_clamped(idx));
    //             g_inverse_filtered_cov3d(idx) = inverse_filtered_cov;
    //             g_filtered_masses(idx) = mass;
    //         });
    // }
    return grads;
}
