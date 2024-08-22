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

#include "vol_marcher_backward.h"

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <stroke/gaussian.h>
#include <stroke/linalg.h>
#include <stroke/scalar_functions.h>
#include <whack/Tensor.h>
#include <whack/kernel.h>
#include <whack/torch_interop.h>

#include "constants.h"
#include "grad/math.h"
#include "marching_steps.h"
#include "math.h"
#include "util.h"

namespace {
using namespace dgmr;
using namespace dgmr::vol_marcher;
namespace gaussian = stroke::gaussian;

// my own:

template <typename scalar_t>
STROKE_DEVICES glm::vec<3, scalar_t> clamp_cov_scales(const glm::vec<3, scalar_t>& cov_scales)
{
    const auto max_value = stroke::min(scalar_t(50), glm::compMax(cov_scales));
    const auto min_value = max_value * scalar_t(0.01);
    return glm::clamp(cov_scales, min_value, max_value);
}
} // namespace

template <typename scalar_t>
dgmr::vol_marcher::Gradients dgmr::vol_marcher::backward(const whack::TensorView<const scalar_t, 3>& framebuffer, const dgmr::vol_marcher::ForwardData<scalar_t>& data, const dgmr::vol_marcher::ForwardCache& cache, const torch::Tensor& incoming_grad)
{
    using Vec2 = glm::vec<2, scalar_t>;
    using Vec3 = glm::vec<3, scalar_t>;
    using Vec4 = glm::vec<4, scalar_t>;
    using Mat3 = glm::mat<3, 3, scalar_t>;
    using Mat4 = glm::mat<4, 4, scalar_t>;
    using Cov3 = stroke::Cov3<scalar_t>;
    using Quat = glm::qua<scalar_t>;
    const auto torch_float_type = (sizeof(scalar_t) == 4) ? torch::kFloat32 : torch::kFloat64;

    const auto fb_width = framebuffer.template size<2>();
    const auto fb_height = framebuffer.template size<1>();
    const auto n_gaussians = data.gm_weights.template size<0>();
    const scalar_t focal_y = fb_height / (2.0f * data.tan_fovy);
    const scalar_t focal_x = fb_width / (2.0f * data.tan_fovx);
    const auto aa_distance_multiplier = (config::filter_kernel_SD * data.tan_fovx * 2) / fb_width;

    constexpr dim3 render_block_dim = { render_block_width, render_block_height };
    constexpr auto render_block_size = render_block_width * render_block_height;
    constexpr auto render_n_warps = render_block_size / 32;
    static_assert(render_n_warps * 32 == render_block_size);
    const dim3 render_grid_dim = whack::grid_dim_from_total_size({ fb_width, fb_height }, render_block_dim);

    // geometry buffers, filled by the forward preprocess pass
    auto g_rects = whack::make_tensor_view<const glm::uvec2>(cache.rects, n_gaussians);
    auto g_rgb = whack::make_tensor_view<const Vec3>(cache.rgb, n_gaussians);
    auto g_rgb_sh_clamped = whack::make_tensor_view<const glm::vec<3, bool>>(cache.rgb_sh_clamped, n_gaussians);
    auto g_depths = whack::make_tensor_view<const scalar_t>(cache.depths, n_gaussians);
    auto g_points_xy_image = whack::make_tensor_view<const Vec2>(cache.points_xy_image, n_gaussians);
    auto g_inverse_filtered_cov3d = whack::make_tensor_view<const Cov3>(cache.inverse_filtered_cov3d, n_gaussians);
    auto g_filtered_masses = whack::make_tensor_view<const scalar_t>(cache.filtered_masses, n_gaussians);
    auto g_tiles_touched = whack::make_tensor_view<const uint32_t>(cache.tiles_touched, n_gaussians);
    auto g_point_offsets = whack::make_tensor_view<const uint32_t>(cache.point_offsets, n_gaussians);
    auto v_remaining_transparency = whack::make_tensor_view<const scalar_t>(cache.remaining_transparency, fb_height, fb_width);

    Gradients grads;
    grads.gm_sh_params = torch::zeros({ n_gaussians, 16 * 3 }, torch::TensorOptions().dtype(torch_float_type).device(torch::kCUDA));
    grads.gm_weights = torch::zeros({ n_gaussians, 1 }, torch::TensorOptions().dtype(torch_float_type).device(torch::kCUDA));
    grads.gm_centroids = torch::zeros({ n_gaussians, 3 }, torch::TensorOptions().dtype(torch_float_type).device(torch::kCUDA));
    grads.gm_cov_scales = torch::zeros({ n_gaussians, 3 }, torch::TensorOptions().dtype(torch_float_type).device(torch::kCUDA));
    grads.gm_cov_rotations = torch::zeros({ n_gaussians, 4 }, torch::TensorOptions().dtype(torch_float_type).device(torch::kCUDA));
    whack::TensorView<SHs<3, scalar_t>, 1> grad_gm_sh_params = whack::make_tensor_view<dgmr::SHs<3, scalar_t>>(grads.gm_sh_params, n_gaussians);
    whack::TensorView<scalar_t, 1> grad_gm_weights = whack::make_tensor_view<scalar_t>(grads.gm_weights, n_gaussians);
    whack::TensorView<Vec3, 1> grad_gm_centroids = whack::make_tensor_view<Vec3>(grads.gm_centroids, n_gaussians);
    whack::TensorView<Vec3, 1> grad_gm_cov_scales = whack::make_tensor_view<Vec3>(grads.gm_cov_scales, n_gaussians);
    whack::TensorView<Quat, 1> grad_gm_cov_rotations = whack::make_tensor_view<Quat>(grads.gm_cov_rotations, n_gaussians);

    auto grad_g_rgb_data = torch::zeros({ n_gaussians, 3 }, torch::TensorOptions().dtype(torch_float_type).device(torch::kCUDA));
    auto grad_g_rgb = whack::make_tensor_view<Vec3>(grad_g_rgb_data, n_gaussians);
    auto grad_g_filtered_masses_data = torch::zeros({ n_gaussians, 1 }, torch::TensorOptions().dtype(torch_float_type).device(torch::kCUDA));
    auto grad_g_filtered_masses = whack::make_tensor_view<scalar_t>(grad_g_filtered_masses_data, n_gaussians);
    auto grad_g_inverse_filtered_cov3d_data = torch::zeros({ n_gaussians, 6 }, torch::TensorOptions().dtype(torch_float_type).device(torch::kCUDA));
    auto grad_g_inverse_filtered_cov3d = whack::make_tensor_view<Cov3>(grad_g_inverse_filtered_cov3d_data, n_gaussians);

    const auto n_render_gaussians = unsigned(cache.point_offsets[n_gaussians - 1].template item<int>());
    if (n_render_gaussians == 0)
        return grads;

    const auto v_incoming_grad = whack::make_tensor_view<const scalar_t>(incoming_grad, 3, fb_height, fb_width);

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
                const Vec2 pix = Vec2(pix_min + glm::uvec2(whack_threadIdx.x, whack_threadIdx.y)) + Vec2(0.5, 0.5);
                const Vec2 pix_ndc = Vec2(pix * Vec2(2)) / Vec2(fb_width, fb_height) - Vec2(1);
                auto view_at_world = inversed_projectin_matrix * Vec4(pix_ndc, 0, 1.0);
                view_at_world /= view_at_world.w;

                const auto ray = stroke::Ray<3, scalar_t> { data.cam_poition, glm::normalize(Vec3(view_at_world) - data.cam_poition) };
                const unsigned thread_rank = whack_blockDim.x * whack_threadIdx.y + whack_threadIdx.x;

                // Check if this thread is associated with a valid pixel or outside.
                bool inside = pix.x < fb_width && pix.y < fb_height;
                // Done threads can help with fetching, but don't rasterize
                bool done = !inside;

                Vec3 grad_current_colour = {};
                scalar_t final_transparency = 1.f;
                scalar_t grad_current_transparency = {};
                Vec3 current_colour = {};
                if (inside) {
                    final_transparency = v_remaining_transparency(pix.y, pix.x);
                    const Vec3 final_colour = { framebuffer(0, pix.y, pix.x), framebuffer(1, pix.y, pix.x), framebuffer(2, pix.y, pix.x) };

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
                __shared__ scalar_t collected_3d_masses[render_block_size];
                __shared__ Vec3 collected_centroid[render_block_size];
                __shared__ stroke::Cov3<scalar_t> collected_inv_cov3[render_block_size];

                bool large_stepping_ongoing = true;
                scalar_t current_large_step_start = 0.f;
                scalar_t current_transparency = 1;

                while (large_stepping_ongoing) {
                    // Iterate over all gaussians and compute sample_sections
                    marching_steps::DensityArray<config::n_large_steps, scalar_t> sample_sections(current_large_step_start);
#ifdef DGMR_TORCH_GRAD_CHECK_CONST_SAMPLES
                    // see vol_marcher_forward.cu
                    sample_sections.put({ 28.0, 30.0, (30.0 - 28.0) / (config::n_steps_per_gaussian - 1) });
                    sample_sections.put({ 30.0, 32.0, (32.0 - 30.0) / (config::n_steps_per_gaussian - 1) });
#else
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
                        // don't forget about the forward pass when editing this loop.
                        // think about moving to a function.
                        for (unsigned j = 0; j < stroke::min(render_block_size, n_toDo); j++) {
                            const auto gaussian1d = gaussian::intersect_with_ray_inv_C(collected_centroid[j], collected_inv_cov3[j], ray);
                            const auto sd = stroke::sqrt(gaussian1d.C);

                            if (sample_sections.end() < g_depths(collected_id[j])) {
                                done_1 = true;
                                break;
                            }

                            auto mass_on_ray = gaussian1d.weight * collected_3d_masses[j];
                            if (mass_on_ray <= 1.0f / 255.f || mass_on_ray > 1'000)
                                continue;
                            if (gaussian1d.C <= 0)
                                continue;
                            if (stroke::isnan(gaussian1d.centre))
                                continue;

                            const scalar_t start = gaussian1d.centre - sd * config::gaussian_relevance_sigma;
                            const scalar_t end = gaussian1d.centre + sd * config::gaussian_relevance_sigma;
                            const scalar_t delta_t = (sd * config::gaussian_relevance_sigma * 2) / (config::n_steps_per_gaussian - 1);

                            sample_sections.put({ start, end, delta_t });
                        }
                    }
#endif

                    // compute sampling
                    const auto bin_borders = marching_steps::sample<config::n_small_steps>(sample_sections);
                    whack::Array<Vec4, config::n_small_steps - 1> bin_eval = {};

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
                        for (unsigned j = 0; j < stroke::min(render_block_size, n_toDo); j++) {
                            math::sample_gaussian(collected_3d_masses[j], g_rgb(collected_id[j]), collected_centroid[j], collected_inv_cov3[j], ray, bin_borders, &bin_eval);
                        }
                    }

                    // gradient for blend
                    whack::Array<Vec4, config::n_small_steps - 1> grad_bin_eval;
                    cuda::std::tie(current_colour, current_transparency, grad_bin_eval) = math::grad::integrate_bins(current_colour, current_transparency, final_transparency, bin_eval, grad_current_colour, grad_current_transparency);

                    // gradient for compute samples and write back to individual gaussians
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
                        for (unsigned j = 0; j < stroke::min(render_block_size, n_toDo); j++) {
                            // math::sample_gaussian(collected_3d_masses[j], g_rgb(collected_id[j]), collected_centroid[j], collected_inv_cov3[j], ray, bin_borders, &bin_eval);
                            const auto grad_weight_rgb_pos_cov = dgmr::math::grad::sample_gaussian(collected_3d_masses[j], g_rgb(collected_id[j]), collected_centroid[j], collected_inv_cov3[j], ray, bin_borders, grad_bin_eval);
                            const auto gaussian_id = collected_id[j];
                            atomicAdd(&grad_g_filtered_masses(gaussian_id), grad_weight_rgb_pos_cov.m_first);

                            atomicAdd(&grad_g_rgb(gaussian_id).x, grad_weight_rgb_pos_cov.m_second.x);
                            atomicAdd(&grad_g_rgb(gaussian_id).y, grad_weight_rgb_pos_cov.m_second.y);
                            atomicAdd(&grad_g_rgb(gaussian_id).z, grad_weight_rgb_pos_cov.m_second.z);

                            atomicAdd(&grad_gm_centroids(gaussian_id).x, grad_weight_rgb_pos_cov.m_third.x);
                            atomicAdd(&grad_gm_centroids(gaussian_id).y, grad_weight_rgb_pos_cov.m_third.y);
                            atomicAdd(&grad_gm_centroids(gaussian_id).z, grad_weight_rgb_pos_cov.m_third.z);

                            for (auto i = 0u; i < 6; ++i)
                                atomicAdd(&grad_g_inverse_filtered_cov3d(gaussian_id)[i], grad_weight_rgb_pos_cov.m_fourth[i]);
                        }
                    }

                    done = done || sample_sections.size() == 0 || current_transparency < 1.f / 255.f;
                    const int num_done = __syncthreads_count(done);
                    if (num_done == render_block_size)
                        break;
                    current_large_step_start = bin_borders[bin_borders.size() - 1];
                }
            });
    }

    // preprocess, run per Gaussian
    {
        math::Camera<scalar_t> camera {
            data.view_matrix, data.proj_matrix, focal_x, focal_y, data.tan_fovx, data.tan_fovy, fb_width, fb_height
        };
        const dim3 block_dim = { 128 };
        const dim3 grid_dim = whack::grid_dim_from_total_size({ data.gm_weights.template size<0>() }, block_dim);
        whack::start_parallel(
            whack::Location::Device, grid_dim, block_dim, WHACK_KERNEL(=) {
                WHACK_UNUSED(whack_gridDim);
                const auto idx = whack_blockIdx.x * whack_blockDim.x + whack_threadIdx.x;
                if (idx >= n_gaussians)
                    return;
                if (g_tiles_touched(idx) == 0)
                    return;

                const auto centroid = data.gm_centroids(idx);
                if ((data.view_matrix * Vec4(centroid, 1.f)).z < 0.2)
                    return;

                const auto weight = data.gm_weights(idx);
                const auto scales = data.gm_cov_scales(idx);
                const auto rotation = data.gm_cov_rotations(idx);
                const auto dist = glm::length(centroid - data.cam_poition);

                const auto cov3d = math::compute_cov(scales, rotation);

                // low pass filter to combat aliasing
                const auto filter_kernel_size = dist * aa_distance_multiplier;
                const auto filter_kernel_size_sq = filter_kernel_size * filter_kernel_size;
                const auto filtered_cov_3d = cov3d + Cov3(filter_kernel_size_sq);
                const auto cov2d = math::affine_transform_and_cut(cov3d, Mat3(camera.view_matrix));
                const auto mass = math::weight_to_mass<vol_marcher::config::gaussian_mixture_formulation>(weight, scales, cov2d);
                if (mass <= 0)
                    return; // clipped

                const auto direction = (centroid - data.cam_poition) / dist;

                // grad computations
                const auto position_delta = (centroid - data.cam_poition);
                const auto grad_sh_and_direction = math::grad::sh_to_colour(data.gm_sh_params(idx), data.sh_degree, direction, grad_g_rgb(idx), g_rgb_sh_clamped(idx));
                grad_gm_sh_params(idx) = grad_sh_and_direction.m_left;
                auto [grad_centroid, grad_dist] = stroke::grad::divide_a_by_b(position_delta, dist, grad_sh_and_direction.m_right);

                const auto grad_filtered_cov_3d = stroke::grad::inverse(filtered_cov_3d, grad_g_inverse_filtered_cov3d(idx));
                const auto grad_mass = grad_g_filtered_masses(idx);
                auto [grad_weight, grad_scales, grad_cov2d] = math::grad::weight_to_mass<vol_marcher::config::gaussian_mixture_formulation>(weight, scales, cov2d, grad_mass);

                auto grad_cov3d = grad_filtered_cov_3d;
                const auto grad_filter_kernel_size_sq = grad_filtered_cov_3d[0] + grad_filtered_cov_3d[3] + grad_filtered_cov_3d[5];
                const auto grad_filter_kernel_size = grad_filter_kernel_size_sq * 2 * filter_kernel_size;
                grad_dist += grad_filter_kernel_size * aa_distance_multiplier;
                Quat grad_rotation = { 0, 0, 0, 0 };
                math::grad::affine_transform_and_cut(cov3d, Mat3(camera.view_matrix), grad_cov2d).addTo(&grad_cov3d, stroke::grad::Ignore::Grad);

                math::grad::compute_cov(scales, rotation, grad_cov3d).addTo(&grad_scales, &grad_rotation);
                grad_centroid += stroke::grad::length(centroid - data.cam_poition, grad_dist);

                grad_gm_cov_rotations(idx) = grad_rotation;
                grad_gm_cov_scales(idx) = grad_scales;
                grad_gm_weights(idx) = grad_weight;
                grad_gm_centroids(idx) += grad_centroid;
            });
    }
    return grads;
}

template dgmr::vol_marcher::Gradients dgmr::vol_marcher::backward<float>(const whack::TensorView<const float, 3>& framebuffer, const ForwardData<float>& forward_data, const ForwardCache& forward_cache, const torch::Tensor& grad);
template dgmr::vol_marcher::Gradients dgmr::vol_marcher::backward<double>(const whack::TensorView<const double, 3>& framebuffer, const ForwardData<double>& forward_data, const ForwardCache& forward_cache, const torch::Tensor& grad);
