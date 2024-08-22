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

#include "constants.h"
#include "math.h"
#include "piecewise_linear.h"
#include "raster_bin_sizers.h"
#include "util.h"
#include "vol_raster_forward.h"

#include <stroke/gaussian.h>
#include <stroke/linalg.h>
#include <whack/Tensor.h>
#include <whack/kernel.h>

using namespace dgmr;
using namespace vol_raster;
using namespace util;
namespace gaussian = stroke::gaussian;

dgmr::VolRasterStatistics dgmr::vol_raster_forward(VolRasterForwardData& data)
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

    // geometry buffers, filled by the preprocess pass
    auto g_rects_data = whack::make_tensor<glm::uvec2>(whack::Location::Device, n_gaussians);
    auto g_rects = g_rects_data.view();

    auto g_rgb_data = whack::make_tensor<glm::vec3>(whack::Location::Device, n_gaussians);
    auto g_rgb = g_rgb_data.view();

    auto g_rgb_sh_clamped_data = whack::make_tensor<glm::vec<3, bool>>(whack::Location::Device, n_gaussians);
    auto g_rgb_sh_clamped = g_rgb_sh_clamped_data.view();

    auto g_depths_data = whack::make_tensor<float>(whack::Location::Device, n_gaussians);
    auto g_depths = g_depths_data.view();

    auto g_points_xy_image_data = whack::make_tensor<glm::vec2>(whack::Location::Device, n_gaussians);
    auto g_points_xy_image = g_points_xy_image_data.view();

    auto g_inverse_filtered_cov3d_data = whack::make_tensor<stroke::Cov3<float>>(whack::Location::Device, n_gaussians);
    auto g_inverse_filtered_cov3d = g_inverse_filtered_cov3d_data.view();

    auto g_filtered_masses_data = whack::make_tensor<float>(whack::Location::Device, n_gaussians);
    auto g_filtered_masses = g_filtered_masses_data.view();

    auto g_tiles_touched_data = whack::make_tensor<uint32_t>(whack::Location::Device, n_gaussians);
    auto g_tiles_touched = g_tiles_touched_data.view();

    // preprocess, run per Gaussian
    {
        math::Camera<float> camera {
            data.view_matrix, data.proj_matrix, focal_x, focal_y, data.tan_fovx, data.tan_fovy, fb_width, fb_height
        };

        const dim3 block_dim = { 128 };
        const dim3 grid_dim = whack::grid_dim_from_total_size({ data.gm_weights.size<0>() }, block_dim);
        whack::start_parallel(
            whack::Location::Device, grid_dim, block_dim, WHACK_KERNEL(=) {
                WHACK_UNUSED(whack_gridDim);
                const auto idx = whack_blockIdx.x * whack_blockDim.x + whack_threadIdx.x;
                if (idx >= n_gaussians)
                    return;

                // Initialize touched tiles to 0. If this isn't changed,
                // this Gaussian will not be processed further.
                g_tiles_touched(idx) = 0;

                const auto centroid = data.gm_centroids(idx);
                if ((data.view_matrix * glm::vec4(centroid, 1.f)).z < 0.2) // adam doesn't understand, why projection matrix > 0 isn't enough.
                    return;

                const auto weights = data.gm_weights(idx);
                const auto scales = data.gm_cov_scales(idx) * data.cov_scale_multiplier;
                const auto rotations = data.gm_cov_rotations(idx);

                const auto screen_space_gaussian = math::splat<vol_raster::config::gaussian_mixture_formulation>(weights, centroid, scales, rotations, camera, 0.3f);

                const auto cov3d = math::compute_cov(scales, rotations);

                // low pass filter to combat aliasing
                const auto filter_kernel_size = glm::distance(centroid, data.cam_poition) * aa_distance_multiplier;
                const auto filtered_cov_3d = cov3d + stroke::Cov3_f(filter_kernel_size * filter_kernel_size);
                g_filtered_masses(idx) = math::weight_to_mass<vol_raster::config::gaussian_mixture_formulation>(weights, scales + glm::vec3(filter_kernel_size * filter_kernel_size), screen_space_gaussian.cov);

                // using the more aggressive computation for calculating overlapping tiles:
                {
                    const glm::uvec2 my_rect = { (int)ceil(3.f * sqrt(screen_space_gaussian.cov[0])), (int)ceil(3.f * sqrt(screen_space_gaussian.cov[2])) };
                    g_rects(idx) = my_rect;
                    glm::uvec2 rect_min, rect_max;
                    getRect(screen_space_gaussian.centroid, my_rect, &rect_min, &rect_max, render_grid_dim);

                    const auto tiles_touched = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
                    if (tiles_touched == 0)
                        return; // serves as clipping (i think)
                    g_tiles_touched(idx) = tiles_touched;
                    g_points_xy_image(idx) = screen_space_gaussian.centroid;
                }

                g_depths(idx) = glm::length(data.cam_poition - centroid);
                // g_depths(idx) = math::gaussian_to_point_distance_bounds(
                // centroid, data.gm_cov_scales(idx), data.gm_cov_rotations(idx), vol_raster::config::gaussian_relevance_sigma, data.cam_poition)
                // .max;

                const auto direction = glm::normalize(centroid - data.cam_poition);
                cuda::std::tie(g_rgb(idx), g_rgb_sh_clamped(idx)) = math::sh_to_colour(data.gm_sh_params(idx), data.sh_degree, direction);
                g_inverse_filtered_cov3d(idx) = stroke::inverse(filtered_cov_3d);
            });
    }

    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    auto g_point_offsets_data = whack::make_tensor<uint32_t>(whack::Location::Device, n_gaussians);
    auto g_point_offsets = g_point_offsets_data.view();
    {
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, g_tiles_touched_data.raw_pointer(), g_tiles_touched_data.raw_pointer(), n_gaussians);
        auto temp_storage = whack::make_tensor<char>(whack::Location::Device, temp_storage_bytes);

        cub::DeviceScan::InclusiveSum(temp_storage.raw_pointer(), temp_storage_bytes, g_tiles_touched_data.raw_pointer(), g_point_offsets_data.raw_pointer(), n_gaussians);
        CHECK_CUDA(data.debug);
    }

    const auto n_render_gaussians = unsigned(g_point_offsets_data.device_vector().back());
    if (n_render_gaussians == 0)
        return { 0 };

    // For each instance to be rendered, produce adequate [ tile | depth ] key
    // and corresponding dublicated Gaussian indices to be sorted
    auto b_point_list_keys_data = whack::make_tensor<uint64_t>(whack::Location::Device, n_render_gaussians);
    auto b_point_list_keys = b_point_list_keys_data.view();
    auto b_point_list_data = whack::make_tensor<uint32_t>(whack::Location::Device, n_render_gaussians);
    auto b_point_list = b_point_list_data.view();
    {
        auto b_point_list_keys_unsorted_data = whack::make_tensor<uint64_t>(whack::Location::Device, n_render_gaussians);
        auto b_point_list_keys_unsorted = b_point_list_keys_unsorted_data.view();
        auto b_point_list_unsorted_data = whack::make_tensor<uint32_t>(whack::Location::Device, n_render_gaussians);
        auto b_point_list_unsorted = b_point_list_unsorted_data.view();

        whack::start_parallel( // duplicateWithKeys
            whack::Location::Device, whack::grid_dim_from_total_size(n_gaussians, 256), 256, WHACK_KERNEL(=) {
                WHACK_UNUSED(whack_gridDim);
                const unsigned idx = whack_blockIdx.x * whack_blockDim.x + whack_threadIdx.x;
                if (idx >= n_gaussians)
                    return;

                // check for invisibility (e.g. outside the frustum)
                if (g_tiles_touched(idx) == 0)
                    return;

                // Find this Gaussian's offset in buffer for writing keys/values.
                uint32_t off = (idx == 0) ? 0 : g_point_offsets(idx - 1);
                glm::uvec2 rect_min, rect_max;
                getRect(g_points_xy_image(idx), g_rects(idx), &rect_min, &rect_max, render_grid_dim);

                // For each tile that the bounding rect overlaps, emit a
                // key/value pair. The key is |  tile ID  |      depth      |,
                // and the value is the ID of the Gaussian. Sorting the values
                // with this key yields Gaussian IDs in a list, such that they
                // are first sorted by tile and then by depth.
                for (unsigned y = rect_min.y; y < rect_max.y; y++) {
                    for (unsigned x = rect_min.x; x < rect_max.x; x++) {
                        uint64_t key = y * render_grid_dim.x + x;
                        key <<= 32;
                        key |= *((uint32_t*)&g_depths(idx)); // take the bits of a float

                        b_point_list_keys_unsorted(off) = key;
                        b_point_list_unsorted(off) = idx;
                        off++;
                    }
                }
            });

        const int bit = getHigherMsb(render_grid_dim.x * render_grid_dim.y);
        // Sort complete list of (duplicated) Gaussian indices by keys
        size_t temp_storage_bytes;
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            temp_storage_bytes,
            b_point_list_keys_unsorted_data.raw_pointer(), b_point_list_keys_data.raw_pointer(),
            b_point_list_unsorted_data.raw_pointer(), b_point_list_data.raw_pointer(),
            n_render_gaussians, 0, 32 + bit);
        auto temp_storage = whack::make_tensor<char>(whack::Location::Device, temp_storage_bytes);
        cub::DeviceRadixSort::SortPairs(
            temp_storage.raw_pointer(),
            temp_storage_bytes,
            b_point_list_keys_unsorted_data.raw_pointer(), b_point_list_keys_data.raw_pointer(),
            b_point_list_unsorted_data.raw_pointer(), b_point_list_data.raw_pointer(),
            n_render_gaussians, 0, 32 + bit);
        CHECK_CUDA(data.debug);
    }

    auto i_ranges_data = whack::make_tensor<glm::uvec2>(whack::Location::Device, render_grid_dim.y * render_grid_dim.x);
    {
        auto i_ranges = i_ranges_data.view();
        thrust::fill(i_ranges_data.device_vector().begin(), i_ranges_data.device_vector().end(), glm::uvec2(0));

        whack::start_parallel( // identifyTileRanges
            whack::Location::Device, whack::grid_dim_from_total_size(n_render_gaussians, 256), 256, WHACK_KERNEL(=) {
                WHACK_UNUSED(whack_gridDim);
                const unsigned idx = whack_blockIdx.x * whack_blockDim.x + whack_threadIdx.x;
                if (idx >= n_render_gaussians)
                    return;

                // Read tile ID from key. Update start/end of tile range if at limit.
                uint64_t key = b_point_list_keys(idx);
                uint32_t currtile = key >> 32;
                if (idx == 0)
                    i_ranges(currtile).x = 0;
                else {
                    uint32_t prevtile = b_point_list_keys(idx - 1) >> 32;
                    if (currtile != prevtile) {
                        i_ranges(prevtile).y = idx;
                        i_ranges(currtile).x = idx;
                    }
                }
                if (idx == n_render_gaussians - 1)
                    i_ranges(currtile).y = n_render_gaussians;
            });
    }

    // render
    // Let each tile blend its range of Gaussians independently in parallel
    {
        // Main rasterization method. Collaboratively works on one tile per
        // block, each thread treats one pixel. Alternates between fetching
        // and rasterizing data.

        const auto inversed_projectin_matrix = glm::inverse(data.proj_matrix);

        auto i_ranges = whack::make_tensor_view(i_ranges_data.device_vector(), render_grid_dim.y, render_grid_dim.x);
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

                const auto render_g_range = i_ranges(whack_blockIdx.y, whack_blockIdx.x);
                const auto n_rounds = ((render_g_range.y - render_g_range.x + render_block_size - 1) / render_block_size);
                auto n_toDo = render_g_range.y - render_g_range.x;

                // Allocate storage for batches of collectively fetched data.
                __shared__ int collected_id[render_block_size];
                __shared__ float collected_3d_masses[render_block_size];
                __shared__ glm::vec3 collected_centroid[render_block_size];
                __shared__ stroke::Cov3<float> collected_inv_cov3[render_block_size];

                whack::Array<glm::vec4, vol_raster::config::n_rasterisation_bins> bin_mass = {};
                whack::Array<glm::vec4, vol_raster::config::n_rasterisation_bins + 1> bin_eval = {};
                math::RasterBinSizer<vol_raster::config> rasterisation_bin_sizer;

                // Iterate over batches until all done or range is complete: compute max depth
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
                        const auto gaussian1d = gaussian::intersect_with_ray_inv_C(collected_centroid[j], collected_inv_cov3[j], ray);
                        auto mass_on_ray = gaussian1d.weight * collected_3d_masses[j];
                        // if (vol_raster::config::use_orientation_dependent_gaussian_density)
                        // weight *= gaussian::norm_factor(gaussian1d.C) / gaussian::norm_factor_inv_C(collected_inv_cov3[j]);

                        if (mass_on_ray < 0.001 || mass_on_ray > 1'000)
                            continue;
                        if (gaussian1d.C + vol_raster::config::workaround_variance_add_along_ray <= 0)
                            continue;
                        rasterisation_bin_sizer.add_gaussian(mass_on_ray, gaussian1d.centre, stroke::sqrt(gaussian1d.C + vol_raster::config::workaround_variance_add_along_ray));
                        if (rasterisation_bin_sizer.is_full()) {
                            // printf("done at %i / %i\n", i * render_block_size + j, render_g_range.y - render_g_range.x);
                            done = true;
                            break;
                        }
                    }
                }

                __syncthreads();

                rasterisation_bin_sizer.finalise();
                done = !inside;
                n_toDo = render_g_range.y - render_g_range.x;
                // float opacity = 1;

                // Iterate over batches until all done or range is complete: rasterise into bins
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
                    // block.sync();
                    __syncthreads();

                    if (done)
                        continue;

                    // Iterate over current batch
                    for (unsigned j = 0; j < min(render_block_size, n_toDo); j++) {
                        const auto inv_cov = collected_inv_cov3[j];
                        const auto gaussian1d = gaussian::intersect_with_ray_inv_C(collected_centroid[j], inv_cov, ray);
                        const auto centroid = gaussian1d.centre;
                        const auto variance = gaussian1d.C + vol_raster::config::workaround_variance_add_along_ray;
                        const auto sd = stroke::sqrt(variance);
                        const auto inv_sd = 1 / sd;
                        auto weight = gaussian1d.weight * collected_3d_masses[j];
                        // if (vol_raster::config::use_orientation_dependent_gaussian_density)
                        // weight *= gaussian::norm_factor(gaussian1d.C) / gaussian::norm_factor_inv_C(collected_inv_cov3[j]);

                        if (variance <= 0 || stroke::isnan(variance) || stroke::isnan(weight) || weight > 100'000)
                            continue; // todo: shouldn't happen any more after implementing AA?
                        if (!(weight >= 0 && weight < 100'000)) {
                            printf("weight: %f, gaussian1d.C: %f, collected_cov3[j]: %f/%f/%f/%f/%f/%f, det: %f\n", weight, variance, inv_cov[0], inv_cov[1], inv_cov[2], inv_cov[3], inv_cov[4], inv_cov[5], det(inv_cov));
                            //							printf("weight: %f, gaussian1d.weight: %f, collected_weight[j]: %f, stroke::gaussian::norm_factor(gaussian1d.C): %f, gaussian1d.C: %f\n", weight, gaussian1d.weight, collected_weight[j], stroke::gaussian::norm_factor(gaussian1d.C), gaussian1d.C);
                        }
                        if (weight * gaussian::integrate_normalised_inv_SD(centroid, inv_sd, { 0, rasterisation_bin_sizer.max_distance() }) <= 0.001f) { // performance critical
                            continue;
                        }

                        auto cdf_start = gaussian::cdf_inv_SD(centroid, inv_sd, 0.f);
                        for (auto k = 0u; k < vol_raster::config::n_rasterisation_bins; ++k) {
                            const auto cdf_end = gaussian::cdf_inv_SD(centroid, inv_sd, rasterisation_bin_sizer.end_of(k));
                            const auto mass = (cdf_end - cdf_start) * weight;
                            cdf_start = cdf_end;
                            if (mass < 0.0001f)
                                continue;
                            //                            const auto integrated = weight * gaussian::integrate_inv_SD(centroid, inv_sd, { rasterisation_bin_sizer.begin_of(k), rasterisation_bin_sizer.end_of(k) });
                            bin_mass[k] += glm::vec4(g_rgb(collected_id[j]) * mass, mass);
                            const auto eval = weight * gaussian::eval_exponential(centroid, variance, rasterisation_bin_sizer.begin_of(k));
                            bin_eval[k] += glm::vec4(g_rgb(collected_id[j]) * eval, eval);
                        }
                        const auto eval = weight * gaussian::eval_exponential(centroid, variance, rasterisation_bin_sizer.end_of(vol_raster::config::n_rasterisation_bins - 1));
                        bin_eval[vol_raster::config::n_rasterisation_bins] += glm::vec4(g_rgb(collected_id[j]) * eval, eval);

                        // terminates too early when an intense gaussian is ordered behind many less intense ones, but its location is in front
                        // it does produce artefacts, e.g. onion rings in the hohe veitsch scene.
                        // it also gives a performance boost, but only in combination with opacity filtering.
                        // todo: in the future, we should order using the front side, and stop once we compute front sides behind max_distance.
                        //                        opacity *= 1 - (cdf_start - gaussian::cdf_inv_C(centroid, inv_variance, 0.f)) * weight; // end - start again, but here cdf_tart refers to the actual end.
                        //                        if (opacity < 0.0001f) {
                        //                            done = true;
                        //                            break;
                        //                        }
                    }
                }

                // blend
                float T = 1.0f;
                glm::vec3 C = glm::vec3(0);
                switch (data.debug_render_mode) {
                case VolRasterForwardData::RenderMode::Full: {

                    glm::vec<3, float> result = {};
                    // float current_m = 0;
                    float current_transparency = 1;
                    for (auto k = 0u; k < vol_raster::config::n_rasterisation_bins; ++k) {
                        // auto current_bin = rasterised_data[k];
                        // for (auto i = 0; i < 3; ++i) {
                        //     current_bin[i] /= current_bin[3] + 0.001f; // make an weighted average out of a weighted sum
                        // }
                        // // Avoid numerical instabilities (see paper appendix).
                        // float alpha = min(0.99f, current_bin[3]);
                        // if (alpha < 1.0f / 255.0f)
                        //     continue;

                        // float test_T = T * (1 - alpha);
                        // if (test_T < 0.0001f) {
                        //     done = true;
                        //     break;
                        // }

                        // // Eq. (3) from 3D Gaussian splatting paper.
                        // C += glm::vec3(current_bin) * alpha * T;

                        // T = test_T;

                        const auto t_right = rasterisation_bin_sizer.end_of(k) - rasterisation_bin_sizer.begin_of(k);
                        if (t_right <= 0)
                            continue;
                        auto percent_left = glm::vec4(rasterisation_bin_sizer.border_mass_begin_of(k) / (rasterisation_bin_sizer.border_mass_begin_of(k) + rasterisation_bin_sizer.border_mass_begin_of(k)));
                        if (stroke::isnan(percent_left.w)) {
                            percent_left = glm::vec4(0.5f);
                        }
                        if (k > 0 && k < vol_raster::config::n_rasterisation_bins - 1) {
                            assert(percent_left.x == percent_left.y);
                            assert(percent_left.x == percent_left.z);
                            assert(percent_left.x == percent_left.w);
                        }
                        // if (k == 0 || k == vol_raster::config::n_rasterisation_steps - 1)
                        //     percent_left = bin_eval[k] / (bin_eval[k] + bin_eval[k + 1]);
                        // if (isnan(percent_left)) {
                        //     percent_left = glm::vec4(0.5f);
                        // }
                        // const auto percent_left = (bin_eval[k] / (bin_eval[k] + bin_eval[k + 1]));
                        // const auto percent_left = glm::vec4(0.5f, 0.5f, 0.5f, 0.5f);
                        const auto f = dgmr::piecewise_linear::create_approximation(percent_left, bin_mass[k], bin_eval[k], bin_eval[k + 1], t_right);

                        const auto delta_t = f.t_right / vol_raster::config::n_steps_per_bin;
                        float t = delta_t / 2;
                        for (unsigned i = 0; i < vol_raster::config::n_steps_per_bin; ++i) {
                            assert(t < f.t_right);
                            const auto eval_t = f.sample(t);
                            // result += glm::dvec3(eval_t) * stroke::exp(-current_m) * delta_t;
                            result += glm::vec<3, float>(eval_t) * current_transparency * delta_t;
                            // current_m += eval_t.w * delta_t;
                            current_transparency *= stroke::max(float(0), 1 - eval_t.w * delta_t);
                            t += delta_t;
                        }
                        C = result;
                    }
                    break;
                }
                case VolRasterForwardData::RenderMode::Bins: {
                    // const auto bin = stroke::min(unsigned(data.debug_render_bin), config::n_rasterisation_steps - 1);
                    // auto current_bin = rasterised_data[bin];
                    // for (auto i = 0; i < 3; ++i) {
                    //     current_bin[i] /= current_bin[3] + 0.001f; // make an weighted average out of a weighted sum
                    // }
                    // float alpha = min(0.99f, current_bin[3]);
                    // float test_T = T * (1 - alpha);
                    // C += glm::vec3(current_bin) * alpha * T;

                    // T = test_T;
                    break;
                }
                case VolRasterForwardData::RenderMode::Depth: {
                    const auto bin = stroke::min(unsigned(data.debug_render_bin), config::n_rasterisation_bins - 1);
                    const auto distance = rasterisation_bin_sizer.end_of(bin);
                    C = glm::vec3(distance / data.max_depth);
                    if (distance == 0)
                        C = glm::vec3(0, 1.0, 0);
                    if (stroke::isnan(distance))
                        C = glm::vec3(1, 0, 0.5);
                    if (distance < 0)
                        C = glm::vec3(1, 0.5, 0);
                    T = 0;
                    break;
                }
                }

                if (!inside)
                    return;
                // All threads that treat valid pixel write out their final
                const auto final_colour = C + T * data.background;
                data.framebuffer(0, pix.y, pix.x) = final_colour.x;
                data.framebuffer(1, pix.y, pix.x) = final_colour.y;
                data.framebuffer(2, pix.y, pix.x) = final_colour.z;
            });
    }

    return { 0 };
}
