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

#include "vol_marcher_forward.h"

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <stroke/gaussian.h>
#include <stroke/linalg.h>
#include <whack/Tensor.h>
#include <whack/kernel.h>
#include <whack/torch_interop.h>

#include "marching_steps.h"
#include "math.h"
#include "util.h"

namespace {
using namespace dgmr;
using namespace dgmr::vol_marcher;
namespace gaussian = stroke::gaussian;

} // namespace


template <typename scalar_t>
dgmr::vol_marcher::ForwardCache dgmr::vol_marcher::forward(whack::TensorView<scalar_t, 3> framebuffer, vol_marcher::ForwardData<scalar_t>& data)
{
    using Vec2 = glm::vec<2, scalar_t>;
    using Vec3 = glm::vec<3, scalar_t>;
    using Vec4 = glm::vec<4, scalar_t>;
    using Mat4 = glm::mat<4, 4, scalar_t>;
    using Cov3 = stroke::Cov3<scalar_t>;

    const auto torch_float_type = (sizeof(scalar_t) == 4) ? torch::kFloat32 : torch::kFloat64;
    const auto fb_width = framebuffer.template size<2>();
    const auto fb_height = framebuffer.template size<1>();
    const auto n_gaussians = data.gm_weights.template size<0>();
    const auto focal_y = fb_height / (2.0f * data.tan_fovy);
    const auto focal_x = fb_width / (2.0f * data.tan_fovx);
    const auto aa_distance_multiplier = (config::filter_kernel_SD * data.tan_fovx * 2) / fb_width;

    constexpr dim3 render_block_dim = { render_block_width, render_block_height };
    constexpr auto render_block_size = render_block_width * render_block_height;
    constexpr auto render_n_warps = render_block_size / 32;
    static_assert(render_n_warps * 32 == render_block_size);
    const dim3 render_grid_dim = whack::grid_dim_from_total_size({ fb_width, fb_height }, render_block_dim);

    // geometry buffers, filled by the preprocess pass
    dgmr::vol_marcher::ForwardCache cache;
    cache.rects = torch::empty({ n_gaussians, 2 }, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    auto g_rects = whack::make_tensor_view<glm::uvec2>(cache.rects, n_gaussians);

    cache.rgb = torch::empty({ n_gaussians, 3 }, torch::TensorOptions().dtype(torch_float_type).device(torch::kCUDA));
    auto g_rgb = whack::make_tensor_view<Vec3>(cache.rgb, n_gaussians);

    cache.rgb_sh_clamped = torch::empty({ n_gaussians, 3 }, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    auto g_rgb_sh_clamped = whack::make_tensor_view<glm::vec<3, bool>>(cache.rgb_sh_clamped, n_gaussians);

    cache.depths = torch::empty({ n_gaussians }, torch::TensorOptions().dtype(torch_float_type).device(torch::kCUDA));
    auto g_depths = whack::make_tensor_view<scalar_t>(cache.depths, n_gaussians);

    cache.points_xy_image = torch::empty({ n_gaussians, 2 }, torch::TensorOptions().dtype(torch_float_type).device(torch::kCUDA));
    auto g_points_xy_image = whack::make_tensor_view<Vec2>(cache.points_xy_image, n_gaussians);

    cache.inverse_filtered_cov3d = torch::empty({ n_gaussians, 6 }, torch::TensorOptions().dtype(torch_float_type).device(torch::kCUDA));
    auto g_inverse_filtered_cov3d = whack::make_tensor_view<Cov3>(cache.inverse_filtered_cov3d, n_gaussians);

    cache.filtered_masses = torch::empty({ n_gaussians }, torch::TensorOptions().dtype(torch_float_type).device(torch::kCUDA));
    auto g_filtered_masses = whack::make_tensor_view<scalar_t>(cache.filtered_masses, n_gaussians);

    cache.tiles_touched = torch::empty({ n_gaussians }, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    auto g_tiles_touched = whack::make_tensor_view<uint32_t>(cache.tiles_touched, n_gaussians);

    cache.point_offsets = torch::empty({ n_gaussians }, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    auto g_point_offsets = whack::make_tensor_view<uint32_t>(cache.point_offsets, n_gaussians);

    cache.remaining_transparency = torch::empty({ fb_height, fb_width }, torch::TensorOptions().dtype(torch_float_type).device(torch::kCUDA));
    auto remaining_transparency = whack::make_tensor_view<scalar_t>(cache.remaining_transparency, fb_height, fb_width);

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

                // Initialize touched tiles to 0. If this isn't changed,
                // this Gaussian will not be processed further.
                g_tiles_touched(idx) = 0;

                const auto centroid = data.gm_centroids(idx);
                if ((data.view_matrix * Vec4(centroid, 1.f)).z < 0.2f)
                    return;

                const auto weight = data.gm_weights(idx);
                const auto scales = data.gm_cov_scales(idx) * data.cov_scale_multiplier;
                const auto rotation = data.gm_cov_rotations(idx);
                const auto dist = glm::length(data.cam_poition - centroid);


                const auto cov3d = math::compute_cov(scales, rotation);

                // low pass filter to combat aliasing
                const auto filter_kernel_size = dist * aa_distance_multiplier;
                const auto filtered_cov_3d = cov3d + Cov3(filter_kernel_size * filter_kernel_size);
                const auto filtered_scales = scales + Vec3(filter_kernel_size * filter_kernel_size);
                const auto mass = math::weight_to_mass<vol_marcher::config::gaussian_mixture_formulation>(weight, filtered_scales);
                if (mass <= 0)
                    return; // clipped

                // using the more aggressive computation for calculating overlapping tiles:
                {
                    const auto screen_space_gaussian = math::splat<vol_marcher::config::gaussian_mixture_formulation>(weight, centroid, scales, rotation, camera, scalar_t(config::filter_kernel_SD * config::filter_kernel_SD));
                    // get exact distance to 1./255. isoline
                    const auto isoline_distance = [](scalar_t w, scalar_t variance, scalar_t isoline) {
                        // solve w * gaussian(x, sd) == isoline for x
                        const auto s = -2 * stroke::log(isoline / w);
                        if (s <= 0)
                            return scalar_t(0);
                        return stroke::sqrt(s * variance);
                    };
                    // const glm::uvec2 my_rect = { (int)ceil(3.f * sqrt(screen_space_gaussian.cov[0])), (int)ceil(3.f * sqrt(screen_space_gaussian.cov[2])) };
                    const glm::uvec2 my_rect = {
                        int(ceil(isoline_distance(screen_space_gaussian.weight, screen_space_gaussian.cov[0], 1.f / 255.f))),
                        int(ceil(isoline_distance(screen_space_gaussian.weight, screen_space_gaussian.cov[2], 1.f / 255.f))),
                    };
                    g_rects(idx) = my_rect;
                    glm::uvec2 rect_min, rect_max;
                    util::getRect(screen_space_gaussian.centroid, my_rect, &rect_min, &rect_max, render_grid_dim);

                    const auto tiles_touched = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
                    if (tiles_touched == 0)
                        return; // clipped
                    g_tiles_touched(idx) = tiles_touched;
                    g_points_xy_image(idx) = screen_space_gaussian.centroid;
                }

                const auto inverse_filtered_cov = stroke::inverse(filtered_cov_3d);

                // g_depths(idx) = dist;
                g_depths(idx) = (dist - math::max(scales) * config::gaussian_relevance_sigma / 2);

                const auto direction = (centroid - data.cam_poition) / dist;
                cuda::std::tie(g_rgb(idx), g_rgb_sh_clamped(idx)) = math::sh_to_colour(data.gm_sh_params(idx), data.sh_degree, direction);
                g_inverse_filtered_cov3d(idx) = inverse_filtered_cov;
                g_filtered_masses(idx) = mass;
            });
    }

    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    {
        size_t temp_storage_bytes = 0;
        auto tiles_touched_ptr = whack::raw_pointer<uint32_t>(cache.tiles_touched);
        auto point_offsets_ptr = whack::raw_pointer<uint32_t>(cache.point_offsets);

        cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, tiles_touched_ptr, point_offsets_ptr, n_gaussians);
        auto temp_storage = torch::empty(temp_storage_bytes, torch::TensorOptions().dtype(torch::kChar).device(torch::kCUDA));
        auto temp_storage_ptr = whack::raw_pointer<char>(temp_storage);

        cub::DeviceScan::InclusiveSum(temp_storage_ptr, temp_storage_bytes, tiles_touched_ptr, point_offsets_ptr, n_gaussians);
        CHECK_CUDA(data.debug);
    }

    const auto n_render_gaussians = unsigned(cache.point_offsets[n_gaussians - 1].template item<int>());
    if (n_render_gaussians == 0)
        return {};

    // For each instance to be rendered, produce adequate [ tile | depth ] key
    // and corresponding dublicated Gaussian indices to be sorted
    auto b_point_list_keys_data = torch::empty({ n_render_gaussians }, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto b_point_list_keys = whack::make_tensor_view<uint64_t>(b_point_list_keys_data, n_render_gaussians);

    cache.b_point_list = torch::empty({ n_render_gaussians }, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto b_point_list = whack::make_tensor_view<uint32_t>(cache.b_point_list, n_render_gaussians);
    {
        auto b_point_list_keys_unsorted_data = torch::empty({ n_render_gaussians }, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
        auto b_point_list_keys_unsorted = whack::make_tensor_view<uint64_t>(b_point_list_keys_unsorted_data, n_render_gaussians);

        auto b_point_list_unsorted_data = torch::empty({ n_render_gaussians }, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        auto b_point_list_unsorted = whack::make_tensor_view<uint32_t>(b_point_list_unsorted_data, n_render_gaussians);

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
                util::getRect(g_points_xy_image(idx), g_rects(idx), &rect_min, &rect_max, render_grid_dim);

                // For each tile that the bounding rect overlaps, emit a
                // key/value pair. The key is |  tile ID  |      depth      |,
                // and the value is the ID of the Gaussian. Sorting the values
                // with this key yields Gaussian IDs in a list, such that they
                // are first sorted by tile and then by depth.
                for (unsigned y = rect_min.y; y < rect_max.y; y++) {
                    for (unsigned x = rect_min.x; x < rect_max.x; x++) {
                        uint64_t key = y * render_grid_dim.x + x;
                        key <<= 32;
                        float depth_f = float(g_depths(idx));
                        key |= *((uint32_t*)&depth_f); // take the bits of a float

                        b_point_list_keys_unsorted(off) = key;
                        b_point_list_unsorted(off) = idx;
                        off++;
                    }
                }
            });

        using whack::raw_pointer;
        const int bit = util::getHigherMsb(render_grid_dim.x * render_grid_dim.y);
        // Sort complete list of (duplicated) Gaussian indices by keys
        size_t temp_storage_bytes;
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            temp_storage_bytes,
            raw_pointer<uint64_t>(b_point_list_keys_unsorted_data), raw_pointer<uint64_t>(b_point_list_keys_data),
            raw_pointer<uint32_t>(b_point_list_unsorted_data), raw_pointer<uint32_t>(cache.b_point_list),
            n_render_gaussians, 0, 32 + bit);

        auto temp_storage = torch::empty(temp_storage_bytes, torch::TensorOptions().dtype(torch::kChar).device(torch::kCUDA));
        cub::DeviceRadixSort::SortPairs(
            raw_pointer<void>(temp_storage),
            temp_storage_bytes,
            raw_pointer<uint64_t>(b_point_list_keys_unsorted_data), raw_pointer<uint64_t>(b_point_list_keys_data),
            raw_pointer<uint32_t>(b_point_list_unsorted_data), raw_pointer<uint32_t>(cache.b_point_list),
            n_render_gaussians, 0, 32 + bit);
        CHECK_CUDA(data.debug);
    }

    cache.i_ranges = torch::zeros({ render_grid_dim.y * render_grid_dim.x, 2 }, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    {
        auto i_ranges = whack::make_tensor_view<glm::uvec2>(cache.i_ranges, render_grid_dim.y * render_grid_dim.x);

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
        auto i_ranges = whack::make_tensor_view<glm::uvec2>(cache.i_ranges, render_grid_dim.y, render_grid_dim.x);
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

                const auto render_g_range = i_ranges(whack_blockIdx.y, whack_blockIdx.x);
                const auto n_rounds = ((render_g_range.y - render_g_range.x + render_block_size - 1) / render_block_size);
                auto n_toDo = render_g_range.y - render_g_range.x;

                // Allocate storage for batches of collectively fetched data.
                __shared__ int collected_id[render_block_size];
                __shared__ scalar_t collected_3d_masses[render_block_size];
                __shared__ Vec3 collected_centroid[render_block_size];
                // __shared__ Cov3 collected_inv_cov3[render_block_size];

                bool large_stepping_ongoing = true;
                scalar_t current_large_step_start = 0.f;

                Vec3 current_colour = Vec3(0);
                scalar_t current_transparency = 1;
                scalar_t distance_marched_tmp = 0;

                while (large_stepping_ongoing) {
                    // Iterate over all gaussians and compute sample_sections
                    marching_steps::DensityArray<config::n_large_steps, scalar_t> sample_sections(current_large_step_start);
#ifdef DGMR_TORCH_GRAD_CHECK_CONST_SAMPLES
                    // if you want to check the gradient without sampling, first run in forward with DGMR_PRINT_G_DENSITIES enabled
                    // and then add the Gaussians manually here. That will fix the sampling positions when computing the numerical gradient
                    // with the central difference method.
                    // define DGMR_TORCH_GRDA_CHECK_CONST_SAMPLES in a place, such that it's also seen in vol_marcher_backward.cu!
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
                            // collected_inv_cov3[thread_rank] = g_inverse_filtered_cov3d(coll_id);
                            collected_3d_masses[thread_rank] = g_filtered_masses(coll_id);
                        }
                        __syncthreads();

                        if (done || done_1)
                            continue;

                        // Iterate over current batch
                        // don't forget about the backward pass when editing this loop.
                        // think about moving to a function.
                        for (unsigned j = 0; j < stroke::min(render_block_size, n_toDo); j++) {
                            const auto gaussian1d = gaussian::intersect_with_ray_inv_C(collected_centroid[j], g_inverse_filtered_cov3d(collected_id[j]), ray);
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
#ifdef DGMR_PRINT_G_DENSITIES
                            printf("bins start: %f, end: %f, delta_t: %f\n", float(start), float(end), float(delta_t));
#endif
                        }
                    }
#endif

                    // compute sampling
                    const auto bin_borders = marching_steps::sample<config::n_small_steps>(sample_sections);
                    whack::Array<Vec4, config::n_small_steps - 1> bin_eval = {};

                    // float dbg_mass_in_bins_closeed = 0;
                    // float dbg_mass_in_bins_numerik_1 = 0;

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
                            // collected_inv_cov3[thread_rank] = g_inverse_filtered_cov3d(coll_id);
                            collected_3d_masses[thread_rank] = g_filtered_masses(coll_id);
                        }
                        __syncthreads();

                        if (done)
                            continue;

                        // Iterate over current batch
                        for (unsigned j = 0; j < stroke::min(render_block_size, n_toDo); j++) {
                            math::sample_gaussian(collected_3d_masses[j], g_rgb(collected_id[j]), collected_centroid[j], g_inverse_filtered_cov3d(collected_id[j]), ray, bin_borders, &bin_eval);
                        }
                    }

                    // blend
                    switch (data.debug_render_mode) {
                    case vol_marcher::ForwardData<scalar_t>::RenderMode::Full: {
                        cuda::std::tie(current_colour, current_transparency) = math::integrate_bins(current_colour, current_transparency, bin_eval);
                        break;
                    }
                    case vol_marcher::ForwardData<scalar_t>::RenderMode::Bins: {
                        const auto bin = stroke::min(unsigned(data.debug_render_bin), bin_eval.size() - 1);
                        const auto mass = sum(bin_eval[bin]);
                        // const auto mass = (bin == 0) ? dbg_mass_in_bins_closeed : dbg_mass_in_bins_numerik_1;
                        current_colour = Vec3(mass * data.max_depth);
                        if (mass == 0)
                            current_colour = Vec3(0, 1.0, 0);
                        if (stroke::isnan(mass))
                            current_colour = Vec3(1, 0, 0.5);
                        if (mass < 0)
                            current_colour = Vec3(1, 0.0, 0);
                        current_transparency = 0;
                        break;
                    }
                    case vol_marcher::ForwardData<scalar_t>::RenderMode::Depth: {
                        for (auto k = 0u; k < bin_eval.size(); ++k) {
                            const auto eval_t = bin_eval[k];
                            current_transparency *= stroke::exp(-eval_t.w);
                        }
                        distance_marched_tmp = stroke::max(sample_sections.end(), distance_marched_tmp);
                        current_colour = Vec3(distance_marched_tmp / data.max_depth, distance_marched_tmp / data.max_depth, distance_marched_tmp / data.max_depth);
                        // const auto bin = stroke::min(unsigned(data.debug_render_bin), sample_sections.size() - 1);
                        // const auto distance = sample_sections[bin].end;
                        // // const auto bin = stroke::min(unsigned(data.debug_render_bin), current_large_steps.size() - 1);
                        // // const auto distance = current_large_steps.data()[bin];
                        // current_colour = glm::vec3(distance / data.max_depth);
                        // if (distance == 0)
                        //     current_colour = glm::vec3(0, 1.0, 0);
                        // if (stroke::isnan(distance))
                        //     current_colour = glm::vec3(1, 0, 0.5);
                        // if (distance < 0)
                        //     current_colour = glm::vec3(1, 0.5, 0);
                        // current_transparency = 0;
                        break;
                    }
                    }

                    done = done || sample_sections.size() == 0 || current_transparency < 1.f / 255.f;
                    const int num_done = __syncthreads_count(done);
                    if (num_done == render_block_size)
                        break;
                    current_large_step_start = bin_borders[bin_borders.size() - 1];
                }

                if (!inside)
                    return;
                // All threads that treat valid pixel write out their final
                const auto final_colour = current_colour + current_transparency * data.background;
                framebuffer(0, pix.y, pix.x) = final_colour.x;
                framebuffer(1, pix.y, pix.x) = final_colour.y;
                framebuffer(2, pix.y, pix.x) = final_colour.z;
                remaining_transparency(pix.y, pix.x) = current_transparency;
            });
    }
    return cache;
}


template dgmr::vol_marcher::ForwardCache dgmr::vol_marcher::forward<float>(whack::TensorView<float, 3> framebuffer, vol_marcher::ForwardData<float>& forward_data);
template dgmr::vol_marcher::ForwardCache dgmr::vol_marcher::forward<double>(whack::TensorView<double, 3> framebuffer, vol_marcher::ForwardData<double>& forward_data);
