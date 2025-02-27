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
#include <whack/kernel.h>
#include <whack/torch_interop.h>

#include "constants.h"
#include "math.h"
#include "splat_forward.h"
#include "util.h"

dgmr::Statistics dgmr::splat_forward(SplatForwardData& data)
{
    const auto fb_width = data.framebuffer.size<2>();
    const auto fb_height = data.framebuffer.size<1>();
    const auto n_gaussians = data.gm_weights.size<0>();
    const float focal_y = fb_height / (2.0f * data.tan_fovy);
    const float focal_x = fb_width / (2.0f * data.tan_fovx);

    constexpr dim3 render_block_dim = { render_block_width, render_block_height };
    constexpr auto render_block_size = render_block_width * render_block_height;
    constexpr auto render_n_warps = render_block_size / 32;
    static_assert(render_n_warps * 32 == render_block_size);
    const dim3 render_grid_dim = whack::grid_dim_from_total_size({ data.framebuffer.size<2>(), data.framebuffer.size<1>() }, render_block_dim);

    // geometry buffers, filled by the preprocess pass
    auto g_rects_data = torch::empty({ n_gaussians, 2 }, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    auto g_rects = whack::make_tensor_view<glm::uvec2>(g_rects_data, n_gaussians);

    auto g_rgb_data = torch::empty({ n_gaussians, 3 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto g_rgb = whack::make_tensor_view<glm::vec3>(g_rgb_data, n_gaussians);

    auto g_rgb_sh_clamped_data = torch::empty({ n_gaussians, 3 }, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    auto g_rgb_sh_clamped = whack::make_tensor_view<glm::vec<3, bool>>(g_rgb_sh_clamped_data, n_gaussians);

    auto g_depths_data = torch::empty({ n_gaussians, 1 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto g_depths = whack::make_tensor_view<float>(g_depths_data, n_gaussians);

    auto g_points_xy_image_data = torch::empty({ n_gaussians, 2 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto g_points_xy_image = whack::make_tensor_view<glm::vec2>(g_points_xy_image_data, n_gaussians);

    auto g_conic_opacity_data = torch::empty({ n_gaussians, 4 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto g_conic_opacity = whack::make_tensor_view<ConicAndOpacity>(g_conic_opacity_data, n_gaussians);

    auto g_tiles_touched_data = torch::empty({ n_gaussians, 1 }, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto g_tiles_touched = whack::make_tensor_view<uint32_t>(g_tiles_touched_data, n_gaussians);

    math::Camera<float> camera {
        data.view_matrix, data.proj_matrix, focal_x, focal_y, data.tan_fovx, data.tan_fovy, fb_width, fb_height
    };

    // preprocess, run per Gaussian
    {
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

                const auto screen_space_gaussian = math::splat<splat::config::gaussian_mixture_formulation>(weights, centroid, scales, rotations, camera, 0.3f);

                // using the more aggressive computation for calculating overlapping tiles:
                {
                    const glm::uvec2 my_rect = { (int)ceil(3.f * sqrt(screen_space_gaussian.cov[0])), (int)ceil(3.f * sqrt(screen_space_gaussian.cov[2])) };
                    g_rects(idx) = my_rect;
                    glm::uvec2 rect_min, rect_max;
                    util::getRect(screen_space_gaussian.centroid, my_rect, &rect_min, &rect_max, render_grid_dim);

                    const auto tiles_touched = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
                    if (tiles_touched == 0)
                        return; // serves as clipping (i think)
                    g_tiles_touched(idx) = tiles_touched;
                    g_points_xy_image(idx) = screen_space_gaussian.centroid;
                }

                g_depths(idx) = glm::length(data.cam_poition - centroid);

                const auto direction = glm::normalize(centroid - data.cam_poition);
                cuda::std::tie(g_rgb(idx), g_rgb_sh_clamped(idx)) = math::sh_to_colour(data.gm_sh_params(idx), data.sh_degree, direction);

                // Inverse 2D covariance and opacity neatly pack into one float4
                const auto conic2d = inverse(screen_space_gaussian.cov);
                g_conic_opacity(idx) = { conic2d, screen_space_gaussian.weight };
            });
    }

    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    auto g_point_offsets_data = torch::empty({ n_gaussians, 1 }, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto g_point_offsets = whack::make_tensor_view<uint32_t>(g_point_offsets_data, n_gaussians);
    {
        using whack::raw_pointer;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(
            nullptr, temp_storage_bytes,
            raw_pointer<uint32_t>(g_tiles_touched_data), raw_pointer<uint32_t>(g_tiles_touched_data),
            n_gaussians);

        auto temp_storage = torch::empty(temp_storage_bytes, torch::TensorOptions().dtype(torch::kChar).device(torch::kCUDA));

        cub::DeviceScan::InclusiveSum(
            raw_pointer<char>(temp_storage), temp_storage_bytes,
            raw_pointer<uint32_t>(g_tiles_touched_data), raw_pointer<uint32_t>(g_point_offsets_data),
            n_gaussians);
        CHECK_CUDA(data.debug);
    }

    const auto n_render_gaussians = unsigned(g_point_offsets_data[n_gaussians - 1].item<int>());
    if (n_render_gaussians == 0)
        return { 0 };

    // For each instance to be rendered, produce adequate [ tile | depth ] key
    // and corresponding dublicated Gaussian indices to be sorted
    auto b_point_list_keys_data = torch::empty({ n_render_gaussians }, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto b_point_list_keys = whack::make_tensor_view<uint64_t>(b_point_list_keys_data, n_render_gaussians);

    auto b_point_list_data = torch::empty({ n_render_gaussians }, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto b_point_list = whack::make_tensor_view<uint32_t>(b_point_list_data, n_render_gaussians);
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
                        key |= *((uint32_t*)&g_depths(idx)); // take the bits of a float

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
            raw_pointer<uint32_t>(b_point_list_unsorted_data), raw_pointer<uint32_t>(b_point_list_data),
            n_render_gaussians, 0, 32 + bit);

        auto temp_storage = torch::empty(temp_storage_bytes, torch::TensorOptions().dtype(torch::kChar).device(torch::kCUDA));
        cub::DeviceRadixSort::SortPairs(
            raw_pointer<void>(temp_storage),
            temp_storage_bytes,
            raw_pointer<uint64_t>(b_point_list_keys_unsorted_data), raw_pointer<uint64_t>(b_point_list_keys_data),
            raw_pointer<uint32_t>(b_point_list_unsorted_data), raw_pointer<uint32_t>(b_point_list_data),
            n_render_gaussians, 0, 32 + bit);
        CHECK_CUDA(data.debug);
    }

    auto i_ranges_data = torch::zeros({ render_grid_dim.y * render_grid_dim.x, 2 }, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    {
        auto i_ranges = whack::make_tensor_view<glm::uvec2>(i_ranges_data, render_grid_dim.y * render_grid_dim.x);

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
        auto i_ranges = whack::make_tensor_view<glm::uvec2>(i_ranges_data, render_grid_dim.y, render_grid_dim.x);
        whack::start_parallel(
            whack::Location::Device, render_grid_dim, render_block_dim, WHACK_DEVICE_KERNEL(=) {
                WHACK_UNUSED(whack_gridDim);
                // Identify current tile and associated min/max pixel range.
                const glm::uvec2 pix_min = { whack_blockIdx.x * whack_blockDim.x, whack_blockIdx.y * whack_blockDim.y };
                const glm::uvec2 pix_max = min(pix_min + glm::uvec2(whack_blockDim.x, whack_blockDim.y), glm::uvec2(fb_width, fb_height));
                const glm::uvec2 pix = pix_min + glm::uvec2(whack_threadIdx.x, whack_threadIdx.y);
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
                __shared__ glm::vec2 collected_xy[render_block_size];
                __shared__ ConicAndOpacity collected_conic_opacity[render_block_size];

                // Initialize helper variables
                float T = 1.0f;
                glm::vec3 C = glm::vec3(0);

                // Iterate over batches until all done or range is complete
                for (unsigned i = 0; i < n_rounds; i++, n_toDo -= render_block_size) {
                    // End if entire block votes that it is done rasterizing
                    int num_done = __syncthreads_count(done);
                    if (num_done == render_block_size)
                        break;

                    // Collectively fetch per-Gaussian data from global to shared
                    int progress = i * render_block_size + thread_rank;
                    if (render_g_range.x + progress < render_g_range.y) {
                        unsigned coll_id = b_point_list(render_g_range.x + progress);
                        assert(coll_id < n_gaussians);
                        collected_id[thread_rank] = coll_id;
                        collected_xy[thread_rank] = g_points_xy_image(coll_id);
                        collected_conic_opacity[thread_rank] = g_conic_opacity(coll_id);
                    }
                    // block.sync();
                    __syncthreads();

                    if (done)
                        continue;

                    // Iterate over current batch
                    for (unsigned j = 0; j < min(render_block_size, n_toDo); j++) {

                        // Resample using conic matrix (cf. "Surface Splatting" by Zwicker et al., 2001)
                        const auto g_eval = stroke::gaussian::eval_exponential_inv_C(collected_xy[j], collected_conic_opacity[j].conic, glm::vec2(pix));
                        // Eq. (2) from 3D Gaussian splatting paper.
                        // Obtain alpha by multiplying with Gaussian opacity
                        // and its exponential falloff from mean.
                        // Avoid numerical instabilities (see paper appendix).
                        float alpha = min(0.99f, collected_conic_opacity[j].opacity * g_eval);
                        if (alpha < 1.0f / 255.0f)
                            continue;

                        float test_T = T * (1 - alpha);
                        if (test_T < 0.0001f) {
                            done = true;
                            break;
                        }

                        // Eq. (3) from 3D Gaussian splatting paper.
                        C += g_rgb(collected_id[j]) * alpha * T;

                        T = test_T;
                    }
                }

                if (!inside)
                    return;
                // All threads that treat valid pixel write out their final
                // rendering data to the frame and auxiliary buffers.if (inside) {
                //			final_T[pix_id] = T;
                //			n_contrib[pix_id] = last_contributor;
                const auto final_colour = C + T * data.background;
                data.framebuffer(0, pix.y, pix.x) = final_colour.x;
                data.framebuffer(1, pix.y, pix.x) = final_colour.y;
                data.framebuffer(2, pix.y, pix.x) = final_colour.z;
            });
    }

    return { 0 };
}
