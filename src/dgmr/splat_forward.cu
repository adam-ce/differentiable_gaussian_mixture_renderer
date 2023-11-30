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
#include "splat_forward.h"
#include "utils.h"

#include <stroke/gaussian.h>
#include <whack/Tensor.h>
#include <whack/kernel.h>

namespace {
using namespace dgmr;

// from inria:

#define CHECK_CUDA(debug)                                                                                              \
    if (debug) {                                                                                                       \
        auto ret = cudaDeviceSynchronize();                                                                            \
        if (ret != cudaSuccess) {                                                                                      \
            std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
            throw std::runtime_error(cudaGetErrorString(ret));                                                         \
        }                                                                                                              \
    }

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
STROKE_DEVICES_INLINE glm::vec3 computeColorFromSH(int deg, const glm::vec3& pos, glm::vec3& campos, const SHs<3>& sh, glm::vec<3, bool>* clamped)
{
    // The implementation is loosely based on code for
    // "Differentiable Point-Based Radiance Fields for
    // Efficient View Synthesis" by Zhang et al. (2022)
    glm::vec3 dir = pos - campos;
    dir = dir / glm::length(dir);

    glm::vec3 result = SH_C0 * sh[0];

    if (deg > 0) {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

        if (deg > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result + SH_C2[0] * xy * sh[4] + SH_C2[1] * yz * sh[5] + SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] + SH_C2[3] * xz * sh[7] + SH_C2[4] * (xx - yy) * sh[8];

            if (deg > 2) {
                result = result + SH_C3[0] * y * (3.0f * xx - yy) * sh[9] + SH_C3[1] * xy * z * sh[10] + SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] + SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] + SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] + SH_C3[5] * z * (xx - yy) * sh[14] + SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }
    result += 0.5f;

    // RGB colors are clamped to positive values. If values are
    // clamped, we need to keep track of this for the backward pass.
    clamped->x = (result.x < 0);
    clamped->y = (result.y < 0);
    clamped->z = (result.z < 0);
    return glm::max(result, 0.0f);
}

STROKE_DEVICES_INLINE void getRect(const glm::vec2& p, const glm::ivec2& ext_rect, glm::uvec2* rect_min, glm::uvec2* rect_max, const dim3& render_grid_dim)
{
    *rect_min = {
        min(render_grid_dim.x, max((int)0, (int)((p.x - ext_rect.x) / render_block_width))),
        min(render_grid_dim.y, max((int)0, (int)((p.y - ext_rect.y) / render_block_height)))
    };
    *rect_max = {
        min(render_grid_dim.x, max((int)0, (int)((p.x + ext_rect.x + render_block_width - 1) / render_block_width))),
        min(render_grid_dim.y, max((int)0, (int)((p.y + ext_rect.y + render_block_height - 1) / render_block_height)))
    };
}

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
    uint32_t msb = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1) {
        step /= 2;
        if (n >> msb)
            msb += step;
        else
            msb -= step;
    }
    if (n >> msb)
        msb++;
    return msb;
}
} // namespace

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

    auto g_conic_opacity_data = whack::make_tensor<ConicAndOpacity>(whack::Location::Device, n_gaussians);
    auto g_conic_opacity = g_conic_opacity_data.view();

    auto g_tiles_touched_data = whack::make_tensor<uint32_t>(whack::Location::Device, n_gaussians);
    auto g_tiles_touched = g_tiles_touched_data.view();

    utils::Camera<float> camera {
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

                const auto cov3d = utils::compute_cov(data.gm_cov_scales(idx) * data.cov_scale_multiplier, data.gm_cov_rotations(idx));

                const auto screen_space_gaussian = utils::splat<splat::config::use_orientation_dependent_gaussian_density>(data.gm_weights(idx), centroid, cov3d, camera, 0.3f);
                const auto opacity = screen_space_gaussian.weight;

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

                // convert spherical harmonics coefficients to RGB color.
                g_rgb(idx) = computeColorFromSH(data.sh_degree, centroid, data.cam_poition, data.gm_sh_params(idx), &g_rgb_sh_clamped(idx));

                // Inverse 2D covariance and opacity neatly pack into one float4
                const auto conic2d = inverse(screen_space_gaussian.cov);
                g_conic_opacity(idx) = { conic2d, opacity };
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
        auto i_ranges = whack::make_tensor_view(i_ranges_data.device_vector(), render_grid_dim.y, render_grid_dim.x);
        whack::start_parallel(
            whack::Location::Device, render_grid_dim, render_block_dim, WHACK_DEVICE_KERNEL(=) {
                WHACK_UNUSED(whack_gridDim);
                // Identify current tile and associated min/max pixel range.
                const auto& current_tile_index = whack_blockIdx;
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
                        const auto g_eval = splat::config::use_orientation_dependent_gaussian_density
                            ? stroke::gaussian::eval_normalised_inv_C(collected_xy[j], collected_conic_opacity[j].conic, glm::vec2(pix))
                            : stroke::gaussian::eval_exponential_inv_C(collected_xy[j], collected_conic_opacity[j].conic, glm::vec2(pix));
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
