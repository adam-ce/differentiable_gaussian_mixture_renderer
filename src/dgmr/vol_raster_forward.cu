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

#include "constants.h"
#include "vol_raster_forward.h"

#include <stroke/gaussian.h>
#include <stroke/matrix.h>
#include <whack/Tensor.h>
#include <whack/kernel.h>

namespace {
using namespace dgmr;
namespace gaussian = stroke::gaussian;

// my own:
STROKE_DEVICES glm::vec3 project(const glm::vec3& point, const glm::mat4& projection_matrix) {
	auto pp = projection_matrix * glm::vec4(point, 1.f);
	pp /= pp.w + 0.0000001f;
	return glm::vec3(pp);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE stroke::Cov2<scalar_t> affine_transform_and_cut(const stroke::Cov3<scalar_t>& S, const glm::mat<3, 3, scalar_t>& M) {
	return {
		M[0][0] * (S[0] * M[0][0] + S[1] * M[1][0] + S[2] * M[2][0]) + M[1][0] * (S[1] * M[0][0] + S[3] * M[1][0] + S[4] * M[2][0]) + M[2][0] * (S[2] * M[0][0] + S[4] * M[1][0] + S[5] * M[2][0]),
		M[0][0] * (S[0] * M[0][1] + S[1] * M[1][1] + S[2] * M[2][1]) + M[1][0] * (S[1] * M[0][1] + S[3] * M[1][1] + S[4] * M[2][1]) + M[2][0] * (S[2] * M[0][1] + S[4] * M[1][1] + S[5] * M[2][1]),
		M[0][1] * (S[0] * M[0][1] + S[1] * M[1][1] + S[2] * M[2][1]) + M[1][1] * (S[1] * M[0][1] + S[3] * M[1][1] + S[4] * M[2][1]) + M[2][1] * (S[2] * M[0][1] + S[4] * M[1][1] + S[5] * M[2][1])
	};
}
struct ConicAndOpacity {
	stroke::SymmetricMat<2, float> conic;
	float opacity;
};
static_assert(sizeof(ConicAndOpacity) == 4 * 4);

// from inria:

#define CHECK_CUDA(debug)                                                                                              \
	if (debug) {                                                                                                       \
		auto ret = cudaDeviceSynchronize();                                                                            \
		if (ret != cudaSuccess) {                                                                                      \
			std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
			throw std::runtime_error(cudaGetErrorString(ret));                                                         \
		}                                                                                                              \
	}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
STROKE_DEVICES_INLINE stroke::Cov3<float> computeCov3D(const glm::vec3& scale, float mod, const glm::vec4& rot) {
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot; // / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	return stroke::Cov3<float>(glm::transpose(M) * M);
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
STROKE_DEVICES_INLINE glm::vec3 computeColorFromSH(int deg, const glm::vec3& pos, glm::vec3& campos, const SHs<3>& sh, glm::vec<3, bool>* clamped) {
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

// Forward version of 2D covariance matrix computation
STROKE_DEVICES_INLINE stroke::Cov2<float> computeCov2D(const glm::vec3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const stroke::Cov3<float>& cov3D, const glm::mat4& viewmatrix) {
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002).
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	auto t = glm::vec3(viewmatrix * glm::vec4(mean, 1.f));

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	// adam does not understand, why the last row is 0, and how aspect / scaling considerations work.
	// glm is column major, the first 3 elements are the first column..
	glm::mat3 J = glm::mat3(
		glm::mat3::col_type(focal_x / t.z, 0.0f, 0.f),
		glm::mat3::col_type(0.0f, focal_y / t.z, 0.f),
		glm::mat3::col_type(-(focal_x * t.x) / (t.z * t.z), -(focal_y * t.y) / (t.z * t.z), 0.f));

	glm::mat3 W = glm::mat3(viewmatrix);
	glm::mat3 T = J * W;
	return affine_transform_and_cut(cov3D, T);
}

STROKE_DEVICES_INLINE void getRect(const glm::vec2& p, const glm::ivec2& ext_rect, glm::uvec2* rect_min, glm::uvec2* rect_max, const dim3& render_grid_dim) {
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
uint32_t getHigherMsb(uint32_t n) {
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
}

dgmr::VolRasterStatistics dgmr::vol_raster_forward(VolRasterForwardData& data) {
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

	auto g_cov3d_data = whack::make_tensor<stroke::Cov3<float>>(whack::Location::Device, n_gaussians);
	auto g_cov3d = g_cov3d_data.view();

	auto g_tiles_touched_data = whack::make_tensor<uint32_t>(whack::Location::Device, n_gaussians);
	auto g_tiles_touched = g_tiles_touched_data.view();

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

				const auto cov3d = computeCov3D(data.gm_cov_scales(idx), data.cov_scale_multiplier, data.gm_cov_rotations(idx));
				const auto projected_centroid = project(centroid, data.proj_matrix);
				if (projected_centroid.z < 0.0)
					return;

				auto cov2d = computeCov2D(centroid, focal_x, focal_y, data.tan_fovx, data.tan_fovy, cov3d, data.view_matrix);

				// anti aliasing:
				// Apply low-pass filter: convolve with a gaussian with variance 0.3, i.e. every Gaussian should be at least
				// one pixel wide/high.
				auto opacity = data.gm_weights(idx);
				{
					constexpr float h_var = 0.3f;
					const auto pre_filter_det = det(cov2d);
					cov2d += stroke::Cov2<float>(h_var);
					const auto post_filter_det = det(cov2d);
					// gaussians are not normalised, we need this additional factor. without it, we would produce energy on small gaussians.
					opacity *= sqrt(max(0.000025f, pre_filter_det / post_filter_det));
				}

				// using the more aggressive computation for calculating overlapping tiles:
				{
					const auto ndc2Pix = [](float v, int S) {
						return ((v + 1.0) * S - 1.0) * 0.5;
					};
					// glm::vec2 point_image = ((glm::vec2(projected_centroid) + 1.f) * glm::vec2(fb_width, fb_height) - 1.f) * 0.5f; // same as ndc2Pix
					glm::vec2 point_image = { ndc2Pix(projected_centroid.x, fb_width), ndc2Pix(projected_centroid.y, fb_height) }; // todo: ndc2Pix looks strange to me.

					const glm::uvec2 my_rect = { (int)ceil(3.f * sqrt(cov2d[0])), (int)ceil(3.f * sqrt(cov2d[2])) };
					g_rects(idx) = my_rect;
					glm::uvec2 rect_min, rect_max;
					getRect(point_image, my_rect, &rect_min, &rect_max, render_grid_dim);

					const auto tiles_touched = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
					if (tiles_touched == 0)
						return; // serves as clipping (i think)
					g_tiles_touched(idx) = tiles_touched;
					g_points_xy_image(idx) = point_image;
				}

				g_depths(idx) = glm::length(data.cam_poition - centroid);

				// convert spherical harmonics coefficients to RGB color.
				g_rgb(idx) = computeColorFromSH(data.sh_degree, centroid, data.cam_poition, data.gm_sh_params(idx), &g_rgb_sh_clamped(idx));

				// Inverse 2D covariance and opacity neatly pack into one float4
				const auto conic2d = inverse(cov2d);
				g_conic_opacity(idx) = { conic2d, opacity };
				g_cov3d(idx) = cov3d;
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
				const auto& current_tile_index = whack_blockIdx;
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
				__shared__ float collected_weight[render_block_size];
				__shared__ glm::vec3 collected_centroid[render_block_size];
				__shared__ stroke::Cov3<float> collected_cov3[render_block_size];

				whack::Array<glm::vec4, vol_raster::config::n_rasterisation_steps> rasterised_data = {};
				float max_distance = 0;
				float opacity = 0;

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
						collected_weight[thread_rank] = data.gm_weights(coll_id);

						// todo filtering convolution  -> kernel size based on camera distance (projected pixel size to gaussian distance)
						//                                and scale weight
						const auto centroid = data.gm_centroids(coll_id);
						collected_centroid[thread_rank] = centroid;
						const auto distance = glm::distance(centroid, ray.origin);
						collected_cov3[thread_rank] = g_cov3d(coll_id) + stroke::Cov3<float>(0.00001f + 0.000005f * distance);
					}
					__syncthreads();

					if (done)
						continue;

					// Iterate over current batch
					for (unsigned j = 0; j < min(render_block_size, n_toDo); j++) {
						const auto gaussian1d = gaussian::project_on_ray(collected_centroid[j], collected_cov3[j], ray);
						const auto weight = gaussian1d.weight * collected_weight[j];
						opacity += weight;
						max_distance = gaussian1d.centre + stroke::sqrt(gaussian1d.C) * 3.f;
						if (opacity > 3) {
							done = true;
							break;
						}
					}
				}
				__syncthreads();
				done = !inside;
				n_toDo = render_g_range.y - render_g_range.x;
				opacity = 0;

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
						collected_weight[thread_rank] = data.gm_weights(coll_id);

						// todo filtering convolution  -> kernel size based on camera distance (projected pixel size to gaussian distance)
						//                                and scale weight
						const auto centroid = data.gm_centroids(coll_id);
						collected_centroid[thread_rank] = centroid;
						const auto distance = glm::distance(centroid, ray.origin);
						collected_cov3[thread_rank] = g_cov3d(coll_id) + stroke::Cov3<float>(0.00001f + 0.000005f * distance);
					}
					// block.sync();
					__syncthreads();

					if (done)
						continue;

					// Iterate over current batch
					for (unsigned j = 0; j < min(render_block_size, n_toDo); j++) {

						const auto gaussian1d = gaussian::project_on_ray(collected_centroid[j], collected_cov3[j], ray);
						const auto weight = gaussian1d.weight * collected_weight[j];
						if (gaussian1d.C <= 0)
							continue; // todo: shouldn't happen any more in the future, after implementing AA
						if (!(weight >= 0 && weight < 100'000)) {
							const auto& C3 = collected_cov3[j];
							printf("gaussian1d.C: %f, collected_cov3[j]: %f/%f/%f/%f/%f/%f, det: %f\n", gaussian1d.C, C3[0], C3[1], C3[2], C3[3], C3[4], C3[5], det(C3));
							//							printf("weight: %f, gaussian1d.weight: %f, collected_weight[j]: %f, stroke::gaussian::norm_factor(gaussian1d.C): %f, gaussian1d.C: %f\n", weight, gaussian1d.weight, collected_weight[j], stroke::gaussian::norm_factor(gaussian1d.C), gaussian1d.C);
						}
						if (weight <= 0.001f) // performance critical
							continue;

						const auto inv_variance = 1 / gaussian1d.C;
						const auto delta = max_distance / vol_raster::config::n_rasterisation_steps;
						auto cdf_start = gaussian::cdf_inv_C(gaussian1d.centre, inv_variance, 0.f);
						for (auto k = 0; k < vol_raster::config::n_rasterisation_steps; ++k) {
							//							const auto current_start = k * delta;
							const auto current_end = (k + 1) * delta;
							const auto cdf_end = gaussian::cdf_inv_C(gaussian1d.centre, inv_variance, current_end);
							const auto integrated = (cdf_end - cdf_start) * weight;
							cdf_start = cdf_end;
							rasterised_data[k] += glm::vec4(g_rgb(collected_id[j]) * integrated, integrated);
						}

						opacity += weight;
						if (opacity > 3) {
							done = true;
							break;
						}
					}
				}

				// blend
				float T = 1.0f;
				glm::vec3 C = glm::vec3(0);
				switch (data.debug_render_mode) {
				case VolRasterForwardData::RenderMode::Full: {
					for (auto k = 0; k < vol_raster::config::n_rasterisation_steps; ++k) {
						auto current_bin = rasterised_data[k];
						for (auto i = 0; i < 3; ++i) {
							current_bin[i] /= current_bin[3] + 0.01f; // make an weighted average out of a weighted sum
						}
						// Avoid numerical instabilities (see paper appendix).
						float alpha = min(0.99f, current_bin[3]);
						if (alpha < 1.0f / 255.0f)
							continue;

						float test_T = T * (1 - alpha);
						if (test_T < 0.0001f) {
							done = true;
							break;
						}

						// Eq. (3) from 3D Gaussian splatting paper.
						C += glm::vec3(current_bin) * alpha * T;

						T = test_T;
					}
				} break;
				case VolRasterForwardData::RenderMode::Bins: {
					if (unsigned(data.debug_render_bin) >= vol_raster::config::n_rasterisation_steps)
						break;
					auto current_bin = rasterised_data[data.debug_render_bin];
					for (auto i = 0; i < 3; ++i) {
						current_bin[i] /= current_bin[3] + 0.01f; // make an weighted average out of a weighted sum
					}
					float alpha = min(0.99f, current_bin[3]);
					float test_T = T * (1 - alpha);
					C += glm::vec3(current_bin) * alpha * T;

					T = test_T;
				} break;
				case VolRasterForwardData::RenderMode::Depth: {
					C = glm::vec3(max_distance / data.max_depth);
					T = 0;
				} break;
				}

				if (!inside)
					return;
				// All threads that treat valid pixel write out their final
				const auto final_colour = C + T * glm::vec3(1, 0, 0); // data.background;
				data.framebuffer(0, pix.y, pix.x) = final_colour.x;
				data.framebuffer(1, pix.y, pix.x) = final_colour.y;
				data.framebuffer(2, pix.y, pix.x) = final_colour.z;
			});
	}

	return { 0 };
}
