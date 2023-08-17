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
#include "forward.h"

#include <stroke/matrix.h>
#include <whack/Tensor.h>
#include <whack/kernel.h>

namespace {
using namespace dgmr;

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

// from inria:

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

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(viewmatrix);

	glm::mat3 T = W * J;
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
}

dgmr::Statistics dgmr::forward(ForwardData& data) {
	const auto fb_width = data.framebuffer.size<2>();
	const auto fb_height = data.framebuffer.size<1>();
	const auto n_gaussians = data.gm_weights.size<0>();
	const float focal_y = fb_height / (2.0f * data.tan_fovy);
	const float focal_x = fb_width / (2.0f * data.tan_fovx);

	const dim3 render_block_dim = { render_block_width, render_block_height };
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
				const auto cov3d = computeCov3D(data.gm_cov_scales(idx), data.cov_scale_multiplier, data.gm_cov_rotations(idx));
				const auto projected_centroid = project(centroid, data.proj_matrix);
				if (projected_centroid.z < 0)
					return;

				const auto cov2d = computeCov2D(centroid, focal_x, focal_y, data.tan_fovx, data.tan_fovy, cov3d, data.view_matrix);

				// todo: anti aliasing
				const auto opacity = data.gm_weights(idx);

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
			});
	}
	// next steps:
	// 1. port sorting / rect / tiling calc
	// 2. draw something out of them for debug..
	// 3. properly port render pass

	// render
	{
		whack::start_parallel(
			whack::Location::Device, render_grid_dim, render_block_dim, WHACK_KERNEL(data) {
				WHACK_UNUSED(whack_gridDim);
				const unsigned p_x = whack_blockIdx.x * whack_blockDim.x + whack_threadIdx.x;
				const unsigned p_y = whack_blockIdx.y * whack_blockDim.y + whack_threadIdx.y;
				const auto fb_width = data.framebuffer.size<2>();
				const auto fb_height = data.framebuffer.size<1>();
				if (p_x >= fb_width || p_y >= fb_height)
					return;
				data.framebuffer(0, p_y, p_x) = float(p_x) / fb_width;
				data.framebuffer(1, p_y, p_x) = float(p_y) / fb_height;
				data.framebuffer(2, p_y, p_x) = 1.0;
			});
	}

	return { 0 };
}
