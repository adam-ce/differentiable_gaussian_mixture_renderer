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

#pragma once

#include <cuda/std/tuple>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <stroke/cuda_compat.h>
#include <stroke/geometry.h>
#include <stroke/matrix.h>

namespace dgmr::utils {

template <typename config>
struct RasterBinSizer {
	static constexpr auto n_bins = config::n_rasterisation_steps;
	static constexpr auto transmission_threshold = config::transmission_threshold;
	whack::Array<float, n_bins> bin_borders = {};
	float transmission = 1.f;
	STROKE_DEVICES_INLINE float begin_of(unsigned i) const {
		if (i == 0)
			return 0;
		return bin_borders[i - 1];
	}
	STROKE_DEVICES_INLINE float end_of(unsigned i) const {
		return bin_borders[i];
	}
	STROKE_DEVICES_INLINE float max_distance() const {
		return bin_borders[n_bins - 1];
	}
	STROKE_DEVICES_INLINE bool is_full() const {
		return transmission < transmission_threshold;
	}
	STROKE_DEVICES_INLINE void add_opacity(float pos, float alpha) {
		const auto bin_for = [](float transmission) {
			const auto t_flipped_and_scaled = 1 - (transmission - transmission_threshold) / (1 - transmission_threshold);
			const auto t_bin = unsigned(stroke::min(t_flipped_and_scaled, 1.0001f) * n_bins);
			return t_bin - 1;
		};
		const auto last_bin = bin_for(transmission);
		transmission *= 1 - alpha;
		const auto current_bin = bin_for(transmission);
		if (last_bin != current_bin && current_bin < n_bins - 1)
			bin_borders[current_bin] = pos;
		bin_borders[n_bins - 1] = stroke::max(pos, bin_borders[n_bins - 1]);
	}
	STROKE_DEVICES_INLINE void finalise() {
		const auto find_empty = [this](unsigned start_pos) {
			for (unsigned i = start_pos; i < n_bins; ++i) {
				if (bin_borders[i] == 0)
					return i;
			}
			return unsigned(-1);
		};
		const auto find_filled = [this](unsigned start_pos) {
			for (unsigned i = start_pos; i < n_bins; ++i) {
				if (bin_borders[i] != 0)
					return i;
			}
			return unsigned(-1);
		};

		unsigned fill_start = find_empty(0);
		while (fill_start < n_bins) {
			unsigned fill_end = find_filled(fill_start);
			assert(fill_end > fill_start);
			assert(fill_end < n_bins); // last bin should be always filled

			const auto delta = (end_of(fill_end) - begin_of(fill_start)) / (fill_end - fill_start + 1);
			const auto offset = begin_of(fill_start);
			for (unsigned i = 0; i < fill_end - fill_start; ++i) {
				bin_borders[fill_start + i] = offset + delta * (i + 1);
			}

			fill_start = find_empty(fill_end);
		}
	}
};

STROKE_DEVICES_INLINE stroke::Cov3<float> compute_cov(const glm::vec3& scale, const glm::quat& rot) {
	const auto RS = glm::toMat3(rot) * glm::mat3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
	return stroke::Cov3<float>(RS * transpose(RS));
}

// todo: this doesn't use resolution or fov. need to compute convolution size based on that. + also need to compute weight adjustment.
struct FilteredCov3AndWeight {
	stroke::Cov3<float> cov;
	float weight_factor;
};

STROKE_DEVICES_INLINE FilteredCov3AndWeight filter_for_aa(const glm::vec3& centroid, const stroke::Cov3<float>& cov, const glm::vec3& camera_position) {
	const auto distance = glm::distance(centroid, camera_position);
	const auto new_cov = cov + stroke::Cov3(0.00001f + 0.000005f * distance);
	return { new_cov, 1.f };
}

STROKE_DEVICES_INLINE stroke::geometry::Aabb1f gaussian_to_point_distance_bounds(
	const glm::vec3& gauss_centr,
	const glm::vec3& gauss_size,
	const glm::quat& gauss_rotation,
	const float gauss_iso_ellipsoid,
	const glm::vec3& query_point) {

	const auto transformed_query_point = glm::toMat3(gauss_rotation) * (query_point - gauss_centr);
	const auto s = gauss_size * (0.5f * gauss_iso_ellipsoid);
	const auto transformed_bb = stroke::geometry::Aabb3f { -s, s };

	return { stroke::geometry::distance(transformed_bb, transformed_query_point), stroke::geometry::largest_distance_to(transformed_bb, transformed_query_point) };
}
}
