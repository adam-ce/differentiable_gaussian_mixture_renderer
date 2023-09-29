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

#include <stroke/cuda_compat.h>
#include <stroke/geometry.h>
#include <stroke/matrix.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

namespace dgmr::utils {

STROKE_DEVICES_INLINE stroke::Cov3<float> compute_cov(const glm::vec3& scale, const glm::quat& rot) {
	const auto RS = glm::toMat3(rot) * glm::mat3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
	return stroke::Cov3<float>(RS * transpose(RS));
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
