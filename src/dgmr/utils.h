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
#include <stroke/gaussian.h>
#include <stroke/geometry.h>
#include <stroke/matrix.h>
#include <stroke/utility.h>
#include <stroke/welford.h>

namespace dgmr::utils {

STROKE_DEVICES_INLINE stroke::Cov3<float> compute_cov(const glm::vec3& scale, const glm::quat& rot)
{
    const auto RS = glm::toMat3(rot) * glm::mat3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
    return stroke::Cov3<float>(RS * transpose(RS));
}

template <typename scalar_t>
STROKE_DEVICES_INLINE glm::mat<3, 3, scalar_t> rotation_matrix_from(const glm::vec<3, scalar_t>& direction)
{
    using Vec = glm::vec<3, scalar_t>;
    using Mat = glm::mat<3, 3, scalar_t>;
    assert(stroke::abs(glm::length(direction) - 1) < 0.0001f);
    const auto dot_z_abs = stroke::abs(dot(direction, Vec(0, 0, 1)));
    const auto dot_x_abs = stroke::abs(dot(direction, Vec(1, 0, 0)));

    const auto other_1 = glm::normalize(glm::cross(direction, dot_z_abs < dot_x_abs ? Vec(0, 0, 1) : Vec(1, 0, 0)));
    const auto other_2 = glm::normalize(glm::cross(other_1, direction));

    return Mat(other_1, other_2, direction);
}

template <typename scalar_t>
struct DirectionAndKernelScales {
    const glm::vec<3, scalar_t>& direction;
    const glm::vec<3, scalar_t>& kernel_scales;
};

template <typename scalar_t>
STROKE_DEVICES_INLINE stroke::Cov3<scalar_t> orient_filter_kernel(const DirectionAndKernelScales<scalar_t>& p)
{
    using Vec = glm::vec<3, scalar_t>;
    using Mat = glm::mat<3, 3, scalar_t>;
    const auto RS = rotation_matrix_from(Vec(p.direction)) * Mat(p.kernel_scales.x, 0, 0, 0, p.kernel_scales.y, 0, 0, 0, p.kernel_scales.z);
    return stroke::Cov3<scalar_t>(RS * transpose(RS));
}

template <glm::length_t n_dims, typename scalar_t>
struct FilteredCovAndWeight {
    stroke::Cov<n_dims, scalar_t> cov;
    float weight_factor;
};

// kernel cov comes from a normalised gaussian, i.e., it integrates to 1 and has no explicit weight
template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE FilteredCovAndWeight<n_dims, scalar_t> convolve_unnormalised_with_normalised(const stroke::Cov<n_dims, scalar_t>& cov, const stroke::Cov<n_dims, scalar_t>& kernel_cov)
{
    const auto new_cov = cov + kernel_cov;
    return { stroke::Cov<n_dims, scalar_t>(new_cov), float(stroke::sqrt(stroke::max(scalar_t(0.000025), scalar_t(det(cov) / det(new_cov))))) };
}

STROKE_DEVICES_INLINE stroke::geometry::Aabb1f gaussian_to_point_distance_bounds(
    const glm::vec3& gauss_centr,
    const glm::vec3& gauss_size,
    const glm::quat& gauss_rotation,
    const float gauss_iso_ellipsoid,
    const glm::vec3& query_point)
{

    const auto transformed_query_point = glm::toMat3(gauss_rotation) * (query_point - gauss_centr);
    const auto s = gauss_size * (0.5f * gauss_iso_ellipsoid);
    const auto transformed_bb = stroke::geometry::Aabb3f { -s, s };

    return { stroke::geometry::distance(transformed_bb, transformed_query_point), stroke::geometry::largest_distance_to(transformed_bb, transformed_query_point) };
}
} // namespace dgmr::utils
