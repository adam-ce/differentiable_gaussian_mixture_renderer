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

struct Gaussian2d {
    float weight;
    glm::vec2 centroid;
    stroke::Cov2<float> cov;
};

struct Camera {
    glm::mat4 view_matrix = {};
    glm::mat4 view_projection_matrix = {};
    float focal_x = 0;
    float focal_y = 0;
    float tan_fovx = 0;
    float tan_fovy = 0;
    unsigned fb_width = 0;
    unsigned fb_height = 0;
};

template <typename scalar_t>
STROKE_DEVICES_INLINE stroke::Cov2<scalar_t> affine_transform_and_cut(const stroke::Cov3<scalar_t>& S, const glm::mat<3, 3, scalar_t>& M)
{
    return {
        M[0][0] * (S[0] * M[0][0] + S[1] * M[1][0] + S[2] * M[2][0]) + M[1][0] * (S[1] * M[0][0] + S[3] * M[1][0] + S[4] * M[2][0]) + M[2][0] * (S[2] * M[0][0] + S[4] * M[1][0] + S[5] * M[2][0]),
        M[0][0] * (S[0] * M[0][1] + S[1] * M[1][1] + S[2] * M[2][1]) + M[1][0] * (S[1] * M[0][1] + S[3] * M[1][1] + S[4] * M[2][1]) + M[2][0] * (S[2] * M[0][1] + S[4] * M[1][1] + S[5] * M[2][1]),
        M[0][1] * (S[0] * M[0][1] + S[1] * M[1][1] + S[2] * M[2][1]) + M[1][1] * (S[1] * M[0][1] + S[3] * M[1][1] + S[4] * M[2][1]) + M[2][1] * (S[2] * M[0][1] + S[4] * M[1][1] + S[5] * M[2][1])
    };
}

STROKE_DEVICES_INLINE glm::vec3 project(const glm::vec3& point, const glm::mat4& projection_matrix)
{
    auto pp = projection_matrix * glm::vec4(point, 1.f);
    pp /= pp.w + 0.0000001f;
    return glm::vec3(pp);
}

STROKE_DEVICES_INLINE glm::vec2 ndc2screen(const glm::vec3& point, unsigned width, unsigned height)
{
    const auto ndc2Pix = [](float v, int S) {
        return ((v + 1.0) * S - 1.0) * 0.5;
    };
    return { ndc2Pix(point.x, width), ndc2Pix(point.y, height) };
}

STROKE_DEVICES_INLINE Gaussian2d splat(float weight, const glm::vec3& centroid, const stroke::Cov3<float>& cov3D, const Camera& camera)
{

    const auto clamp_to_fov = [&](const glm::vec3& t) {
        const auto lim_x = 1.3f * camera.tan_fovx * t.z;
        const auto lim_y = 1.3f * camera.tan_fovy * t.z;
        return glm::vec3 { stroke::clamp(t.x, -lim_x, lim_x), stroke::clamp(t.y, -lim_y, lim_y), t.z };
    };

    const auto t = clamp_to_fov(glm::vec3(camera.view_matrix * glm::vec4(centroid, 1.f))); // clamps the size of the jakobian

    // following zwicker et al. "EWA Splatting"

    const auto l_prime = glm::length(t);
    // clang-format off
    const auto J = glm::mat3(
        glm::mat3::col_type(                              1 / t.z,                                  0.0f,                      t.x / l_prime),
        glm::mat3::col_type(                                 0.0f,                               1 / t.z,                      t.y / l_prime),
        glm::mat3::col_type(                 -(t.x) / (t.z * t.z),                  -(t.y) / (t.z * t.z),                      t.z / l_prime));
    const auto S = glm::mat3(
        glm::mat3::col_type(                       camera.focal_x,                                  0.0f,                               0.0f),
        glm::mat3::col_type(                                 0.0f,                        camera.focal_y,                               0.0f),
        glm::mat3::col_type(                                 0.0f,                                  0.0f,                               1.0f));
    // Avoid matrix multiplication S * J:
    const auto SJ = glm::mat3(
        glm::mat3::col_type(                 camera.focal_x / t.z,                                  0.0f,   (camera.focal_x * t.x) / l_prime),
        glm::mat3::col_type(                                 0.0f,                  camera.focal_y / t.z,   (camera.focal_y * t.y) / l_prime),
        glm::mat3::col_type(-(camera.focal_x * t.x) / (t.z * t.z), -(camera.focal_y * t.y) / (t.z * t.z),                      t.z / l_prime));
    // clang-format on

    const glm::mat3 W = glm::mat3(camera.view_matrix);
    glm::mat3 T = SJ * W;

    const auto projected_centroid = project(centroid, camera.view_projection_matrix);
    dgmr::utils::Gaussian2d screen_space_gaussian;
    screen_space_gaussian.weight = weight * det(S) * det(J);
    screen_space_gaussian.centroid = ndc2screen(projected_centroid, camera.fb_width, camera.fb_height);
    screen_space_gaussian.cov = affine_transform_and_cut(cov3D, T);

    return screen_space_gaussian;
}

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
