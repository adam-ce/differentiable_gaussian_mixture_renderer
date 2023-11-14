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

#include <stroke/grad/util.h>
#include <stroke/linalg.h>

#include "../utils.h"

namespace dgmr::utils::grad {

template <typename scalar_t = float>
STROKE_DEVICES_INLINE stroke::grad::ThreeGrads<scalar_t, glm::vec<3, scalar_t>, stroke::Cov3<scalar_t>>
splat(scalar_t weight, const glm::vec<3, scalar_t>& centroid, const stroke::Cov3<scalar_t>& cov3D, const Gaussian2d<scalar_t>& incoming_grad, const Camera& camera)
{

    // const auto clamp_to_fov = [&](const glm::vec3& t) {
    //     const auto lim_x = 1.3f * camera.tan_fovx * t.z;
    //     const auto lim_y = 1.3f * camera.tan_fovy * t.z;
    //     return glm::vec3 { stroke::clamp(t.x, -lim_x, lim_x), stroke::clamp(t.y, -lim_y, lim_y), t.z };
    // };

    // const auto t = clamp_to_fov(glm::vec3(camera.view_matrix * glm::vec4(centroid, 1.f))); // clamps the size of the jakobian

    // // following zwicker et al. "EWA Splatting"

    // const auto l_prime = glm::length(t);
    // // clang-format off
    // const auto J = glm::mat3(
    //     glm::mat3::col_type(                              1 / t.z,                                  0.0f,                      t.x / l_prime),
    //     glm::mat3::col_type(                                 0.0f,                               1 / t.z,                      t.y / l_prime),
    //     glm::mat3::col_type(                 -(t.x) / (t.z * t.z),                  -(t.y) / (t.z * t.z),                      t.z / l_prime));
    // const auto S = glm::mat3(
    //     glm::mat3::col_type(                       camera.focal_x,                                  0.0f,                               0.0f),
    //     glm::mat3::col_type(                                 0.0f,                        camera.focal_y,                               0.0f),
    //     glm::mat3::col_type(                                 0.0f,                                  0.0f,                               1.0f));
    // // Avoid matrix multiplication S * J:
    // const auto SJ = glm::mat3(
    //     glm::mat3::col_type(                 camera.focal_x / t.z,                                  0.0f,   (camera.focal_x * t.x) / l_prime),
    //     glm::mat3::col_type(                                 0.0f,                  camera.focal_y / t.z,   (camera.focal_y * t.y) / l_prime),
    //     glm::mat3::col_type(-(camera.focal_x * t.x) / (t.z * t.z), -(camera.focal_y * t.y) / (t.z * t.z),                      t.z / l_prime));
    // // clang-format on

    // const glm::mat3 W = glm::mat3(camera.view_matrix);
    // glm::mat3 T = SJ * W;

    // const auto projected_centroid = project(centroid, camera.view_projection_matrix);
    // dgmr::utils::Gaussian2d screen_space_gaussian;
    // screen_space_gaussian.weight = weight * camera.focal_x * camera.focal_y * det(J); // det(S) == camera.focal_x * camera.focal_y
    // screen_space_gaussian.centroid = ndc2screen(projected_centroid, camera.fb_width, camera.fb_height);
    // screen_space_gaussian.cov = affine_transform_and_cut(cov3D, T);

    // return screen_space_gaussian;
    return {};
}
}
