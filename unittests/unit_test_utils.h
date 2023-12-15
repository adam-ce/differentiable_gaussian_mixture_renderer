/*****************************************************************************
 * DGMR
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

#include <catch2/catch_approx.hpp>
#include <dgmr/math.h>
#include <glm/glm.hpp>
#include <stroke/linalg.h>
#include <whack/random/generators.h>

namespace dgmr::unittest {
template <glm::length_t n_dims, typename scalar_t, typename Generator>
glm::mat<n_dims, n_dims, scalar_t> host_random_matrix(Generator* rnd)
{
    glm::mat<n_dims, n_dims, scalar_t> mat;
    for (auto c = 0; c < n_dims; ++c) {
        for (auto r = 0; r < n_dims; ++r) {
            mat[c][r] = rnd->normal();
        }
    }
    return mat;
}

template <glm::length_t n_dims, typename scalar_t, typename Generator>
stroke::Cov<n_dims, scalar_t> host_random_cov(Generator* rnd)
{
    const auto mat = host_random_matrix<n_dims, scalar_t>(rnd);
    return stroke::Cov<n_dims, scalar_t>(mat * transpose(mat)) + stroke::Cov<n_dims, scalar_t>(scalar_t(0.05));
}

template <typename scalar_t, typename Generator>
dgmr::math::Camera<scalar_t> random_camera(Generator* rnd)
{
    const scalar_t fovy = scalar_t(3.14 / 4);

    dgmr::math::Camera<scalar_t> c;
    c.fb_height = 600;
    c.fb_width = 800;
    const auto aspect = scalar_t(c.fb_width) / scalar_t(c.fb_height);
    c.tan_fovy = std::tan(fovy);
    c.tan_fovx = c.tan_fovy * aspect; // via https://stackoverflow.com/questions/5504635/computing-fovx-opengl
    c.focal_x = scalar_t(c.fb_width) / (scalar_t(2.0) * c.tan_fovx);
    c.focal_y = scalar_t(c.fb_height) / (scalar_t(2.0) * c.tan_fovy);

    const auto pos = glm::normalize(rnd->normal3()) * scalar_t(15);
    const auto target = rnd->normal3() * scalar_t(1.5);
    const auto up_vector = glm::normalize(rnd->normal3());
    c.view_matrix = glm::lookAt(pos, target, up_vector);
    c.view_projection_matrix = glm::perspective(scalar_t(fovy), aspect, scalar_t(0.1), scalar_t(100.)) * c.view_matrix;
    return c;
}

// template <int n_dims>
// bool equals(const glm::vec<n_dims, double>& a, const glm::vec<n_dims, double>& b, double scale = 1)
// {
//     const auto delta = glm::length(a - b);
//     return delta == Catch::Approx(0).scale(scale);
// }

} // namespace dgmr::unittest
