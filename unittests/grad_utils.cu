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

#include <stroke/unittest/gradcheck.h>

#include <catch2/catch_test_macros.hpp>
#include <dgmr/grad/utils.h>
#include <dgmr/utils.h>
#include <glm/glm.hpp>
#include <stroke/linalg.h>
#include <whack/random/generators.h>

using namespace dgmr::utils;

namespace {
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
    return stroke::Cov<n_dims, scalar_t>(mat * transpose(mat)) + stroke::Cov<n_dims, scalar_t>(0.05);
}

template <typename Generator>
Camera random_camera(Generator* rnd)
{
    const auto fovy = rnd->uniform() * 3.14 / 2;
    using scalar_t = decltype(rnd->normal());

    Camera c;
    c.fb_height = rnd->uniform() * 1000;
    c.fb_width = rnd->uniform() * 1000;
    c.tan_fovx = std::tan(fovy);
    c.tan_fovy = std::tan(rnd->uniform() * 3.14 / 2);
    c.focal_x = c.fb_width / (2.0f * c.tan_fovx);
    c.focal_y = c.fb_height / (2.0f * c.tan_fovy);
    c.view_matrix = glm::lookAt(rnd->normal3() * scalar_t(10.), rnd->normal3() * scalar_t(0.5), glm::normalize(rnd->normal3()));
    c.view_projection_matrix = glm::perspective(float(fovy), float(c.fb_width) / c.fb_height, 0.1f, 100.f) * c.view_matrix;
    return c;
}

void check_splat()
{
    using scalar_t = float;
    using vec3_t = glm::vec<3, scalar_t>;
    using cov3_t = stroke::Cov3<scalar_t>;

    whack::random::HostGenerator<scalar_t> rnd;

    for (int i = 0; i < 1; ++i) {
        const auto cam = random_camera(&rnd);
        const auto fun = [cam](const whack::Tensor<scalar_t, 1>& input) {
            const auto [weight, pos, cov] = stroke::extract<scalar_t, vec3_t, cov3_t>(input);
            Gaussian2d<scalar_t> g = splat<scalar_t>(weight, pos, cov, cam);
            return stroke::pack_tensor<scalar_t>(g);
        };

        const auto fun_grad = [cam](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
            const auto [weight, pos, cov] = stroke::extract<scalar_t, vec3_t, cov3_t>(input);
            const Gaussian2d<scalar_t> grad_incoming = stroke::extract<Gaussian2d<scalar_t>>(grad_output);
            const auto grad_outgoing = grad::splat<scalar_t>(weight, pos, cov, grad_incoming, cam);
            return stroke::pack_tensor<scalar_t>(grad_outgoing);
        };

        const auto test_data = stroke::pack_tensor<scalar_t>(
            rnd.normal(),
            rnd.normal3(),
            host_random_cov<3, scalar_t>(&rnd));
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.0000001));
    }
}
} // namespace

TEST_CASE("dgmr gaussian gradients")
{
    check_splat();
}
