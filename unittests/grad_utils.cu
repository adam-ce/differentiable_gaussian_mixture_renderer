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

#include <stroke/pretty_printers.h>
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
    return stroke::Cov<n_dims, scalar_t>(mat * transpose(mat)) + stroke::Cov<n_dims, scalar_t>(scalar_t(0.05));
}

template <typename scalar_t, typename Generator>
Camera<scalar_t> random_camera(Generator* rnd)
{
    const scalar_t fovy = scalar_t(3.14 / 4);

    Camera<scalar_t> c;
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

void check_splat()
{
    using scalar_t = double;
    using vec3_t = glm::vec<3, scalar_t>;
    using cov3_t = stroke::Cov3<scalar_t>;

    whack::random::HostGenerator<scalar_t> rnd;

    for (int i = 0; i < 10; ++i) {
        const auto cam = random_camera<scalar_t>(&rnd);
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
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
}

void check_project()
{
    using scalar_t = double;
    using vec3_t = glm::vec<3, scalar_t>;
    using cov3_t = stroke::Cov3<scalar_t>;

    whack::random::HostGenerator<scalar_t> rnd;

    for (int i = 0; i < 10; ++i) {
        const auto cam = random_camera<scalar_t>(&rnd);
        const auto fun = [cam](const whack::Tensor<scalar_t, 1>& input) {
            const auto pos = stroke::extract<vec3_t>(input);
            const auto p = dgmr::utils::project<scalar_t>(pos, cam.view_projection_matrix);
            return stroke::pack_tensor<scalar_t>(p);
        };

        const auto fun_grad = [cam](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
            const auto pos = stroke::extract<vec3_t>(input);
            const auto grad_incoming = stroke::extract<glm::vec<3, scalar_t>>(grad_output);
            const auto grad_outgoing = grad::project<scalar_t>(pos, cam.view_projection_matrix, grad_incoming);
            return stroke::pack_tensor<scalar_t>(grad_outgoing);
        };

        const auto test_data = stroke::pack_tensor<scalar_t>(rnd.normal3());
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
}

void check_affine_transform_and_cut()
{
    using scalar_t = double;
    using vec3_t = glm::vec<3, scalar_t>;
    using cov2_t = stroke::Cov2<scalar_t>;
    using cov3_t = stroke::Cov3<scalar_t>;
    using mat3_t = glm::mat<3, 3, scalar_t>;

    whack::random::HostGenerator<scalar_t> rnd;

    for (int i = 0; i < 10; ++i) {
        const auto fun = [](const whack::Tensor<scalar_t, 1>& input) {
            const auto [cov, mat] = stroke::extract<cov3_t, mat3_t>(input);
            const auto p = dgmr::utils::affine_transform_and_cut<scalar_t>(cov, mat);
            return stroke::pack_tensor<scalar_t>(p);
        };

        const auto fun_grad = [](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
            const auto [cov, mat] = stroke::extract<cov3_t, mat3_t>(input);
            const auto grad_incoming = stroke::extract<cov2_t>(grad_output);
            const auto [grad_cov, grad_mat] = grad::affine_transform_and_cut<scalar_t>(cov, mat, grad_incoming);
            return stroke::pack_tensor<scalar_t>(grad_cov, grad_mat);
        };

        const auto test_data = stroke::pack_tensor<scalar_t>(host_random_cov<3, scalar_t>(&rnd), host_random_matrix<3, scalar_t>(&rnd));
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
}

void check_ndc2screent()
{
    using scalar_t = double;
    using vec2_t = glm::vec<2, scalar_t>;
    using vec3_t = glm::vec<3, scalar_t>;

    whack::random::HostGenerator<scalar_t> rnd;

    for (int i = 0; i < 10; ++i) {
        unsigned width = unsigned(rnd.uniform() * 500) + 200;
        unsigned height = unsigned(rnd.uniform() * 500) + 200;
        const auto fun = [=](const whack::Tensor<scalar_t, 1>& input) {
            const auto pos = stroke::extract<vec3_t>(input);
            const auto p = dgmr::utils::ndc2screen<scalar_t>(pos, width, height);
            return stroke::pack_tensor<scalar_t>(p);
        };

        const auto fun_grad = [=](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
            const auto pos = stroke::extract<vec3_t>(input);
            const auto grad_incoming = stroke::extract<vec2_t>(grad_output);
            const auto grad_pos = grad::ndc2screen<scalar_t>(pos, width, height, grad_incoming);
            return stroke::pack_tensor<scalar_t>(grad_pos);
        };

        const auto test_data = stroke::pack_tensor<scalar_t>(rnd.normal3());
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
}

void check_make_jakobian()
{
    using scalar_t = double;
    using vec3_t = glm::vec<3, scalar_t>;
    using cov3_t = stroke::Cov3<scalar_t>;

    whack::random::HostGenerator<scalar_t> rnd;

    for (int i = 0; i < 10; ++i) {
        const auto cam = random_camera<scalar_t>(&rnd);
        const auto fun = [cam](const whack::Tensor<scalar_t, 1>& input) {
            const auto [pos, l_prime] = stroke::extract<vec3_t, scalar_t>(input);
            const auto p1 = dgmr::utils::make_jakobian<scalar_t>(pos, l_prime, 1, 1);
            //            const auto p2 = dgmr::utils::make_jakobian<scalar_t>(pos, l_prime, cam.focal_x, cam.focal_y);
            //            return stroke::pack_tensor<scalar_t>(p1, p2);
            return stroke::pack_tensor<scalar_t>(p1);
        };

        const auto fun_grad = [cam](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
            const auto [pos, l_prime] = stroke::extract<vec3_t, scalar_t>(input);
            //            const auto [grad_incoming1, grad_incoming2] = stroke::extract<glm::mat<3, 3, scalar_t>, glm::mat<3, 3, scalar_t>>(grad_output);
            const auto grad_incoming1 = stroke::extract<glm::mat<3, 3, scalar_t>>(grad_output);
            const auto [grad_pos1, grad_l_prime1] = grad::make_jakobian<scalar_t>(pos, l_prime, grad_incoming1, 1, 1);
            //            const auto [grad_pos2, grad_l_prime2] = grad::make_jakobian<scalar_t>(pos, l_prime, grad_incoming2, cam.focal_x, cam.focal_y);
            //            return stroke::pack_tensor<scalar_t>(grad_pos1 + grad_pos2, grad_l_prime1 + grad_l_prime2);
            return stroke::pack_tensor<scalar_t>(grad_pos1, grad_l_prime1);
        };

        const auto pos = rnd.normal3() + vec3_t(0, 0, 1.5);
        const auto test_data = stroke::pack_tensor<scalar_t>(pos, length(pos));
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
}

} // namespace

TEST_CASE("dgmr splat gradient")
{
    check_splat();
}

TEST_CASE("dgmr project gradient")
{
    check_project();
}

TEST_CASE("dgmr affine_transform_and_cut gradient")
{
    check_affine_transform_and_cut();
}

TEST_CASE("dgmr ndc2screen gradient")
{
    check_ndc2screent();
}

TEST_CASE("dgmr make_jakobian gradient")
{
    check_make_jakobian();
}
