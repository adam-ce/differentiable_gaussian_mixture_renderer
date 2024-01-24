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
#include <dgmr/grad/math.h>

#include "unit_test_utils.h"

using namespace dgmr::math;
using namespace dgmr::unittest;

namespace {

template <dgmr::Formulation formulation>
void check_splat(double filter_kernel_size)
{
    using scalar_t = double;
    using vec3_t = glm::vec<3, scalar_t>;
    using cov3_t = stroke::Cov3<scalar_t>;
    using quat_t = glm::qua<scalar_t>;

    whack::random::HostGenerator<scalar_t> rnd;

    for (int i = 0; i < 10; ++i) {
        const auto cam = random_camera<scalar_t>(&rnd);
        const auto fun = [cam, filter_kernel_size](const whack::Tensor<scalar_t, 1>& input) {
            const auto [weight, pos, scales, rot] = stroke::extract<scalar_t, vec3_t, vec3_t, quat_t>(input);
            Gaussian2d<scalar_t> g = splat<formulation, scalar_t>(weight, pos, scales, rot, cam, scalar_t(filter_kernel_size));
            return stroke::pack_tensor<scalar_t>(g);
        };

        const auto fun_grad = [cam, filter_kernel_size](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
            const auto [weight, pos, scales, rot] = stroke::extract<scalar_t, vec3_t, vec3_t, quat_t>(input);
            const Gaussian2d<scalar_t> grad_incoming = stroke::extract<Gaussian2d<scalar_t>>(grad_output);
            const auto grad_outgoing = grad::splat<formulation, scalar_t>(weight, pos, scales, rot, grad_incoming, cam, scalar_t(filter_kernel_size));
            return stroke::pack_tensor<scalar_t>(grad_outgoing);
        };

        const auto test_data = stroke::pack_tensor<scalar_t>(
            rnd.normal(),
            rnd.normal3(),
            rnd.uniform3(),
            stroke::host_random_quaternion<scalar_t>(&rnd));
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000002), scalar_t(100)); // i think the gradient is correct, but a bit unstable.
    }
}

// template <bool orientation_dependent_gaussian_density>
// void check_splat_cached(double filter_kernel_size)
// {
//     using scalar_t = double;
//     using vec3_t = glm::vec<3, scalar_t>;
//     using cov3_t = stroke::Cov3<scalar_t>;

//     whack::random::HostGenerator<scalar_t> rnd;

//     for (int i = 0; i < 10; ++i) {
//         const auto cam = random_camera<scalar_t>(&rnd);
//         const auto fun = [cam, filter_kernel_size](const whack::Tensor<scalar_t, 1>& input) {
//             const auto [weight, pos, cov] = stroke::extract<scalar_t, vec3_t, cov3_t>(input);
//             const auto gc = splat_with_cache<orientation_dependent_gaussian_density, scalar_t>(weight, pos, cov, cam, scalar_t(filter_kernel_size));
//             return stroke::pack_tensor<scalar_t>(gc.gaussian);
//         };

//         const auto fun_grad = [cam, filter_kernel_size](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
//             const auto [weight, pos, cov] = stroke::extract<scalar_t, vec3_t, cov3_t>(input);
//             const Gaussian2d<scalar_t> grad_incoming = stroke::extract<Gaussian2d<scalar_t>>(grad_output);
//             const auto gc = splat_with_cache<orientation_dependent_gaussian_density, scalar_t>(weight, pos, cov, cam, scalar_t(filter_kernel_size));
//             const auto grad_outgoing = grad::splat_with_cache<orientation_dependent_gaussian_density, scalar_t>(weight, pos, cov, grad_incoming, gc, cam, scalar_t(filter_kernel_size));
//             return stroke::pack_tensor<scalar_t>(grad_outgoing);
//         };

//         const auto test_data = stroke::pack_tensor<scalar_t>(
//             rnd.normal(),
//             rnd.normal3(),
//             host_random_cov<3, scalar_t>(&rnd));
//         stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000002), scalar_t(100)); // i think the gradient is correct, but a bit unstable.
//     }
// }

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
            const auto p = dgmr::math::project<scalar_t>(pos, cam.view_projection_matrix);
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
            const auto p = dgmr::math::affine_transform_and_cut<scalar_t>(cov, mat);
            return stroke::pack_tensor<scalar_t>(p);
        };

        const auto fun_grad = [](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
            const auto [cov, mat] = stroke::extract<cov3_t, mat3_t>(input);
            const auto grad_incoming = stroke::extract<cov2_t>(grad_output);
            const auto [grad_cov, grad_mat] = grad::affine_transform_and_cut<scalar_t>(cov, mat, grad_incoming);
            return stroke::pack_tensor<scalar_t>(grad_cov, grad_mat);
        };

        const auto test_data = stroke::pack_tensor<scalar_t>(stroke::host_random_cov<3, scalar_t>(&rnd), stroke::host_random_matrix<3, scalar_t>(&rnd));
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
}

void check_convolve_unnormalised_with_normalised(double filter_kernel_size)
{
    using scalar_t = double;
    using vec3_t = glm::vec<3, scalar_t>;
    using cov2_t = stroke::Cov2<scalar_t>;
    using cov3_t = stroke::Cov3<scalar_t>;
    using mat3_t = glm::mat<3, 3, scalar_t>;

    whack::random::HostGenerator<scalar_t> rnd;
    const auto filter_kernel = stroke::Cov2<scalar_t>(filter_kernel_size);
    const auto fun = [filter_kernel](const whack::Tensor<scalar_t, 1>& input) {
        const auto cov = stroke::extract<cov2_t>(input);
        const auto [filtered_cov, weight_factor] = dgmr::math::convolve_unnormalised_with_normalised<2, scalar_t>(cov, filter_kernel);
        return stroke::pack_tensor<scalar_t>(filtered_cov, weight_factor);
    };

    const auto fun_grad = [filter_kernel](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
        const auto cov = stroke::extract<cov2_t>(input);
        const auto [grad_filtered_cov, grad_weight_factor] = stroke::extract<cov2_t, scalar_t>(grad_output);
        const auto grad_cov = grad::convolve_unnormalised_with_normalised<2, scalar_t>(cov, filter_kernel, grad_filtered_cov, grad_weight_factor);
        return stroke::pack_tensor<scalar_t>(grad_cov);
    };

    for (int i = 0; i < 10; ++i) {
        const auto test_data = stroke::pack_tensor<scalar_t>(stroke::host_random_cov<2, scalar_t>(&rnd) + stroke::Cov2<scalar_t>(0.0)); // make it better conditinoed by adding something
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
    if (filter_kernel_size > 0) {
        const auto test_data = stroke::pack_tensor<scalar_t>(stroke::Cov2<scalar_t>(0.0)); // test max branch
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
            const auto p = dgmr::math::ndc2screen<scalar_t>(pos, width, height);
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
            const auto p1 = dgmr::math::make_jakobian<scalar_t>(pos, l_prime, 1, 1);
            const auto p2 = dgmr::math::make_jakobian<scalar_t>(pos, l_prime, cam.focal_x, cam.focal_y);
            return stroke::pack_tensor<scalar_t>(p1, p2);
            // return stroke::pack_tensor<scalar_t>(p1);
        };

        const auto fun_grad = [cam](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
            const auto [pos, l_prime] = stroke::extract<vec3_t, scalar_t>(input);
            const auto [grad_incoming1, grad_incoming2] = stroke::extract<glm::mat<3, 3, scalar_t>, glm::mat<3, 3, scalar_t>>(grad_output);
            // const auto grad_incoming1 = stroke::extract<glm::mat<3, 3, scalar_t>>(grad_output);
            const auto [grad_pos1, grad_l_prime1] = grad::make_jakobian<scalar_t>(pos, l_prime, grad_incoming1, 1, 1);
            const auto [grad_pos2, grad_l_prime2] = grad::make_jakobian<scalar_t>(pos, l_prime, grad_incoming2, cam.focal_x, cam.focal_y);
            return stroke::pack_tensor<scalar_t>(grad_pos1 + grad_pos2, grad_l_prime1 + grad_l_prime2);
            // return stroke::pack_tensor<scalar_t>(grad_pos1, grad_l_prime1);
        };

        const auto pos = rnd.normal3() + vec3_t(0, 0, 1.5);
        const auto test_data = stroke::pack_tensor<scalar_t>(pos, length(pos));
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.0000001));
    }
}
void check_compute_cov()
{
    using scalar_t = double;
    using vec3_t = glm::vec<3, scalar_t>;
    using cov3_t = stroke::Cov3<scalar_t>;
    using quat_t = glm::qua<scalar_t>;

    whack::random::HostGenerator<scalar_t> rnd;

    for (int i = 0; i < 10; ++i) {
        const auto fun = [](const whack::Tensor<scalar_t, 1>& input) {
            const auto [scales, rotation] = stroke::extract<vec3_t, quat_t>(input);
            const auto cov = dgmr::math::compute_cov<scalar_t>(scales, rotation);
            return stroke::pack_tensor<scalar_t>(cov);
        };

        const auto fun_grad = [](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
            const auto [scales, rotation] = stroke::extract<vec3_t, quat_t>(input);
            const auto grad_incoming = stroke::extract<cov3_t>(grad_output);
            const auto [grad_scales, grad_rotation] = grad::compute_cov<scalar_t>(scales, rotation, grad_incoming);
            return stroke::pack_tensor<scalar_t>(grad_scales, grad_rotation);
        };

        const auto test_data = stroke::pack_tensor<scalar_t>(rnd.uniform3(), stroke::host_random_quaternion<scalar_t>(&rnd));
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
}

void check_integrate_exponential2d()
{
    using scalar_t = double;
    using vec2_t = glm::vec<2, scalar_t>;
    using cov2_t = stroke::Cov2<scalar_t>;

    whack::random::HostGenerator<scalar_t> rnd;

    for (int i = 0; i < 10; ++i) {
        const auto fun = [](const whack::Tensor<scalar_t, 1>& input) {
            const auto scales = stroke::extract<vec2_t>(input);
            const auto i = dgmr::math::integrate_exponential<scalar_t>(scales);
            return stroke::pack_tensor<scalar_t>(i);
        };

        const auto fun_grad = [](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
            const auto scales = stroke::extract<vec2_t>(input);
            const auto grad_incoming = stroke::extract<scalar_t>(grad_output);
            const auto grad_outgoing = grad::integrate_exponential<scalar_t>(scales, grad_incoming);
            return stroke::pack_tensor<scalar_t>(grad_outgoing);
        };

        const auto test_data = stroke::pack_tensor<scalar_t>(vec2_t(rnd.uniform(), rnd.uniform()));
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
}

void check_integrate_exponential3d()
{
    using scalar_t = double;
    using vec3_t = glm::vec<3, scalar_t>;
    using cov3_t = stroke::Cov3<scalar_t>;

    whack::random::HostGenerator<scalar_t> rnd;

    for (int i = 0; i < 10; ++i) {
        const auto fun = [](const whack::Tensor<scalar_t, 1>& input) {
            const auto scales = stroke::extract<vec3_t>(input);
            const auto i = dgmr::math::integrate_exponential<scalar_t>(scales);
            return stroke::pack_tensor<scalar_t>(i);
        };

        const auto fun_grad = [](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
            const auto scales = stroke::extract<vec3_t>(input);
            const auto grad_incoming = stroke::extract<scalar_t>(grad_output);
            const auto grad_outgoing = grad::integrate_exponential<scalar_t>(scales, grad_incoming);
            return stroke::pack_tensor<scalar_t>(grad_outgoing);
        };

        const auto test_data = stroke::pack_tensor<scalar_t>(vec3_t(rnd.uniform(), rnd.uniform(), rnd.uniform()));
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
}

void check_larger2()
{
    using scalar_t = double;
    using vec2_t = glm::vec<2, scalar_t>;
    using vec3_t = glm::vec<3, scalar_t>;

    whack::random::HostGenerator<scalar_t> rnd;

    for (int i = 0; i < 10; ++i) {
        const auto fun = [](const whack::Tensor<scalar_t, 1>& input) {
            const auto scales = stroke::extract<vec3_t>(input);
            const auto i = dgmr::math::larger2<scalar_t>(scales);
            return stroke::pack_tensor<scalar_t>(i);
        };

        const auto fun_grad = [](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
            const auto scales = stroke::extract<vec3_t>(input);
            const auto grad_incoming = stroke::extract<vec2_t>(grad_output);
            const auto grad_outgoing = grad::larger2<scalar_t>(scales, grad_incoming);
            return stroke::pack_tensor<scalar_t>(grad_outgoing);
        };

        const auto test_data = stroke::pack_tensor<scalar_t>(vec3_t(rnd.uniform(), rnd.uniform(), rnd.uniform()));
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
}

} // namespace

TEST_CASE("dgmr splat gradient (formulation: Opacity, filter kernel size: 0)")
{
    check_splat<dgmr::Formulation::Opacity>(0);
}
TEST_CASE("dgmr splat gradient (formulation: Mass, filter kernel size: 0)")
{
    check_splat<dgmr::Formulation::Mass>(0);
}
TEST_CASE("dgmr splat gradient (formulation: Density, filter kernel size: 0)")
{
    check_splat<dgmr::Formulation::Density>(0);
}
TEST_CASE("dgmr splat gradient (formulation: Ots, filter kernel size: 0)")
{
    check_splat<dgmr::Formulation::Ots>(0);
}
TEST_CASE("dgmr splat gradient (formulation: Opacity, filter kernel size: 0.3)")
{
    check_splat<dgmr::Formulation::Opacity>(0.3);
}
TEST_CASE("dgmr splat gradient (formulation: Mass, filter kernel size: 0.3)")
{
    check_splat<dgmr::Formulation::Mass>(0.3);
}
TEST_CASE("dgmr splat gradient (formulation: Density, filter kernel size: 0.3)")
{
    check_splat<dgmr::Formulation::Density>(0.3);
}
TEST_CASE("dgmr splat gradient (formulation: Ots, filter kernel size: 0.3)")
{
    check_splat<dgmr::Formulation::Ots>(0.3);
}

// TEST_CASE("dgmr splat gradient (cached, with orientation dependence, filter kernel size 0)")
// {
//     check_splat_cached<true>(0);
// }
// TEST_CASE("dgmr splat gradient (cached, with orientation dependence, filter kernel size 0.3)")
// {
//     check_splat_cached<true>(0.3);
// }
// TEST_CASE("dgmr splat gradient (cached, withOUT orientation dependence, filter kernel size 0)")
// {
//     check_splat_cached<false>(0);
// }
// TEST_CASE("dgmr splat gradient (cached, withOUT orientation dependence, filter kernel size 0.3)")
// {
//     check_splat_cached<false>(0.3);
// }

TEST_CASE("dgmr project gradient")
{
    check_project();
}

TEST_CASE("dgmr convolve_unnormalised_with_normalised")
{
    check_convolve_unnormalised_with_normalised(0.3);
    check_convolve_unnormalised_with_normalised(0);
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

TEST_CASE("dgmr check_compute_cov gradient")
{
    check_compute_cov();
}

TEST_CASE("dgmr integrate_exponential gradient")
{
    check_integrate_exponential2d();
    check_integrate_exponential3d();
}

TEST_CASE("dgmr larger2 gradient")
{
    check_larger2();
}
