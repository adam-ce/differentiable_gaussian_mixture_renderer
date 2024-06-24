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

#include <stroke/unittest/gradcheck.h> // must go first (so that pretty printers work)

#include <catch2/catch_test_macros.hpp>

#include <dgmr/grad/math.h>
#include <dgmr/math.h>

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

template <dgmr::Formulation formulation>
void check_weight_to_mass()
{
    using scalar_t = double;
    using vec3_t = glm::vec<3, scalar_t>;
    using cov3_t = stroke::Cov3<scalar_t>;
    using quat_t = glm::qua<scalar_t>;

    whack::random::HostGenerator<scalar_t> rnd;

    for (int i = 0; i < 10; ++i) {
        const auto fun = [](const whack::Tensor<scalar_t, 1>& input) {
            const auto [weight, scales] = stroke::extract<scalar_t, vec3_t>(input);
            scalar_t g = weight_to_mass<formulation, scalar_t>(weight, scales);
            return stroke::pack_tensor<scalar_t>(g);
        };

        const auto fun_grad = [](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
            const auto [weight, scales] = stroke::extract<scalar_t, vec3_t>(input);
            const scalar_t grad_incoming = stroke::extract<scalar_t>(grad_output);
            const auto grad_outgoing = grad::weight_to_mass<formulation, scalar_t>(weight, scales, grad_incoming);
            return stroke::pack_tensor<scalar_t>(grad_outgoing);
        };

        const auto test_data = stroke::pack_tensor<scalar_t>(rnd.uniform(), rnd.uniform3());
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000002), scalar_t(100)); // i think the gradient is correct, but a bit unstable.
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

void check_smaller2()
{
    using scalar_t = double;
    using vec2_t = glm::vec<2, scalar_t>;
    using vec3_t = glm::vec<3, scalar_t>;

    whack::random::HostGenerator<scalar_t> rnd;

    for (int i = 0; i < 10; ++i) {
        const auto fun = [](const whack::Tensor<scalar_t, 1>& input) {
            const auto scales = stroke::extract<vec3_t>(input);
            const auto i = dgmr::math::smaller2<scalar_t>(scales);
            return stroke::pack_tensor<scalar_t>(i);
        };

        const auto fun_grad = [](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
            const auto scales = stroke::extract<vec3_t>(input);
            const auto grad_incoming = stroke::extract<vec2_t>(grad_output);
            const auto grad_outgoing = grad::smaller2<scalar_t>(scales, grad_incoming);
            return stroke::pack_tensor<scalar_t>(grad_outgoing);
        };

        const auto test_data = stroke::pack_tensor<scalar_t>(vec3_t(rnd.uniform(), rnd.uniform(), rnd.uniform()));
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
}

template <unsigned sh_degree>
void check_sh_to_color()
{
    using scalar_t = double;
    using vec3_t = glm::vec<3, scalar_t>;

    whack::random::HostGenerator<scalar_t> rnd;
    const auto gen_harmonic = [&]() {
        dgmr::SHs<3, scalar_t> retval = {};
        for (auto i = 0u; i < retval.size(); ++i) {
            // if (is_special_first) { // not differentiable iff the sh gives exactly 0. no need to test. leaving comment for future reference.
            //     retval[i] = vec3_t(-.5 / 0.28209479177387814);
            //     is_special_first = false;
            //     continue;
            // }
            retval[i] = rnd.normal3() * 10.;
        }
        return retval;
    };

    for (int i = 0; i < 10; ++i) {
        const auto fun = [&](const whack::Tensor<scalar_t, 1>& input) {
            const auto [sh, dir] = stroke::extract<dgmr::SHs<3, scalar_t>, vec3_t>(input);
            const auto [rgb, clamped] = dgmr::math::sh_to_colour(sh, sh_degree, dir);
            return stroke::pack_tensor<scalar_t>(rgb);
        };

        const auto fun_grad = [&](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
            const auto [sh, dir] = stroke::extract<dgmr::SHs<3, scalar_t>, vec3_t>(input);
            const auto [rgb, clamped] = dgmr::math::sh_to_colour(sh, sh_degree, dir);

            const auto grad_incoming = stroke::extract<vec3_t>(grad_output);
            const auto grad_outgoing = dgmr::math::grad::sh_to_colour(sh, sh_degree, dir, grad_incoming, clamped);
            return stroke::pack_tensor<scalar_t>(grad_outgoing);
        };

        const auto test_data = stroke::pack_tensor<scalar_t>(gen_harmonic(), normalize(rnd.normal3()));
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.001));
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
TEST_CASE("dgmr splat gradient (formulation: Ols, filter kernel size: 0)")
{
    check_splat<dgmr::Formulation::Ols>(0);
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
TEST_CASE("dgmr splat gradient (formulation: Ols, filter kernel size: 0.3)")
{
    check_splat<dgmr::Formulation::Ols>(0.3);
}

TEST_CASE("dgmr weight_to_mass gradient (formulation: Mass)")
{
    check_weight_to_mass<dgmr::Formulation::Mass>();
}
TEST_CASE("dgmr weight_to_mass gradient (formulation: Density)")
{
    check_weight_to_mass<dgmr::Formulation::Density>();
}
TEST_CASE("dgmr weight_to_mass gradient (formulation: Ots)")
{
    check_weight_to_mass<dgmr::Formulation::Ots>();
}
TEST_CASE("dgmr weight_to_mass gradient (formulation: Ols)")
{
    check_weight_to_mass<dgmr::Formulation::Ols>();
}

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

TEST_CASE("dgmr smaller2 gradient")
{
    check_smaller2();
}

TEST_CASE("dgmr integrate_bins gradient 1 bin array, 1 bin")
{
    using scalar_t = double;
    using Arr = whack::Array<glm::dvec4, 1>;

    whack::random::HostGenerator<scalar_t> rnd;

    const auto fun = [](const whack::Tensor<scalar_t, 1>& input) {
        const auto bins0 = stroke::extract<Arr>(input);
        const auto [color0, transparency0] = dgmr::math::integrate_bins(glm::dvec3(0), 1., bins0);
        return stroke::pack_tensor<scalar_t>(color0, transparency0);
    };

    const auto fun_grad = [](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
        const auto bins0 = stroke::extract<Arr>(input);
        const auto [color0, transparency0] = dgmr::math::integrate_bins(glm::dvec3(0), 1., bins0);
        const auto [incoming_grad_colour, incoming_grad_transparency] = stroke::extract<glm::dvec3, scalar_t>(grad_output);
        const auto [remaining_color0, remaining_transparency0, grad_bins0] = dgmr::math::grad::integrate_bins(color0, 1.0, transparency0, bins0, incoming_grad_colour, incoming_grad_transparency);
        return stroke::pack_tensor<scalar_t>(grad_bins0);
    };

    for (int i = 0; i < 10; ++i) {
        const auto rnd0 = rnd.uniform();
        const auto first = Arr {
            glm::dvec4(rnd.uniform() * rnd0, rnd.uniform() * rnd0, rnd.uniform() * rnd0, rnd0),
        };
        const auto test_data = stroke::pack_tensor<scalar_t>(first);
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
}

TEST_CASE("dgmr integrate_bins gradient 1 bin array 2 bins")
{
    using scalar_t = double;
    using Arr = whack::Array<glm::dvec4, 2>;

    whack::random::HostGenerator<scalar_t> rnd;

    const auto fun = [](const whack::Tensor<scalar_t, 1>& input) {
        const auto bins0 = stroke::extract<Arr>(input);
        const auto [color0, transparency0] = dgmr::math::integrate_bins(glm::dvec3(0), 1., bins0);
        return stroke::pack_tensor<scalar_t>(color0, transparency0);
    };

    const auto fun_grad = [](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
        const auto bins0 = stroke::extract<Arr>(input);
        const auto [color0, transparency0] = dgmr::math::integrate_bins(glm::dvec3(0), 1., bins0);
        const auto [incoming_grad_colour, incoming_grad_transparency] = stroke::extract<glm::dvec3, scalar_t>(grad_output);

        const auto [remaining_color0, remaining_transparency0, grad_bins0] = dgmr::math::grad::integrate_bins(color0, 1.0, transparency0, bins0, incoming_grad_colour, incoming_grad_transparency);
        return stroke::pack_tensor<scalar_t>(grad_bins0);
    };

    for (int i = 0; i < 10; ++i) {

        const auto rnd0 = rnd.uniform();
        const auto rnd1 = rnd.uniform();
        const auto first = Arr {
            glm::dvec4(rnd.uniform() * rnd0, rnd.uniform() * rnd0, rnd.uniform() * rnd0, rnd0),
            glm::dvec4(rnd.uniform() * rnd1, rnd.uniform() * rnd1, rnd.uniform() * rnd1, rnd1),
        };
        const auto test_data = stroke::pack_tensor<scalar_t>(first);
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
}

TEST_CASE("dgmr integrate_bins gradient 1 bin array 4 bins")
{
    using scalar_t = double;
    using Arr = whack::Array<glm::dvec4, 4>;

    whack::random::HostGenerator<scalar_t> rnd;

    const auto fun = [](const whack::Tensor<scalar_t, 1>& input) {
        const auto bins0 = stroke::extract<Arr>(input);
        const auto [color0, transparency0] = dgmr::math::integrate_bins(glm::dvec3(0), 1., bins0);
        return stroke::pack_tensor<scalar_t>(color0, transparency0);
    };

    const auto fun_grad = [](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
        const auto bins0 = stroke::extract<Arr>(input);
        const auto [color0, transparency0] = dgmr::math::integrate_bins(glm::dvec3(0), 1., bins0);
        const auto [incoming_grad_colour, incoming_grad_transparency] = stroke::extract<glm::dvec3, scalar_t>(grad_output);

        const auto [remaining_color0, remaining_transparency0, grad_bins0] = dgmr::math::grad::integrate_bins(color0, 1.0, transparency0, bins0, incoming_grad_colour, incoming_grad_transparency);
        return stroke::pack_tensor<scalar_t>(grad_bins0);
    };

    for (int i = 0; i < 10; ++i) {

        const auto rnd0 = rnd.uniform();
        const auto rnd1 = rnd.uniform();
        const auto rnd2 = rnd.uniform();
        const auto rnd3 = rnd.uniform();
        const auto first = Arr {
            glm::dvec4(rnd.uniform() * rnd0, rnd.uniform() * rnd0, rnd.uniform() * rnd0, rnd0),
            glm::dvec4(rnd.uniform() * rnd1, rnd.uniform() * rnd1, rnd.uniform() * rnd1, rnd1),
            glm::dvec4(rnd.uniform() * rnd2, rnd.uniform() * rnd2, rnd.uniform() * rnd2, rnd2),
            glm::dvec4(rnd.uniform() * rnd3, rnd.uniform() * rnd3, rnd.uniform() * rnd3, rnd3),
        };
        const auto test_data = stroke::pack_tensor<scalar_t>(first);
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
}

TEST_CASE("dgmr integrate_bins gradient 2 bin array 2 bins")
{
    using scalar_t = double;
    using Arr = whack::Array<glm::dvec4, 2>;

    whack::random::HostGenerator<scalar_t> rnd;

    const auto fun = [](const whack::Tensor<scalar_t, 1>& input) {
        const auto [bins0, bins1] = stroke::extract<Arr, Arr>(input);
        const auto [color0, transparency0] = dgmr::math::integrate_bins(glm::dvec3(0), 1., bins0);
        const auto [color1, transparency1] = dgmr::math::integrate_bins(color0, transparency0, bins1);
        return stroke::pack_tensor<scalar_t>(color1, transparency1);
    };

    const auto fun_grad = [](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
        const auto [bins0, bins1] = stroke::extract<Arr, Arr>(input);
        const auto [color0, transparency0] = dgmr::math::integrate_bins(glm::dvec3(0), 1., bins0);
        const auto [color1, transparency1] = dgmr::math::integrate_bins(color0, transparency0, bins1);

        const auto [incoming_grad_colour, incoming_grad_transparency] = stroke::extract<glm::dvec3, scalar_t>(grad_output);

        const auto [remaining_color0, remaining_transparency0, grad_bins0] = dgmr::math::grad::integrate_bins(color1, 1.0, transparency1, bins0, incoming_grad_colour, incoming_grad_transparency);

        const auto [remaining_color1, remaining_transparency1, grad_bins1] = dgmr::math::grad::integrate_bins(remaining_color0, remaining_transparency0, transparency1, bins1, incoming_grad_colour, incoming_grad_transparency);
        return stroke::pack_tensor<scalar_t>(grad_bins0, grad_bins1);
    };

    for (int i = 0; i < 10; ++i) {

        const auto rnd0 = rnd.uniform();
        const auto rnd1 = rnd.uniform();
        const auto rnd2 = rnd.uniform();
        const auto rnd3 = rnd.uniform();
        const auto a0 = Arr {
            glm::dvec4(rnd.uniform() * rnd0, rnd.uniform() * rnd0, rnd.uniform() * rnd0, rnd0),
            glm::dvec4(rnd.uniform() * rnd1, rnd.uniform() * rnd1, rnd.uniform() * rnd1, rnd1),
        };
        const auto a1 = Arr {
            glm::dvec4(rnd.uniform() * rnd2, rnd.uniform() * rnd2, rnd.uniform() * rnd2, rnd2),
            glm::dvec4(rnd.uniform() * rnd3, rnd.uniform() * rnd3, rnd.uniform() * rnd3, rnd3),
        };
        const auto test_data = stroke::pack_tensor<scalar_t>(a0, a1);
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
}

TEST_CASE("dgmr integrate_bins gradient 2 bin array 256 bins")
{
    using scalar_t = double;
    using Arr = whack::Array<glm::dvec4, 256>;

    whack::random::HostGenerator<scalar_t> rnd;

    const auto fun = [](const whack::Tensor<scalar_t, 1>& input) {
        const auto [bins0, bins1] = stroke::extract<Arr, Arr>(input);
        const auto [color0, transparency0] = dgmr::math::integrate_bins(glm::dvec3(0), 1., bins0);
        const auto [color1, transparency1] = dgmr::math::integrate_bins(color0, transparency0, bins1);
        return stroke::pack_tensor<scalar_t>(color1, transparency1);
    };

    const auto fun_grad = [](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
        const auto [bins0, bins1] = stroke::extract<Arr, Arr>(input);
        const auto [color0, transparency0] = dgmr::math::integrate_bins(glm::dvec3(0), 1., bins0);
        const auto [color1, transparency1] = dgmr::math::integrate_bins(color0, transparency0, bins1);

        const auto [incoming_grad_colour, incoming_grad_transparency] = stroke::extract<glm::dvec3, scalar_t>(grad_output);

        const auto [remaining_color0, remaining_transparency0, grad_bins0] = dgmr::math::grad::integrate_bins(color1, 1.0, transparency1, bins0, incoming_grad_colour, incoming_grad_transparency);

        const auto [remaining_color1, remaining_transparency1, grad_bins1] = dgmr::math::grad::integrate_bins(remaining_color0, remaining_transparency0, transparency1, bins1, incoming_grad_colour, incoming_grad_transparency);
        return stroke::pack_tensor<scalar_t>(grad_bins0, grad_bins1);
    };

    for (int i = 0; i < 10; ++i) {

        const auto a0 = [&]() -> Arr {
            Arr a;
            for (auto& v : a) {
                const auto rnd0 = rnd.uniform() / 32.;
                v = glm::dvec4(rnd.uniform() * rnd0, rnd.uniform() * rnd0, rnd.uniform() * rnd0, rnd0);
            }
            return a;
        }();
        const auto a1 = [&]() -> Arr {
            Arr a;
            for (auto& v : a) {
                const auto rnd0 = rnd.uniform() / 32.;
                v = glm::dvec4(rnd.uniform() * rnd0, rnd.uniform() * rnd0, rnd.uniform() * rnd0, rnd0);
            }
            return a;
        }();
        const auto test_data = stroke::pack_tensor<scalar_t>(a0, a1);
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
}

TEST_CASE("dgmr sample_gaussian grad")
{
    using scalar_t = double;
    using vec3_t = glm::vec<3, scalar_t>;
    using vec4_t = glm::vec<4, scalar_t>;
    using cov3_t = stroke::Cov3<scalar_t>;
    using Gaussian1d = stroke::gaussian::ParamsWithWeight<1, scalar_t>;

    constexpr auto n_bins = 4u;

    whack::random::HostGenerator<scalar_t> rnd;
    const auto gen_sampling = [&]() {
        whack::Array<scalar_t, n_bins + 1> retval = {};
        retval[0] = rnd.uniform() < 0.2 ? 0. : rnd.uniform();
        for (auto i = 1u; i < retval.size(); ++i) {
            retval[i] = retval[i - 1] + rnd.uniform() * 5. * 4. / n_bins;
        }
        return retval;
    };

    for (int i = 0; i < 10; ++i) {
        const auto g_weight = rnd.uniform() * 10;
        const auto rgb = rnd.uniform3();
        const auto g_pos = rnd.normal3();
        const auto g_cov = inverse(stroke::host_random_cov<3, double>(&rnd) * 1.);

        const auto ray_pos = rnd.normal3() * 5.;
        const auto target_pos = g_pos + rnd.normal3() * 0.5;
        const auto ray = stroke::Ray<3, scalar_t> { ray_pos, normalize(target_pos - ray_pos) };
        const auto bin_borders = gen_sampling();

        const auto fun = [&](const whack::Tensor<scalar_t, 1>& input) {
            const auto [g_weight, rgb, g_pos, g_cov] = stroke::extract<scalar_t, vec3_t, vec3_t, cov3_t>(input);
            whack::Array<vec4_t, n_bins> samples = {};
            dgmr::math::sample_gaussian(g_weight, rgb, g_pos, g_cov, ray, bin_borders, &samples);
            return stroke::pack_tensor<scalar_t>(samples);
        };

        const auto fun_grad = [&](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
            const auto [g_weight, rgb, g_pos, g_cov] = stroke::extract<scalar_t, vec3_t, vec3_t, cov3_t>(input);
            const auto grad_incoming = stroke::extract<whack::Array<vec4_t, n_bins>>(grad_output);
            const auto grad_outgoing = dgmr::math::grad::sample_gaussian(g_weight, rgb, g_pos, g_cov, ray, bin_borders, grad_incoming);
            return stroke::pack_tensor<scalar_t>(grad_outgoing);
        };

        const auto test_data = stroke::pack_tensor<scalar_t>(g_weight, rgb, g_pos, g_cov);
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
}

TEST_CASE("dgmr spherical harmonics grad degree 0")
{
    check_sh_to_color<0>();
}

TEST_CASE("dgmr spherical harmonics grad degree 1")
{
    check_sh_to_color<1>();
}

TEST_CASE("dgmr spherical harmonics grad degree 2")
{
    check_sh_to_color<2>();
}

TEST_CASE("dgmr spherical harmonics grad degree 3")
{
    check_sh_to_color<3>();
}
