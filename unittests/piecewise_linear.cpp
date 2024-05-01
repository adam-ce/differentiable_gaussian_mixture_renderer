/*****************************************************************************
 * DGMR
 * Copyright (C) 2024 Adam Celarek
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

#include <array>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <dgmr/piecewise_linear.h>

#include <stroke/linalg.h>

TEST_CASE("dgmr piecewise_linear function group")
{
    const auto fun = dgmr::piecewise_linear::FunctionGroup<double> {
        glm::dvec4(0.8, 0.4, 0.5, 0.2), // d0
        glm::dvec4(-0.8, -0.2, 0.5, 0.0), // k0
        glm::dvec4(1.0, 2.0, 0.5, 2.0), // t1
        glm::dvec4(2.0, 2.0, 0.5, 2.0), // t2
        glm::dvec4(0.1, 0.2, 0.3, 0.4), // k2
    };
    fun.check();
    CHECK(glm::ivec4(fun.sample(0.0) * 1000.0) == glm::ivec4(glm::dvec4(0.8, 0.4, 0.50, 0.2) * 1000.0));
    CHECK(glm::ivec4(fun.sample(0.5) * 1000.0) == glm::ivec4(glm::dvec4(0.4, 0.3, 0.75, 0.2) * 1000.0));
    CHECK(glm::ivec4(fun.sample(1.0) * 1000.0) == glm::ivec4(glm::dvec4(0.0, 0.2, 0.90, 0.2) * 1000.0));
    CHECK(glm::ivec4(fun.sample(2.0) * 1000.0) == glm::ivec4(glm::dvec4(0.0, 0.0, 1.20, 0.2) * 1000.0));
    CHECK(glm::ivec4(fun.sample(3.0) * 1000.0) == glm::ivec4(glm::dvec4(0.1, 0.2, 1.50, 0.6) * 1000.0));
}

TEST_CASE("dgmr piecewise_linear function fitting 1")
{
    const auto masses_percent_left = glm::dvec4(0.8, 0.4, 0.5, 0.2);
    const auto masses_total = glm::dvec4(0.3, 0.6, 0.3, 0.8);
    const auto eval_left = glm::dvec4(0.4, 0.7, 0.2, 0.9);
    const auto eval_right = glm::dvec4(0.2, 0.1, 0.0, 0.1);
    const auto t_right = 0.8;
    const auto td = 0.001;
    const auto lin_approx = dgmr::piecewise_linear::create_approximation(masses_percent_left, masses_total, eval_left, eval_right, t_right);
    lin_approx.check();

    CHECK(glm::ivec4(lin_approx.sample(0.0) * 1000.0 + 0.5) == glm::ivec4(eval_left * 1000.0 + 0.5));
    CHECK(glm::ivec4(lin_approx.sample(t_right) * 1000.0 + 0.5) == glm::ivec4(eval_right * 1000.0 + 0.5));
    auto numeric_masses_total = glm::dvec4();
    for (double t = 0; t < t_right; t += td) {
        numeric_masses_total += lin_approx.sample(t) * td;
    }
    CHECK(glm::ivec4(numeric_masses_total * 1000.0) == glm::ivec4(masses_total * 1000.0));
}

TEST_CASE("dgmr piecewise_linear function fitting 2")
{
    const auto masses_percent_left = glm::dvec4(0.5, 0.5, 0.5, 0.5);
    const auto masses_total = glm::dvec4(0.8, 0.8, 0.8, 0.0);
    const auto eval_left = glm::dvec4(0.0, 0.0, 0.2, 0.0);
    const auto eval_right = glm::dvec4(0.0, 0.1, 0.0, 0.0);
    const auto t_right = 0.8;
    const auto td = 0.000001;
    const auto lin_approx = dgmr::piecewise_linear::create_approximation(masses_percent_left, masses_total, eval_left, eval_right, t_right);
    lin_approx.check();

    CHECK(glm::ivec4(lin_approx.sample(0.0) * 1000.0 + 0.5) == glm::ivec4(eval_left * 1000.0 + 0.5));
    CHECK(glm::ivec4(lin_approx.sample(t_right) * 1000.0 + 0.5) == glm::ivec4(eval_right * 1000.0 + 0.5));
    auto numeric_masses_total = glm::dvec4();
    for (double t = 0; t < t_right; t += td) {
        numeric_masses_total += lin_approx.sample(t) * td;
    }
    CHECK(glm::ivec4(numeric_masses_total * 1000.0 + 0.5) == glm::ivec4(masses_total * 1000.0 + 0.5));
}

TEST_CASE("dgmr piecewise_linear vol int")
{
    constexpr auto n_ref_steps = 1000000;
    constexpr auto n_steps_per_bin = 256;

    // create 8 bins with piecewise_linear FunctionGroups
    const whack::Array<dgmr::piecewise_linear::FunctionGroup<double>, 8> bins = {
        //                                           masses_percent_left,            masses_total,                   eval_left,                      eval_right,                     t_right
        dgmr::piecewise_linear::create_approximation(glm::dvec4(0.0, 0.0, 0.0, 0.0), glm::dvec4(0.3, 0.6, 0.3, 0.6), glm::dvec4(0.0, 0.0, 0.0, 0.0), glm::dvec4(0.2, 0.1, 0.0, 0.1), 0.8),
        dgmr::piecewise_linear::create_approximation(glm::dvec4(0.8, 0.4, 0.5, 0.3), glm::dvec4(0.5, 0.4, 0.8, 0.2), glm::dvec4(0.2, 0.1, 0.0, 0.1), glm::dvec4(0.4, 0.7, 0.2, 0.9), 0.5),
        dgmr::piecewise_linear::create_approximation(glm::dvec4(0.6, 0.7, 0.1, 0.8), glm::dvec4(0.1, 0.0, 0.1, 0.1), glm::dvec4(0.4, 0.7, 0.2, 0.9), glm::dvec4(0.8, 0.1, 0.8, 0.7), 1.5),
        dgmr::piecewise_linear::create_approximation(glm::dvec4(0.1, 0.8, 0.7, 0.5), glm::dvec4(0.1, 0.2, 0.0, 0.2), glm::dvec4(0.8, 0.1, 0.8, 0.7), glm::dvec4(0.4, 0.7, 0.2, 0.1), 2.5),
        dgmr::piecewise_linear::create_approximation(glm::dvec4(0.5, 0.2, 0.6, 0.3), glm::dvec4(0.5, 0.6, 0.7, 0.7), glm::dvec4(0.4, 0.7, 0.2, 0.1), glm::dvec4(0.4, 0.7, 0.2, 0.2), 0.5),
        dgmr::piecewise_linear::create_approximation(glm::dvec4(0.7, 0.8, 0.2, 0.5), glm::dvec4(0.5, 0.2, 0.3, 0.5), glm::dvec4(0.4, 0.7, 0.2, 0.2), glm::dvec4(0.4, 0.7, 0.2, 1.9), 0.8),
        dgmr::piecewise_linear::create_approximation(glm::dvec4(0.8, 0.4, 0.8, 0.2), glm::dvec4(0.3, 1.6, 1.3, 1.9), glm::dvec4(0.4, 0.7, 0.2, 1.9), glm::dvec4(0.4, 0.7, 0.2, 0.5), 0.1),
        dgmr::piecewise_linear::create_approximation(glm::dvec4(0.1, 0.1, 0.4, 0.9), glm::dvec4(0.8, 0.7, 0.3, 2.4), glm::dvec4(0.4, 0.7, 0.2, 0.5), glm::dvec4(0.2, 0.1, 0.0, 0.1), 1.2),
    };

    double max_t = 0;
    for (const auto& f : bins) {
        max_t += f.t_right;
    }
    double delta_t = max_t / n_ref_steps;

    // int full numeric Eq 15
    const auto eval = [&](double t) {
        for (const auto& f : bins) {
            if (f.t_right < t) {
                t -= f.t_right;
                continue;
            }
            return f.sample(t);
        }
        return glm::dvec4();
    };

    glm::dvec3 reference_vol_int = {};
    double t = 0;
    double current_m = 0;
    for (unsigned i = 0; i < n_ref_steps; ++i) {
        const auto eval_t = eval(t);
        reference_vol_int += glm::dvec3(eval_t) * stroke::exp(-current_m) * delta_t;
        current_m += eval_t.w * delta_t;
        t += delta_t;
    }
    CHECK(current_m > 0.0);
    CHECK(current_m < 20.0);
    CHECK(reference_vol_int.x >= 0.0);
    CHECK(reference_vol_int.y >= 0.0);
    CHECK(reference_vol_int.z >= 0.0);
    CHECK(reference_vol_int.x < 2.0);
    CHECK(reference_vol_int.y < 2.0);
    CHECK(reference_vol_int.z < 2.0);

    // compare with faster implementation
    const auto vol_int = dgmr::piecewise_linear::volume_integrate(bins, n_steps_per_bin);
    CHECK(stroke::abs(vol_int.x - reference_vol_int.x) < 1.0 / n_steps_per_bin);
    CHECK(stroke::abs(vol_int.y - reference_vol_int.y) < 1.0 / n_steps_per_bin);
    CHECK(stroke::abs(vol_int.z - reference_vol_int.z) < 1.0 / n_steps_per_bin);
}
