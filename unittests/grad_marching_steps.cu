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

#include <dgmr/grad/marching_steps.h>
#include <dgmr/marching_steps.h>

#include "unit_test_utils.h"

using namespace dgmr::marching_steps;
using namespace dgmr::unittest;

TEST_CASE("dgmr grad marching steps sample")
{
    using scalar_t = double;
    using DensityArray = dgmr::marching_steps::DensityArray<8, scalar_t>;
    constexpr auto n_samples = 16u;
    constexpr auto n_samples_per_g = 10;

    whack::random::HostGenerator<scalar_t> rnd;

    const auto fun = [](const whack::Tensor<scalar_t, 1>& input) {
        const auto arr = stroke::extract<DensityArray>(input);
        const auto samples = dgmr::marching_steps::sample<n_samples, n_samples_per_g>(arr);
        return stroke::pack_tensor<scalar_t>(samples);
    };

    const auto fun_grad = [](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
        const auto arr = stroke::extract<DensityArray>(input);

        const auto grad_samples = stroke::extract<whack::Array<scalar_t, n_samples>>(grad_output);

        const auto grad_densities = dgmr::marching_steps::grad::sample<n_samples, n_samples_per_g>(arr, grad_samples);

        return stroke::pack_tensor<scalar_t>(grad_densities);
    };

    std::srand(0);
    const auto r = []() {
        return scalar_t(std::rand()) / scalar_t(RAND_MAX);
    };
    for (auto i = 0u; i < 4; ++i) {
        const auto smallest = r();
        DensityArray arr(smallest);
        for (auto j = 0u; j < 50; ++j) {
            const auto start = r() * scalar_t(10.0);
            const auto end = start + r();
            arr.put({ start, end, (end - start) / (n_samples_per_g - 1) });
            const auto test_data = stroke::pack_tensor<scalar_t>(arr);
            stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
        }
    }
}
