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

TEST_CASE("dgmr grad marching steps next sample")
{
    using scalar_t = double;
    using DensityArray = dgmr::marching_steps::DensityArray<8, scalar_t>;
    constexpr auto n_samples_per_g = 10;

    whack::random::HostGenerator<scalar_t> rnd;

    std::srand(0);
    const auto r = []() {
        return scalar_t(std::rand()) / scalar_t(RAND_MAX);
    };
    for (auto i = 0u; i < 10; ++i) {
        const auto g_start = r() * 5;
        const auto current_start = r() * 5;
        const auto current_end = current_start + r() * 5;
        const auto g_end = current_end + r() * 5;
        const auto delta_t = (g_end - g_start) / (n_samples_per_g - 1);
        const auto t = r() * g_end;

        const auto fun = [=](const whack::Tensor<scalar_t, 1>& input) {
            const auto [g_start, delta_t] = stroke::extract<scalar_t, scalar_t>(input);
            DensityEntry<scalar_t> d { 0, current_start, current_end, delta_t };
            d.g_start = g_start;
            const auto sample = dgmr::marching_steps::next_sample<scalar_t>(d, t);
            return stroke::pack_tensor<scalar_t>(sample);
        };

        const auto fun_grad = [=](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
            const auto [g_start, delta_t] = stroke::extract<scalar_t, scalar_t>(input);
            DensityEntry<scalar_t> d { 0, current_start, current_end, delta_t };
            d.g_start = g_start;

            const auto grad_sample = stroke::extract<scalar_t>(grad_output);

            const auto grad_density = dgmr::marching_steps::grad::next_sample<scalar_t>(d, t, grad_sample);

            return stroke::pack_tensor<scalar_t>(grad_density.g_start, grad_density.delta_t);
        };
        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<scalar_t>(g_start, delta_t), scalar_t(0.000001));
    }
}

TEST_CASE("dgmr grad marching steps sample")
{
    using scalar_t = double;
    constexpr auto n_densities = 16;
    using DensityArray = dgmr::marching_steps::DensityArray<n_densities, scalar_t>;
    constexpr auto n_samples = 16u;
    constexpr auto n_samples_per_g = 32;

    whack::random::HostGenerator<scalar_t> rnd;

    std::srand(0);
    const auto r = []() {
        return scalar_t(std::rand()) / scalar_t(RAND_MAX);
    };
    for (auto i = 0u; i < 4; ++i) {
        const auto smallest = r();
        DensityArray arr(smallest);
        for (auto j = 0u; j < 100; ++j) {
            const auto start = r() * scalar_t(10.0);
            const auto end = start + r() + scalar_t(0.2);
            arr.put({ 0, start, end, (end - start) / (n_samples_per_g - 1) });

            auto arr_prime = arr;
            arr_prime.reset_non_differentiable_values(arr.size(), smallest + 0.000002, arr.end());
            for (auto i = 0u; i < arr.size(); ++i) {
                // arr_prime[i].start = stroke::min(arr[i].start, arr_prime[i].g_start);
                // arr_prime[i].end = stroke::max(arr[i].end, arr_prime[i].g_start + n_samples_per_g * arr_prime[i].delta_t);
                arr_prime[i].start = arr[i].start + 0.000002;
                arr_prime[i].end = arr[i].end - n_samples_per_g * 0.000002;
            }

            const auto fun = [=](const whack::Tensor<scalar_t, 1>& input) {
                auto arr_pp = stroke::extract<DensityArray>(input);
                // not computing gradient for decision boundaries (therefore resetting them, so no delta is added for symmetric difference)
                arr_pp.reset_non_differentiable_values(arr_prime.size(), arr_prime.start(), arr_prime.end());
                for (auto i = 0u; i < arr_prime.size(); ++i) {
                    // arr_prime[i].start = stroke::min(arr[i].start, arr_prime[i].g_start);
                    // arr_prime[i].end = stroke::max(arr[i].end, arr_prime[i].g_start + n_samples_per_g * arr_prime[i].delta_t);
                    arr_pp[i].gaussian_id = 0;
                    arr_pp[i].start = arr_prime[i].start;
                    arr_pp[i].end = arr_prime[i].end;
                }
                const auto samples = dgmr::marching_steps::sample<n_samples, n_samples_per_g>(arr_pp);
                return stroke::pack_tensor<scalar_t>(samples);
            };

            const auto fun_grad = [=](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
                const auto arr = stroke::extract<DensityArray>(input);

                const auto grad_samples = stroke::extract<whack::Array<scalar_t, n_samples>>(grad_output);

                auto grad_densities = dgmr::marching_steps::grad::sample<n_samples, n_samples_per_g>(arr, grad_samples);
                grad_densities.set_size(0);

                static_assert(sizeof(grad_densities) == 3 * 8 + n_densities * 5 * 8);
                static_assert(sizeof(scalar_t) == 8);
                static_assert(sizeof(grad_densities) % sizeof(scalar_t) == 0);
                return stroke::pack_tensor<scalar_t>(grad_densities);
            };

            const auto test_data = stroke::pack_tensor<scalar_t>(arr_prime);
            stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.0000001));
        }
    }
}
