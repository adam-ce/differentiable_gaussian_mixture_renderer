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

#pragma once

#include <gcem.hpp>
#include <stroke/scalar_functions.h>
#include <stroke/utility.h>
#include <whack/array.h>

#include "../marching_steps.h"

namespace dgmr::marching_steps::grad {

template <typename scalar_t>
STROKE_DEVICES_INLINE DensityEntry<scalar_t> next_sample(const DensityEntry<scalar_t>& density, scalar_t t, scalar_t incoming_grad)
{
    DensityEntry<scalar_t> gd = {};
    gd.g_start = incoming_grad;
    if (t < density.g_start)
        return gd;
    unsigned n_steps = stroke::ceil((t - density.g_start) / density.delta_t);
    gd.delta_t = n_steps * incoming_grad;
    return gd;
}

template <unsigned n_samples, unsigned n_samples_per_gaussian, unsigned n_density_sections, typename scalar_t>
STROKE_DEVICES_INLINE DensityArray<n_density_sections, scalar_t> sample(const DensityArray<n_density_sections, scalar_t>& densities, const whack::Array<scalar_t, n_samples>& incoming_grad)
{
    auto grad_densities = DensityArray<n_density_sections, scalar_t>();

    if (densities.size() == 0)
        return grad_densities;

    grad_densities.set_size(densities.size());
    for (auto i = 0u; i < densities.size(); ++i) {
        grad_densities[i].gaussian_id = densities[i].gaussian_id;
    }

    unsigned current_density_index = 0;
    unsigned last_sample_density_index = 0;
    scalar_t last_sample = densities.start() - scalar_t(0.000001) - densities[0].delta_t / 2;
    scalar_t last_sample_t = last_sample;
    for (auto i = 0u; i < n_samples; ++i) {
        auto sample_t = last_sample + densities[current_density_index].delta_t / 2;
        auto sample = next_sample(densities[current_density_index], sample_t);
        while (sample > densities[current_density_index].end) {
            ++current_density_index;
            if (current_density_index >= densities.size()) {
                if (i == 0)
                    return grad_densities; // no samples produced -> no gradient

                scalar_t total_grad = 0;
                for (auto j = i; j < n_samples; ++j)
                    total_grad += incoming_grad[j];

                const auto tmp = grad::next_sample(densities[last_sample_density_index], last_sample_t, total_grad);
                grad_densities[last_sample_density_index].g_start += tmp.g_start;
                grad_densities[last_sample_density_index].delta_t += tmp.delta_t;
                return grad_densities;
            }
            sample_t = densities[current_density_index].start - scalar_t(0.000001);
            sample = next_sample(densities[current_density_index], sample_t);
        }
        // samples[i] = sample;

        const auto tmp = grad::next_sample(densities[current_density_index], sample_t, incoming_grad[i]);
        grad_densities[current_density_index].g_start += tmp.g_start;
        grad_densities[current_density_index].delta_t += tmp.delta_t;
        last_sample = sample;
        last_sample_t = sample_t;
        last_sample_density_index = current_density_index;
    }

    return grad_densities;
}

} // namespace dgmr::marching_steps::grad
