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

#include <cstdio>
#include <stroke/cuda_compat.h>
#include <stroke/gaussian.h>
#include <stroke/scalar_functions.h>
#include <stroke/utility.h>
#include <whack/array.h>

#include "math.h"

namespace dgmr::math {

template <typename config>
struct RasterBinSizer_1 {
    static constexpr auto n_bins = config::n_rasterisation_steps;
    static constexpr auto transmission_threshold = config::transmission_threshold;
    whack::Array<float, n_bins> bin_borders = {};
    float transmission = 1.f;
    STROKE_DEVICES_INLINE float begin_of(unsigned i) const
    {
        if (i == 0)
            return 0;
        return bin_borders[i - 1];
    }
    STROKE_DEVICES_INLINE float end_of(unsigned i) const
    {
        return bin_borders[i];
    }
    STROKE_DEVICES_INLINE float max_distance() const
    {
        return bin_borders[n_bins - 1];
    }
    STROKE_DEVICES_INLINE bool is_full() const
    {
        return transmission < transmission_threshold;
    }
    STROKE_DEVICES_INLINE void add_opacity(float pos, float alpha)
    {
        assert(!stroke::isnan(pos));
        assert(!stroke::isnan(alpha));

        const auto bin_for = [](float transmission) {
            const auto t_flipped_and_scaled = 1 - (transmission - transmission_threshold) / (1 - transmission_threshold);
            const auto t_bin = unsigned(stroke::min(t_flipped_and_scaled, 1.0001f) * n_bins);
            return t_bin - 1;
        };
        const auto last_bin = bin_for(transmission);
        transmission *= 1 - alpha;
        const auto current_bin = bin_for(transmission);
        if (last_bin != current_bin && current_bin < n_bins - 1)
            bin_borders[current_bin] = pos;
        bin_borders[n_bins - 1] = stroke::max(pos, bin_borders[n_bins - 1]);
    }
    STROKE_DEVICES_INLINE void add_gaussian(float opacity, float centre, float sd)
    {
        namespace gaussian = stroke::gaussian;
        const auto g_end_position = centre + sd * config::gaussian_relevance_sigma;
        if (g_end_position < 0)
            return;
        const auto alpha = stroke::min(0.99f, opacity * gaussian::integrate_normalised_SD(centre, sd, { 0, g_end_position }));
        add_opacity(g_end_position, alpha);
    }
    STROKE_DEVICES_INLINE void finalise()
    {
        const auto find_empty = [this](unsigned start_pos) {
            for (unsigned i = start_pos; i < n_bins; ++i) {
                if (bin_borders[i] == 0)
                    return i;
            }
            return unsigned(-1);
        };
        const auto find_filled = [this](unsigned start_pos) {
            for (unsigned i = start_pos; i < n_bins; ++i) {
                if (bin_borders[i] != 0)
                    return i;
            }
            return unsigned(-1);
        };

        if (bin_borders[n_bins - 1] == 0)
            return;

        unsigned fill_start = find_empty(0);
        while (fill_start < n_bins) {
            unsigned fill_end = find_filled(fill_start);
            assert(fill_end > fill_start);
            assert(fill_end < n_bins); // last bin is always filled

            const auto delta = (end_of(fill_end) - begin_of(fill_start)) / (fill_end - fill_start + 1);
            assert(!stroke::isnan(delta));
            const auto offset = begin_of(fill_start);
            for (unsigned i = 0; i < fill_end - fill_start; ++i) {
                bin_borders[fill_start + i] = offset + delta * (i + 1);
                assert(!stroke::isnan(bin_borders[fill_start + i]));
            }

            fill_start = find_empty(fill_end);
        }

        // bubble sort
        for (auto i = 0u; i < n_bins - 1; ++i) {
            auto swapped = false;
            for (auto j = 0u; j < n_bins - i - 1; ++j) {
                if (bin_borders[j] > bin_borders[j + 1]) {
                    stroke::swap(bin_borders[j], bin_borders[j + 1]);
                    swapped = true;
                }
            }
            if (swapped == false)
                break;
        }
    }
};

template <typename config>
class RasterBinSizer_2 {
    static constexpr auto n_bins = config::n_rasterisation_steps;
    static constexpr auto transmission_threshold = config::transmission_threshold;
    whack::Array<float, n_bins> weight_sum = {};
    whack::Array<float, n_bins> centroids = {};
    whack::Array<float, n_bins> SDs = {};
    float transmission = 1.f;
    float max_depth = 0.f;

public:
    STROKE_DEVICES_INLINE float begin_of(unsigned i) const
    {
        if (i == 0)
            return 0;

        return end_of(i - 1);
    }
    STROKE_DEVICES_INLINE float end_of(unsigned i) const
    {
        if (i == n_bins - 1)
            return max_depth;

        assert(centroids[i] <= centroids[i + 1]);

        const auto delta = centroids[i + 1] - centroids[i];
        const auto percentage = SDs[i] / (SDs[i] + SDs[i + 1] + 0.000001f);

        return centroids[i] + percentage * delta;
    }
    STROKE_DEVICES_INLINE float max_distance() const
    {
        return end_of(n_bins - 1);
    }
    STROKE_DEVICES_INLINE bool is_full() const
    {
        return transmission < transmission_threshold;
    }
    STROKE_DEVICES_INLINE void add_gaussian(float opacity, float centre, float sd)
    {
        namespace gaussian = stroke::gaussian;
        const auto bin_for = [](float transmission) {
            const auto t_flipped_and_scaled = 1 - (transmission - transmission_threshold) / (1 - transmission_threshold);
            const auto t_bin = unsigned(stroke::min(t_flipped_and_scaled, 1.0001f) * (n_bins - 1));
            return t_bin;
        };

        assert(!stroke::isnan(centre));
        assert(!stroke::isnan(sd));
        assert(sd > 0);

        const auto alpha = stroke::min(0.999f, opacity * gaussian::integrate_normalised_SD(centre, sd, { 0, centre + sd * config::gaussian_relevance_sigma }));
        assert(!stroke::isnan(alpha));

        transmission *= 1 - alpha;
        const auto bin = bin_for(transmission);
        assert(bin < n_bins);

        weight_sum[bin] += alpha;
        centroids[bin] += centre * alpha;
        SDs[bin] += sd * alpha;
        max_depth = stroke::max(centre + 3 * sd, max_depth);
    }
    STROKE_DEVICES_INLINE void finalise()
    {
        for (auto i = 0; i < n_bins; ++i) {
            if (weight_sum[i] > 0) {
                const auto sum_inv = 1 / weight_sum[i];
                centroids[i] *= sum_inv;
                SDs[i] *= sum_inv;
                weight_sum[i] = 1;
            }
        }
        if (weight_sum[0] == 0) {
            weight_sum[0] = 1;
            centroids[0] = 0;
            SDs[0] = 0.1f;
        }

        const auto find_empty = [this](unsigned start_pos) {
            for (unsigned i = start_pos; i < n_bins; ++i) {
                if (weight_sum[i] == 0)
                    return i;
            }
            return unsigned(-1);
        };
        const auto find_filled = [this](unsigned start_pos) {
            for (unsigned i = start_pos; i < n_bins; ++i) {
                if (weight_sum[i] != 0)
                    return i;
            }
            return unsigned(-1);
        };

        unsigned fill_start = find_empty(0);
        assert(fill_start > 0);
        while (fill_start < n_bins) {
            unsigned fill_end = find_filled(fill_start);
            assert(fill_end > fill_start);
            assert(fill_start - 1 < n_bins);

            if (fill_end >= n_bins) {
                fill_end = n_bins - 1;
                centroids[fill_end] = centroids[fill_start - 1] + config::gaussian_relevance_sigma * SDs[fill_start - 1];
                SDs[fill_end] = SDs[fill_start - 1];
                weight_sum[fill_end] = 1;
            }

            const auto delta = (centroids[fill_end] - centroids[fill_start - 1]) / (fill_end - fill_start + 1);
            assert(!stroke::isnan(delta));
            const auto offset = centroids[fill_start - 1];
            for (unsigned i = 0; i < fill_end - fill_start; ++i) {
                centroids[fill_start + i] = offset + delta * (i + 1);
                SDs[fill_start + i] = 0.1f;
                assert(!stroke::isnan(centroids[fill_start + i]));
                assert(!stroke::isnan(SDs[fill_start + i]));
            }

            fill_start = find_empty(fill_end);
        }

        // bubble sort
        for (auto i = 0u; i < n_bins - 1; ++i) {
            auto swapped = false;
            for (auto j = 0u; j < n_bins - i - 1; ++j) {
                if (centroids[j] > centroids[j + 1]) {
                    stroke::swap(centroids[j], centroids[j + 1]);
                    stroke::swap(SDs[j], SDs[j + 1]);
                    swapped = true;
                }
            }
            if (swapped == false)
                break;
        }
    }
};

template <typename config>
class RasterBinSizer {
    static constexpr auto n_bins = config::n_rasterisation_bins;
    static constexpr auto mass_threshold = -gcem::log(config::transmission_threshold);
    whack::Array<float, n_bins> masses = {};
    whack::Array<float, n_bins> borders = {};
    whack::Array<float, n_bins> SDs = {};
    whack::Array<float, n_bins> transparencies = {};
    float accumulated_mass = 0.f;
    float max_depth = 0.f;

public:
    STROKE_DEVICES_INLINE RasterBinSizer()
    {
        for (float& b : borders) {
            b = 1.f / 0.f;
        }
        for (float& t : transparencies) {
            t = 1.f;
        }
    }

    STROKE_DEVICES_INLINE float begin_of(unsigned i) const
    {
        if (i == 0)
            return 0;

        return end_of(i - 1);
    }
    STROKE_DEVICES_INLINE float end_of(unsigned i) const
    {
        if (i == n_bins - 1)
            return max_depth;
        if (!(borders[i] <= borders[i + 1])) {
            printf("borders[%i] = %f\n", i, borders[i]);
            printf("borders[%i+ 1] = %f\n", i, borders[i + 1]);
        }
        assert(borders[i] <= borders[i + 1]);
        return borders[i];
    }

    STROKE_DEVICES_INLINE float border_mass_begin_of(unsigned i) const
    {
        if (i == 0)
            return 0;

        return border_mass_end_of(i - 1);
    }
    STROKE_DEVICES_INLINE float border_mass_end_of(unsigned i) const
    {
        if (i == n_bins - 1)
            return 0;

        return masses[i];
    }

    STROKE_DEVICES_INLINE float max_distance() const
    {
        return end_of(n_bins - 1);
    }
    STROKE_DEVICES_INLINE bool is_full() const
    {
        return accumulated_mass >= mass_threshold;
    }
    STROKE_DEVICES_INLINE void add_gaussian(float mass, float centre, float sd)
    {
        namespace gaussian = stroke::gaussian;
        using Gaussian1d = stroke::gaussian::ParamsWithWeight<1, float>;

        assert(!stroke::isnan(centre));
        assert(!stroke::isnan(sd));
        assert(!stroke::isnan(mass));
        assert(sd > 0);
        if (mass < 0.0001f)
            return;

        const auto t_relevance_threshold = centre + sd * config::gaussian_relevance_sigma;
        const auto added_mass = mass * gaussian::integrate_normalised_SD(centre, sd, { 0, t_relevance_threshold });
        assert(!stroke::isnan(added_mass));
        max_depth = stroke::max(t_relevance_threshold, max_depth);
        accumulated_mass += added_mass;

        constexpr auto last_bin = config::n_rasterisation_bins - 1;
        masses[last_bin] = added_mass;
        borders[last_bin] = centre;
        SDs[last_bin] = sd;

        // insertion sort
        for (auto i = last_bin; i > 0; --i) {
            if (borders[i - 1] > borders[i]) {
                stroke::swap(borders[i - 1], borders[i]);
                stroke::swap(SDs[i - 1], SDs[i]);
                stroke::swap(masses[i - 1], masses[i]);
                continue;
            }
            auto transparency = (i == 0) ? 1.f : transparencies[i - 1];
            for (; i < last_bin; ++i) {
                transparency *= stroke::exp(-masses[i]);
                transparencies[i] = transparency;
            }
            break;
        }

        const auto transparency_before = [&](int index) {
            if (index <= 0)
                return 1.f;
            return transparencies[index - 1];
        };

        const auto merge = [&](int a, int b) {
            assert(a >= 0);
            assert(a < b);
            assert(b < n_bins);
            const auto scaled_m_a = transparency_before(a) * masses[a];
            const auto scaled_m_b = transparency_before(b) * masses[b];
            const auto percentage_a = (scaled_m_a + scaled_m_b >= 0.0000001f) ? stroke::min(scaled_m_a / (scaled_m_a + scaled_m_b), 0.999999f) : 0.5f;

            // const auto percentage_a = masses[a] / (masses[a] + masses[b]);
            const auto percentage_b = 1 - percentage_a;

            // if (!(scaled_m_a / (scaled_m_a + scaled_m_b) >= 0.f && scaled_m_a / (scaled_m_a + scaled_m_b) <= 1.000001f)) {
            //     printf("percentage_a = %f\n", scaled_m_a / (scaled_m_a + scaled_m_b));
            //     printf("a=%i, b=%i\n", a, b);
            //     printf("transparency_before(a) = %f\n", transparency_before(a));
            //     printf("masses[a] = %f\n", masses[a]);
            //     printf("transparency_before(b) = %f\n", transparency_before(b));
            //     printf("masses[b] = %f\n", masses[b]);
            // }
            assert(percentage_a >= 0.f);
            // assert(scaled_m_a / (scaled_m_a + scaled_m_b) <= 1.000001f);
            assert(percentage_b >= 0.f);
            assert(percentage_b <= 1.0000001f);

            // if tb_centr[a] + tb_SD[a] * 1.5 > tb_centr[b] - tb_SD[b] * 1.5 or True:
            const auto new_centr = percentage_a * borders[a] + percentage_b * borders[b];
            // else:
            // new_centr = tb_centr[a if tb_trans[a] * tb_mass[a] > tb_trans[b] * tb_mass[b] else b]
            const auto new_mass = masses[a] + masses[b];
            const auto var_a = stroke::sq(SDs[a]);
            const auto var_b = stroke::sq(SDs[b]);
            const auto new_var = percentage_a * (var_a + (borders[a] - new_centr) * (borders[a] - new_centr)) + percentage_b * (var_b + (borders[b] - new_centr) * (borders[b] - new_centr));
            return Gaussian1d { new_mass, new_centr, stroke::sqrt(new_var) };
        };

        const auto cost = [&](int index) {
            assert(index >= 0);
            assert(index < n_bins - 1);

            const auto kl_div = [](float c1, float sd1, float c2, float sd2) {
                return (stroke::sq(c1 - c2) + stroke::sq(sd1) - stroke::sq(sd2)) / (2 * stroke::sq(sd2)) + stroke::log(sd2 / sd1);
            };

            const Gaussian1d m = merge(index, index + 1);
            const auto& c1 = borders[index];
            const auto& c2 = borders[index + 1];
            const auto& sd1 = SDs[index];
            const auto& sd2 = SDs[index + 1];
            return transparency_before(index + 1) * (kl_div(c1, sd1, m.centre, m.C) + kl_div(c2, sd2, m.centre, m.C));
        };

        if (masses[last_bin] > 0) {
            // compact by merging

            // search for merge candidate
            int best_idx = -1;
            float best_idx_val = 1.f / 0.f;
            for (int j = 0; j < n_bins - 1; ++j) {
                const auto val = cost(j);
                if (val < best_idx_val) {
                    best_idx_val = val;
                    best_idx = j;
                }
            }
            if (best_idx < 0) {
                // printf("best_idx_val: %f\n", best_idx_val);

                for (int j = 0; j < n_bins - 1; ++j) {
                    printf("borders(%i): %f\n", j, borders[j]);
                    printf("SDs(%i): %f\n", j, SDs[j]);
                    printf("transparencies(%i): %f\n", j, transparencies[j]);

                    printf("cost(%i): %f\n", j, cost(j));
                }
            }
            assert(best_idx >= 0);
            assert(best_idx < n_bins - 1);

            // merge
            const Gaussian1d m = merge(best_idx, best_idx + 1);
            borders[best_idx] = m.centre;
            masses[best_idx] = m.weight;
            SDs[best_idx] = m.C;

            auto transparency = transparency_before(best_idx);
            transparency *= stroke::exp(-masses[best_idx]);
            transparencies[best_idx] = transparency;

            for (unsigned j = best_idx + 1; j < n_bins - 1; ++j) {
                borders[j] = borders[j + 1];
                masses[j] = masses[j + 1];
                SDs[j] = SDs[j + 1];
                transparency *= stroke::exp(-masses[j]);
                transparencies[j] = transparency;
            }
            transparency *= stroke::exp(-masses[n_bins - 1]);
            transparencies[n_bins - 1] = transparency;
        }
    }
    STROKE_DEVICES_INLINE void finalise()
    {
        float last_valid_border = 0;
        for (unsigned i = 0; i < n_bins; ++i) {
            float& b = borders[i];
            if (!stroke::isinf(b)) {
                b = stroke::max(b, last_valid_border + 0.00001f); // numerical inaccuracy if borders are very close.
                last_valid_border = b;
            } else {
                b = last_valid_border + (max_depth - last_valid_border) / (n_bins - i);
                last_valid_border = b;
                masses[i] = 0;
            }
        }
    }
};

} // namespace dgmr::utils
