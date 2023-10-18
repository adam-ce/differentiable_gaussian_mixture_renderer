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

#include <cuda/std/tuple>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <stroke/cuda_compat.h>
#include <stroke/gaussian.h>
#include <stroke/geometry.h>
#include <stroke/matrix.h>
#include <stroke/utility.h>
#include <stroke/welford.h>

namespace dgmr::utils {

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
    STROKE_DEVICES_INLINE void add_gaussian(float opacity, float centre, float variance)
    {
        namespace gaussian = stroke::gaussian;
        const auto g_end_position = centre + stroke::sqrt(variance) * config::gaussian_relevance_sigma;
        if (g_end_position < 0)
            return;
        const auto alpha = stroke::min(0.99f, opacity * gaussian::integrate(centre, variance, { 0, g_end_position }));
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
class RasterBinSizer {
    static constexpr auto n_bins = config::n_rasterisation_steps;
    static constexpr auto transmission_threshold = config::transmission_threshold;
    whack::Array<float, n_bins> weight_sum = {};
    whack::Array<float, n_bins> centroids = {};
    whack::Array<float, n_bins> SDs = {};
    float transmission = 1.f;

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
            return centroids[i] + SDs[i];

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
    STROKE_DEVICES_INLINE void add_gaussian(float opacity, float centre, float variance)
    {
        namespace gaussian = stroke::gaussian;
        const auto bin_for = [](float transmission) {
            const auto t_flipped_and_scaled = 1 - (transmission - transmission_threshold) / (1 - transmission_threshold);
            const auto t_bin = unsigned(stroke::min(t_flipped_and_scaled, 1.0001f) * (n_bins - 1));
            return t_bin;
        };

        assert(!stroke::isnan(centre));
        assert(!stroke::isnan(variance));
        assert(variance > 0);

        const auto SD = stroke::sqrt(variance);
        const auto alpha = stroke::min(0.999f, opacity * gaussian::integrate(centre, variance, { 0, centre + SD * config::gaussian_relevance_sigma }));
        assert(!stroke::isnan(alpha));

        transmission *= 1 - alpha;
        const auto bin = bin_for(transmission);
        assert(bin < n_bins);

        weight_sum[bin] += alpha;
        centroids[bin] += centre * alpha;
        SDs[bin] += SD * alpha;
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

STROKE_DEVICES_INLINE stroke::Cov3<float> compute_cov(const glm::vec3& scale, const glm::quat& rot)
{
    const auto RS = glm::toMat3(rot) * glm::mat3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
    return stroke::Cov3<float>(RS * transpose(RS));
}

template <typename scalar_t>
STROKE_DEVICES_INLINE glm::mat<3, 3, scalar_t> rotation_matrix_from(const glm::vec<3, scalar_t>& direction)
{
    using Vec = glm::vec<3, scalar_t>;
    using Mat = glm::mat<3, 3, scalar_t>;
    assert(stroke::abs(glm::length(direction) - 1) < 0.0001f);
    const auto dot_z_abs = stroke::abs(dot(direction, Vec(0, 0, 1)));
    const auto dot_x_abs = stroke::abs(dot(direction, Vec(1, 0, 0)));

    const auto other_1 = glm::normalize(glm::cross(direction, dot_z_abs < dot_x_abs ? Vec(0, 0, 1) : Vec(1, 0, 0)));
    const auto other_2 = glm::normalize(glm::cross(other_1, direction));

    return Mat(other_1, other_2, direction);
}

template <typename scalar_t>
struct DirectionAndKernelScales {
    const glm::vec<3, scalar_t>& direction;
    const glm::vec<3, scalar_t>& kernel_scales;
};

template <typename scalar_t>
STROKE_DEVICES_INLINE stroke::Cov3<scalar_t> orient_filter_kernel(const DirectionAndKernelScales<scalar_t>& p)
{
    using Vec = glm::vec<3, scalar_t>;
    using Mat = glm::mat<3, 3, scalar_t>;
    const auto RS = rotation_matrix_from(Vec(p.direction)) * Mat(p.kernel_scales.x, 0, 0, 0, p.kernel_scales.y, 0, 0, 0, p.kernel_scales.z);
    return stroke::Cov3<scalar_t>(RS * transpose(RS));
}

template <glm::length_t n_dims, typename scalar_t>
struct FilteredCovAndWeight {
    stroke::Cov<n_dims, scalar_t> cov;
    float weight_factor;
};

// kernel cov comes from a normalised gaussian, i.e., it integrates to 1 and has no explicit weight
template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE FilteredCovAndWeight<n_dims, scalar_t> convolve(const stroke::Cov<n_dims, scalar_t>& cov, const stroke::Cov<n_dims, scalar_t>& kernel_cov)
{
    const auto new_cov = cov + kernel_cov;
    return { stroke::Cov<n_dims, scalar_t>(new_cov), float(stroke::sqrt(stroke::max(scalar_t(0.000025), scalar_t(det(cov) / det(new_cov))))) };
}

STROKE_DEVICES_INLINE stroke::geometry::Aabb1f gaussian_to_point_distance_bounds(
    const glm::vec3& gauss_centr,
    const glm::vec3& gauss_size,
    const glm::quat& gauss_rotation,
    const float gauss_iso_ellipsoid,
    const glm::vec3& query_point)
{

    const auto transformed_query_point = glm::toMat3(gauss_rotation) * (query_point - gauss_centr);
    const auto s = gauss_size * (0.5f * gauss_iso_ellipsoid);
    const auto transformed_bb = stroke::geometry::Aabb3f { -s, s };

    return { stroke::geometry::distance(transformed_bb, transformed_query_point), stroke::geometry::largest_distance_to(transformed_bb, transformed_query_point) };
}
} // namespace dgmr::utils
