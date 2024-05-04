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

#include <stroke/scalar_functions.h>
#include <stroke/utility.h>
#include <whack/array.h>

namespace dgmr::marching_steps {

template <unsigned max_size>
class Array {
    static_assert(max_size >= 2);
    whack::Array<float, max_size> m_data = {};
    unsigned m_size = 1;

public:
    STROKE_DEVICES_INLINE Array(float smallest)
    {
        m_data[0] = smallest;
    }

    STROKE_DEVICES_INLINE void add(float value)
    {
        if (value < m_data[0])
            return;

        assert(!stroke::isnan(value));

        if (m_size < max_size)
            m_data[m_size++] = value;
        else if (value < m_data[max_size - 1])
            m_data[max_size - 1] = value;

        auto& data = *this;
        for (unsigned i = m_size - 1; i > 1; --i) { // m_data[0] is always the smallest, so we can stop after i==2
            if (data[i - 1] > data[i])
                stroke::swap(data[i - 1], data[i]);
            else
                break;
        }
    }
    STROKE_DEVICES_INLINE unsigned size() const
    {
        assert(m_size <= max_size);
        return m_size;
    }

    STROKE_DEVICES_INLINE float& operator[](unsigned index)
    {
        assert(index < m_size);
        return m_data[index];
    }

    STROKE_DEVICES_INLINE float operator[](unsigned index) const
    {
        assert(index < m_size);
        return m_data[index];
    }

    STROKE_DEVICES_INLINE const whack::Array<float, max_size>& data() const { return m_data; }
};

template <unsigned n>
struct Bins {
    whack::Array<float, n + 1> borders = {};

    STROKE_DEVICES_INLINE float begin_of(unsigned index) const
    {
        assert(index < n);
        return borders[index];
    }

    STROKE_DEVICES_INLINE float end_of(unsigned index) const
    {
        assert(index < n);
        return borders[index + 1];
    }

    STROKE_DEVICES_INLINE constexpr unsigned size() const
    {
        return n;
    }
};

template <unsigned n_small_steps, unsigned n_large_steps>
STROKE_DEVICES_INLINE Bins<n_small_steps*(n_large_steps - 1)> make_bins(const Array<n_large_steps>& arr)
{
    static_assert(n_small_steps > 0);
    static_assert(n_large_steps > 1);

    Bins<n_small_steps*(n_large_steps - 1)> bins;
    static_assert(bins.size() > 0);
    unsigned current_bin = 0;
    for (unsigned i = 0; i < arr.size() - 1; ++i) {
        float curr_start = arr[i];
        assert(!stroke::isnan(curr_start));
        const float curr_size = (arr[i + 1] - curr_start) / n_small_steps;
        for (unsigned j = 0; j < n_small_steps; ++j) {
            assert(!stroke::isnan(curr_start));
            assert(curr_start >= 0.f);
            bins.borders[current_bin++] = curr_start;
            curr_start += curr_size;
        }
    }
    bins.borders[current_bin] = arr[arr.size() - 1];

    return bins;
}

} // namespace dgmr::marching_steps
