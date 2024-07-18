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

namespace dgmr::marching_steps {

template<typename scalar_t>
struct DensityEntry {
    scalar_t start;
    scalar_t end;
    scalar_t delta_t;
    scalar_t g_start;

    STROKE_DEVICES_INLINE DensityEntry() = default;

    STROKE_DEVICES_INLINE DensityEntry(scalar_t start, scalar_t end, scalar_t delta_t)
        : start(start)
        , end(end)
        , delta_t(delta_t)
        , g_start(start)
    {
    }
};

template <unsigned max_size, typename scalar_t>
class DensityArray {
    unsigned m_size = 0;
    whack::Array<DensityEntry<scalar_t>, max_size> m_data;
    scalar_t m_start;
    scalar_t m_end = scalar_t(1) / scalar_t(0);

public:
    STROKE_DEVICES_INLINE DensityArray(scalar_t start)
        : m_start(start)
    {
    }
    STROKE_DEVICES_INLINE DensityArray()
        : m_start(0)
        , m_end(0)
        , m_data({})
    {
    }
    STROKE_DEVICES_INLINE unsigned size() const
    {
        return m_size;
    }
    STROKE_DEVICES_INLINE void set_size(unsigned int size)
    {
        m_size = size;
    }
    STROKE_DEVICES_INLINE void reset_non_differentiable_values(unsigned int size, scalar_t start, scalar_t end)
    {
        m_size = size;
        m_start = start;
        m_end = end;
    }
    STROKE_DEVICES_INLINE scalar_t start() const
    {
        return m_start;
    }
    STROKE_DEVICES_INLINE scalar_t end() const
    {
        return m_end;
    }

    STROKE_DEVICES_INLINE DensityEntry<scalar_t>& operator[](unsigned index)
    {
        assert(index < m_size);
        return m_data[index];
    }

    STROKE_DEVICES_INLINE const DensityEntry<scalar_t>& operator[](unsigned index) const
    {
        assert(index < m_size);
        return m_data[index];
    }

    STROKE_DEVICES_INLINE static whack::Array<DensityEntry<scalar_t>, 3> combine(const DensityEntry<scalar_t>& a, const DensityEntry<scalar_t>& b)
    {
        whack::Array<DensityEntry<scalar_t>, 3> tmp;
        // not overlapping
        if (a.start < b.start) {
            tmp[0].start = a.start;
            tmp[0].end = stroke::min(a.end, b.start);
            tmp[0].delta_t = a.delta_t;
            tmp[0].g_start = a.g_start;
        } else {
            tmp[0].start = b.start;
            tmp[0].end = stroke::min(b.end, a.start);
            tmp[0].delta_t = b.delta_t;
            tmp[0].g_start = b.g_start;
        }
        if (b.end < a.end) {
            assert(a.start < b.end);
            tmp[2].start = b.end; // stroke::max(b.end, a.start);
            tmp[2].end = a.end;
            tmp[2].delta_t = a.delta_t;
            tmp[2].g_start = a.g_start;
        } else {
            assert(b.start < a.end); // otherwise
            tmp[2].start = a.end; // stroke::max(a.end, b.start);
            tmp[2].end = b.end;
            tmp[2].delta_t = b.delta_t;
            tmp[2].g_start = b.g_start;
        }
        // overlapping
        tmp[1].start = tmp[0].end;
        tmp[1].end = tmp[2].start;
        if (a.delta_t < b.delta_t) {
            tmp[1].delta_t = a.delta_t;
            tmp[1].g_start = a.g_start;
        }
        else {
            tmp[1].delta_t = b.delta_t;
            tmp[1].g_start = b.g_start;
        }
        return tmp;
    }

    STROKE_DEVICES_INLINE void put(DensityEntry<scalar_t> entry)
    {
        const auto find = [this](scalar_t v, unsigned find_from = 0u) {
            for (auto i = find_from; i < m_size; ++i) {
                const DensityEntry<scalar_t>& e = m_data[i];
                if (e.end > v)
                    return i;
            }
            return m_size;
        };

        const auto compact = [](whack::Array<DensityEntry<scalar_t>, 3> data) {
            if (data[0].delta_t == data[1].delta_t) {
                data[1].start = data[0].start;
                data[0].end = data[0].start;
            }
            return data;
        };

        if (entry.end <= m_start)
            return;
        if (entry.start >= m_end)
            return;
        entry.start = stroke::max(m_start, entry.start);
        entry.end = stroke::min(m_end, entry.end);

        unsigned p_s = find(entry.start);
        if (p_s >= m_size) {
            if (p_s < max_size) {
                m_data[m_size++] = entry;
            } else
                m_end = entry.start;
            return;
        }
        // assert this is the first entry touched
        assert(p_s == 0 || m_data[p_s - 1].end <= entry.start);
        assert(m_data[p_s].end >= entry.start);

        unsigned p_e = find(entry.end, p_s);
        if (p_e < m_size && m_data[p_e].start < entry.end)
            ++p_e;
        // assert this entry is the first not touched any more
        assert(p_e == m_size || m_data[p_e].start >= entry.end);

        whack::Array<DensityEntry<scalar_t>, max_size> tmp;
        unsigned tmp_read = 0;
        unsigned tmp_write = 0;
        auto i_read = p_s;
        auto i_write = p_s;
        unsigned n_added = 0;
        while (i_read < p_e + n_added && i_write < max_size) {
            if (i_read < m_size)
                tmp[(tmp_write++) % max_size] = m_data[i_read++];
            if (tmp_read >= tmp_write)
                break;
            if (tmp[tmp_read % max_size].end <= entry.start) {
                m_data[i_write++] = tmp[(tmp_read++) % max_size];
                continue;
            }
            if (tmp[tmp_read % max_size].start >= entry.end) {
                break;
            }

            const auto combined = combine(tmp[(tmp_read++) % max_size], entry);
            const auto new_entries = compact(combined);
            entry = new_entries[2];
            if (new_entries[0].end - new_entries[0].start <= 0) {
                if (new_entries[1].delta_t == new_entries[2].delta_t) {
                    entry.start = new_entries[1].start;
                    continue;
                }
                m_data[i_write++] = new_entries[1];
            } else {
                m_data[i_write++] = new_entries[0];
                if (new_entries[1].delta_t == new_entries[2].delta_t) {
                    entry.start = new_entries[1].start;
                    continue;
                }
                ++n_added;
                tmp[(--tmp_read) % max_size] = new_entries[1];
            }
        }
        if (entry.end - entry.start > 0 && i_write < max_size) {
            if (i_read < m_size)
                tmp[(tmp_write++) % max_size] = m_data[i_read++];
            m_data[i_write++] = entry;
        }
        while (tmp_write - tmp_read && i_write < max_size) {
            if (i_read < m_size)
                tmp[(tmp_write++) % max_size] = m_data[i_read++];
            m_data[i_write++] = tmp[(tmp_read++) % max_size];
        }
        m_size = i_write + m_size - i_read;

        if (m_size == max_size)
            m_end = stroke::min(m_end, m_data[m_size - 1].end);
    }
};
template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t next_sample(const DensityEntry<scalar_t>& density, scalar_t t)
{
    if (t < density.g_start)
        return density.g_start;
    // const unsigned n_steps = stroke::ceil((t - density.g_start) / density.delta_t);

    const auto tmp = (t - density.g_start) / density.delta_t;
    unsigned n_steps = stroke::ceil((t - density.g_start) / density.delta_t);

    return density.g_start + n_steps * density.delta_t;
}

template <unsigned n_samples, unsigned n_samples_per_gaussian, unsigned n_density_sections, typename scalar_t>
STROKE_DEVICES_INLINE whack::Array<scalar_t, n_samples> sample(const DensityArray<n_density_sections, scalar_t>& densities)
{
    if (densities.size() == 0)
        return {};

    whack::Array<scalar_t, n_samples> samples;
    unsigned current_density_index = 0;
    scalar_t last_sample = densities.start() - densities[0].delta_t / 2;
    for (auto i = 0u; i < samples.size(); ++i) {
        auto sample = next_sample(densities[current_density_index], last_sample + densities[current_density_index].delta_t / 2);
        while (sample > densities[current_density_index].end) {
            ++current_density_index;
            if (current_density_index >= densities.size())
                break;
            sample = next_sample(densities[current_density_index], densities[current_density_index].start);
        }
        if (current_density_index >= densities.size()) {
            assert(last_sample > densities.start()); // no samples produced
            for (; i < samples.size(); ++i)
                samples[i] = last_sample;
            break;
        }
        assert(sample > densities.start()); // no samples produced
        samples[i] = sample;
        last_sample = sample;
    }

    return samples;
}

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

    template <unsigned n_values>
    STROKE_DEVICES_INLINE void add(const whack::Array<float, n_values>& values)
    {
        auto v_idx = 0u;
        for (auto v : values) {
            if (v >= m_data[0])
                break;
            ++v_idx;
        }

        while (m_size != max_size && v_idx < values.size()) {
            assert(v_idx < values.size());
            add(values[v_idx++]);
        }

        if (v_idx >= values.size())
            return;

        auto d_idx = 0u;

        // for (auto d : m_data) {
        //     if (d >= values[v_idx])
        //         break;
        //     ++d_idx;
        // }
        // for (auto d : m_data) {
        for (auto i = 0u; i < m_size; ++i) {
            const auto d = m_data[i];
            assert(v_idx < values.size());
            if (d >= values[v_idx])
                break;
            ++d_idx;
        }

        whack::Array<float, gcem::min(values.size(), max_size) + 1> tmp;
        auto t_idx_write = 0u;
        auto t_idx_read = 0u;

        // const auto original_size = m_size;
        // auto values_in_tmp = 0;
        // while (m_size != max_size && (v_idx < values.size() || values_in_tmp > 0) && d_idx < max_size) {
        //     tmp[t_idx_write++] = m_data[d_idx];
        //     if (d_idx >= original_size) {
        //         tmp[t_idx_write - 1] = 1.0f / 0.0f;
        //     } else {
        //         ++values_in_tmp;
        //     }
        //     if (v_idx >= values.size() || tmp[t_idx_read] < values[v_idx]) {
        //         m_data[d_idx++] = tmp[t_idx_read++];
        //         --values_in_tmp;
        //     } else {
        //         assert(v_idx < values.size());
        //         m_data[d_idx++] = values[v_idx++];
        //         ++m_size;
        //     }
        //     t_idx_write = t_idx_write % tmp.size();
        //     t_idx_read = t_idx_read % tmp.size();
        // }

        if (m_size != max_size)
            return;

        for (; d_idx < max_size; ++d_idx) {
            tmp[t_idx_write++] = m_data[d_idx];
            if (v_idx >= values.size() || tmp[t_idx_read] < values[v_idx]) {
                m_data[d_idx] = tmp[t_idx_read++];
            } else {
                assert(v_idx < values.size());
                m_data[d_idx] = values[v_idx++];
            }
            t_idx_write = t_idx_write % tmp.size();
            t_idx_read = t_idx_read % tmp.size();
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
        const float curr_size = std::max(0.001f, (arr[i + 1] - curr_start) / n_small_steps);
        for (unsigned j = 0; j < n_small_steps; ++j) {
            assert(!stroke::isnan(curr_start));
            assert(curr_start >= 0.f);
            bins.borders[current_bin++] = curr_start;
            curr_start += curr_size;
            curr_start = stroke::min(arr[i + 1], curr_start);
        }
    }
    const auto v = arr[arr.size() - 1];
    for (; current_bin < bins.size() + 1; ++current_bin)
        bins.borders[current_bin] = v;

    return bins;
}

} // namespace dgmr::marching_steps
