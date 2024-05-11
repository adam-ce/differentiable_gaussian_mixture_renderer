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

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <dgmr/marching_steps.h>

using Catch::Approx;

TEST_CASE("dgmr marching step DensityArray sampler")
{
    const auto target_delta_t_at = [](const auto& arr, float t) {
        for (auto i = 0u; i < arr.size(); ++i) {
            const auto& e = arr[i];
            if (e.start <= t && t < e.end) {
                return e.delta_t;
            }
        }
        return 1.0f / 0.0f;
    };

    SECTION("empty")
    {
        dgmr::marching_steps::DensityArray<4> arr(0.1f);
        const auto samples = dgmr::marching_steps::sample<16>(arr);
        for (auto i = 0u; i < samples.size(); ++i) {
            CHECK(samples[i] == 0);
        }
    }

    SECTION("filled")
    {
        dgmr::marching_steps::DensityArray<4> arr(0.1f);
        arr.put({ 0.2f, 1.0f, 0.1f });
        arr.put({ 1.0f, 2.0f, 0.2f });
        arr.put({ 2.0f, 2.5f, 0.5f });
        arr.put({ 3.0f, 4.0f, 1.0f });

        const auto samples = dgmr::marching_steps::sample<16>(arr);
        CHECK(samples.front() == 0.2f);
        for (auto i = 1u; i < samples.size(); ++i) {
            const auto delta_t = samples[i] - samples[i - 1];
            CHECK(delta_t > 0);
            for (auto t = samples[i - 1]; t < samples[i]; t += 0.01f)
                CHECK(delta_t <= target_delta_t_at(arr, t) + 0.001f);
        }
    }
}

TEST_CASE("dgmr marching step DensityArray")
{
    SECTION("combine function enveloping ab/ba")
    {
        using DensityArray = dgmr::marching_steps::DensityArray<4>;
        {
            auto v = DensityArray::combine({ 1, 4, 0.5f }, { 0, 5, 1 });
            CHECK(v[0].start == 0);
            CHECK(v[0].end == 1);
            CHECK(v[0].delta_t == 1);

            CHECK(v[1].start == 1);
            CHECK(v[1].end == 4);
            CHECK(v[1].delta_t == 0.5f);

            CHECK(v[2].start == 4);
            CHECK(v[2].end == 5);
            CHECK(v[2].delta_t == 1);
        }
        {
            auto v = DensityArray::combine({ 1, 4, 1.5f }, { 0, 5, 1 });
            CHECK(v[0].start == 0);
            CHECK(v[0].end == 1);
            CHECK(v[0].delta_t == 1);

            CHECK(v[1].start == 1);
            CHECK(v[1].end == 4);
            CHECK(v[1].delta_t == 1);

            CHECK(v[2].start == 4);
            CHECK(v[2].end == 5);
            CHECK(v[2].delta_t == 1);
        }
        {
            auto v = DensityArray::combine({ 0, 5, 1 }, { 1, 4, 0.5f });
            CHECK(v[0].start == 0);
            CHECK(v[0].end == 1);
            CHECK(v[0].delta_t == 1);

            CHECK(v[1].start == 1);
            CHECK(v[1].end == 4);
            CHECK(v[1].delta_t == 0.5f);

            CHECK(v[2].start == 4);
            CHECK(v[2].end == 5);
            CHECK(v[2].delta_t == 1);
        }
        {
            auto v = DensityArray::combine({ 0, 5, 1 }, { 1, 4, 1.5f });
            CHECK(v[0].start == 0);
            CHECK(v[0].end == 1);
            CHECK(v[0].delta_t == 1);

            CHECK(v[1].start == 1);
            CHECK(v[1].end == 4);
            CHECK(v[1].delta_t == 1);

            CHECK(v[2].start == 4);
            CHECK(v[2].end == 5);
            CHECK(v[2].delta_t == 1);
        }
    }
    SECTION("combine function same start")
    {
        using DensityArray = dgmr::marching_steps::DensityArray<4>;
        {
            auto v = DensityArray::combine({ 0, 2, 0.5f }, { 0, 5, 1 });
            CHECK(v[0].start == 0);
            CHECK(v[0].end == 0);

            CHECK(v[1].start == 0);
            CHECK(v[1].end == 2);
            CHECK(v[1].delta_t == 0.5f);

            CHECK(v[2].start == 2);
            CHECK(v[2].end == 5);
            CHECK(v[2].delta_t == 1);
        }
        {
            auto v = DensityArray::combine({ 0, 5, 1 }, { 0, 2, 0.5f });
            CHECK(v[0].start == 0);
            CHECK(v[0].end == 0);

            CHECK(v[1].start == 0);
            CHECK(v[1].end == 2);
            CHECK(v[1].delta_t == 0.5f);

            CHECK(v[2].start == 2);
            CHECK(v[2].end == 5);
            CHECK(v[2].delta_t == 1);
        }
        {
            auto v = DensityArray::combine({ 0, 2, 1.5f }, { 0, 5, 1 });
            CHECK(v[0].start == 0);
            CHECK(v[0].end == 0);

            CHECK(v[1].start == 0);
            CHECK(v[1].end == 2);
            CHECK(v[1].delta_t == 1);

            CHECK(v[2].start == 2);
            CHECK(v[2].end == 5);
            CHECK(v[2].delta_t == 1);
        }
        {
            auto v = DensityArray::combine({ 0, 5, 1 }, { 0, 2, 1.5f });
            CHECK(v[0].start == 0);
            CHECK(v[0].end == 0);

            CHECK(v[1].start == 0);
            CHECK(v[1].end == 2);
            CHECK(v[1].delta_t == 1);

            CHECK(v[2].start == 2);
            CHECK(v[2].end == 5);
            CHECK(v[2].delta_t == 1);
        }
    }
    SECTION("combine function same end")
    {
        using DensityArray = dgmr::marching_steps::DensityArray<4>;
        {
            auto v = DensityArray::combine({ 0, 5, 1 }, { 2, 5, 0.5 });
            CHECK(v[0].start == 0);
            CHECK(v[0].end == 2);
            CHECK(v[0].delta_t == 1);

            CHECK(v[1].start == 2);
            CHECK(v[1].end == 5);
            CHECK(v[1].delta_t == 0.5f);

            CHECK(v[2].start == 5);
            CHECK(v[2].end == 5);
        }
        {
            auto v = DensityArray::combine({ 0, 5, 1 }, { 2, 5, 1.5 });
            CHECK(v[0].start == 0);
            CHECK(v[0].end == 2);
            CHECK(v[0].delta_t == 1);

            CHECK(v[1].start == 2);
            CHECK(v[1].end == 5);
            CHECK(v[1].delta_t == 1);

            CHECK(v[2].start == 5);
            CHECK(v[2].end == 5);
        }
        {
            auto v = DensityArray::combine({ 2, 5, 0.5 }, { 0, 5, 1 });
            CHECK(v[0].start == 0);
            CHECK(v[0].end == 2);
            CHECK(v[0].delta_t == 1);

            CHECK(v[1].start == 2);
            CHECK(v[1].end == 5);
            CHECK(v[1].delta_t == 0.5f);

            CHECK(v[2].start == 5);
            CHECK(v[2].end == 5);
        }
        {
            auto v = DensityArray::combine({ 2, 5, 1.5 }, { 0, 5, 1 });
            CHECK(v[0].start == 0);
            CHECK(v[0].end == 2);
            CHECK(v[0].delta_t == 1);

            CHECK(v[1].start == 2);
            CHECK(v[1].end == 5);
            CHECK(v[1].delta_t == 1);

            CHECK(v[2].start == 5);
            CHECK(v[2].end == 5);
        }
    }

    SECTION("non-overlapping end")
    {
        dgmr::marching_steps::DensityArray<4> arr(0.5f);
        CHECK(arr.size() == 0);
        arr.put({ 0.0f, 0.4f, 0.8f });
        CHECK(arr.size() == 0);
        arr.put({ 0.0f, 1.0f, 0.8f });
        CHECK(arr.size() == 1);
        CHECK(arr[0].start == 0.5f);
        CHECK(arr[0].end == 1.0f);
        CHECK(arr[0].delta_t == 0.8f);

        arr.put({ 1.0f, 2.0f, 0.5f });
        CHECK(arr.size() == 2);
        CHECK(arr[1].start == 1.0f);
        CHECK(arr[1].end == 2.0f);
        CHECK(arr[1].delta_t == 0.5f);

        arr.put({ 2.0f, 2.5f, 0.9f });
        CHECK(arr.size() == 3);
        CHECK(arr[2].start == 2.0f);
        CHECK(arr[2].end == 2.5f);
        CHECK(arr[2].delta_t == 0.9f);

        arr.put({ 3.0f, 4.0f, 0.5f });
        CHECK(arr.size() == 4);
        CHECK(arr[3].start == 3.0f);
        CHECK(arr[3].end == 4.0f);
        CHECK(arr[3].delta_t == 0.5f);

        arr.put({ 4.0f, 5.0f, 1.5f });
        CHECK(arr.size() == 4);
        CHECK(arr[3].start == 3.0f);
        CHECK(arr[3].end == 4.0f);
        CHECK(arr[3].delta_t == 0.5f);
    }

    SECTION("non-overlapping middle")
    {
        dgmr::marching_steps::DensityArray<4> arr(0.5f);
        CHECK(arr.size() == 0);
        arr.put({ 0.0f, 0.9f, 0.8f });
        arr.put({ 4.0f, 5.0f, 1.5f });

        arr.put({ 1.0f, 2.0f, 0.5f });
        CHECK(arr.size() == 3);
        CHECK(arr[0].start == 0.5f);
        CHECK(arr[0].end == 0.9f);
        CHECK(arr[0].delta_t == 0.8f);
        CHECK(arr[1].start == 1.0f);
        CHECK(arr[1].end == 2.0f);
        CHECK(arr[1].delta_t == 0.5f);
        CHECK(arr[2].start == 4.0f);
        CHECK(arr[2].end == 5.0f);
        CHECK(arr[2].delta_t == 1.5f);

        arr.put({ 3.0f, 4.0f, 0.5f });
        CHECK(arr.size() == 4);
        CHECK(arr[0].start == 0.5f);
        CHECK(arr[0].end == 0.9f);
        CHECK(arr[0].delta_t == 0.8f);
        CHECK(arr[1].start == 1.0f);
        CHECK(arr[1].end == 2.0f);
        CHECK(arr[1].delta_t == 0.5f);
        CHECK(arr[2].start == 3.0f);
        CHECK(arr[2].end == 4.0f);
        CHECK(arr[2].delta_t == 0.5f);
        CHECK(arr[3].start == 4.0f);
        CHECK(arr[3].end == 5.0f);
        CHECK(arr[3].delta_t == 1.5f);

        arr.put({ 2.0f, 2.5f, 0.9f });
        CHECK(arr.size() == 4);
        CHECK(arr[0].start == 0.5f);
        CHECK(arr[0].end == 0.9f);
        CHECK(arr[0].delta_t == 0.8f);
        CHECK(arr[1].start == 1.0f);
        CHECK(arr[1].end == 2.0f);
        CHECK(arr[1].delta_t == 0.5f);
        CHECK(arr[2].start == 2.0f);
        CHECK(arr[2].end == 2.5f);
        CHECK(arr[2].delta_t == 0.9f);
        CHECK(arr[3].start == 3.0f);
        CHECK(arr[3].end == 4.0f);
        CHECK(arr[3].delta_t == 0.5f);
    }

    SECTION("non-overlapping start")
    {
        dgmr::marching_steps::DensityArray<4> arr(0.5f);
        CHECK(arr.size() == 0);
        arr.put({ 4.0f, 5.0f, 1.5f });
        CHECK(arr.size() == 1);
        CHECK(arr[0].start == 4.0f);
        CHECK(arr[0].end == 5.0f);
        CHECK(arr[0].delta_t == 1.5f);

        arr.put({ 3.0f, 4.0f, 0.5f });
        CHECK(arr.size() == 2);
        CHECK(arr[0].start == 3.0f);
        CHECK(arr[0].end == 4.0f);
        CHECK(arr[0].delta_t == 0.5f);
        CHECK(arr[1].start == 4.0f);
        CHECK(arr[1].end == 5.0f);
        CHECK(arr[1].delta_t == 1.5f);

        arr.put({ 2.0f, 2.5f, 0.9f });
        CHECK(arr.size() == 3);
        CHECK(arr[0].start == 2.0f);
        CHECK(arr[0].end == 2.5f);
        CHECK(arr[0].delta_t == 0.9f);
        CHECK(arr[1].start == 3.0f);
        CHECK(arr[1].end == 4.0f);
        CHECK(arr[1].delta_t == 0.5f);
        CHECK(arr[2].start == 4.0f);
        CHECK(arr[2].end == 5.0f);
        CHECK(arr[2].delta_t == 1.5f);

        arr.put({ 1.0f, 2.0f, 0.5f });
        CHECK(arr.size() == 4);
        CHECK(arr[0].start == 1.0f);
        CHECK(arr[0].end == 2.0f);
        CHECK(arr[0].delta_t == 0.5f);
        CHECK(arr[1].start == 2.0f);
        CHECK(arr[1].end == 2.5f);
        CHECK(arr[1].delta_t == 0.9f);
        CHECK(arr[2].start == 3.0f);
        CHECK(arr[2].end == 4.0f);
        CHECK(arr[2].delta_t == 0.5f);
        CHECK(arr[3].start == 4.0f);
        CHECK(arr[3].end == 5.0f);
        CHECK(arr[3].delta_t == 1.5f);

        arr.put({ 0.0f, 0.4f, 0.8f });
        arr.put({ 0.0f, 1.0f, 0.8f });
        CHECK(arr.size() == 4);
        CHECK(arr[0].start == 0.5f);
        CHECK(arr[0].end == 1.0f);
        CHECK(arr[0].delta_t == 0.8f);
        CHECK(arr[1].start == 1.0f);
        CHECK(arr[1].end == 2.0f);
        CHECK(arr[1].delta_t == 0.5f);
        CHECK(arr[2].start == 2.0f);
        CHECK(arr[2].end == 2.5f);
        CHECK(arr[2].delta_t == 0.9f);
        CHECK(arr[3].start == 3.0f);
        CHECK(arr[3].end == 4.0f);
        CHECK(arr[3].delta_t == 0.5f);
    }

    SECTION("many overlapping not full")
    {
        dgmr::marching_steps::DensityArray<16> arr(0.0f);
        arr.put({ 0.2f, 0.5f, 0.5f });
        arr.put({ 1.0f, 1.5f, 0.5f });
        arr.put({ 2.0f, 2.5f, 0.5f });
        arr.put({ 3.0f, 3.5f, 0.5f });
        arr.put({ 4.0f, 4.5f, 0.5f });
        arr.put({ 5.0f, 5.5f, 0.5f });
        arr.put({ 6.0f, 6.5f, 0.5f });
        CHECK(arr.size() == 7);

        arr.put({ 0.0f, 7.0f, 1.0f });
        CHECK(arr.size() == 15);

        // clang-format off
        CHECK(arr[0].start == 0.0f);
        CHECK(arr[0].end == 0.2f);
        CHECK(arr[0].delta_t == 1.0f);
        
        CHECK(arr[1].start == 0.2f);
        CHECK(arr[1].end == 0.5f);
        CHECK(arr[1].delta_t == 0.5f);
        
        CHECK(arr[2].start == 0.5f);
        CHECK(arr[2].end == 1.0f);
        CHECK(arr[2].delta_t == 1.0f);
        
        
        CHECK(arr[3].start == 1.0f);
        CHECK(arr[3].end == 1.5f);
        CHECK(arr[3].delta_t == 0.5f);
        
        CHECK(arr[4].start == 1.5f);
        CHECK(arr[4].end == 2.0f);
        CHECK(arr[4].delta_t == 1.0f);
        
        
        CHECK(arr[5].start == 2.0f);
        CHECK(arr[5].end == 2.5f);
        CHECK(arr[5].delta_t == 0.5f);
        
        CHECK(arr[6].start == 2.5f);
        CHECK(arr[6].end == 3.0f);
        CHECK(arr[6].delta_t == 1.0f);
        
        
        CHECK(arr[7].start == 3.0f);
        CHECK(arr[7].end == 3.5f);
        CHECK(arr[7].delta_t == 0.5f);
        
        CHECK(arr[8].start == 3.5f);
        CHECK(arr[8].end == 4.0f);
        CHECK(arr[8].delta_t == 1.0f);
        
        
        CHECK(arr[9].start == 4.0f);
        CHECK(arr[9].end == 4.5f);
        CHECK(arr[9].delta_t == 0.5f);
        
        CHECK(arr[10].start == 4.5f);
        CHECK(arr[10].end == 5.0f);
        CHECK(arr[10].delta_t == 1.0f);
        
        
        CHECK(arr[11].start == 5.0f);
        CHECK(arr[11].end == 5.5f);
        CHECK(arr[11].delta_t == 0.5f);
        
        CHECK(arr[12].start == 5.5f);
        CHECK(arr[12].end == 6.0f);
        CHECK(arr[12].delta_t == 1.0f);
        
        
        CHECK(arr[13].start == 6.0f);
        CHECK(arr[13].end == 6.5f);
        CHECK(arr[13].delta_t == 0.5f);
        
        CHECK(arr[14].start == 6.5f);
        CHECK(arr[14].end == 7.0f);
        CHECK(arr[14].delta_t == 1.0f);
        // clang-format on
    }

    SECTION("many overlapping over full")
    {
        dgmr::marching_steps::DensityArray<16> arr(0.0f);
        arr.put({ 0.2f, 0.5f, 0.5f });
        arr.put({ 1.0f, 1.5f, 0.5f });
        arr.put({ 2.0f, 2.5f, 0.5f });
        arr.put({ 3.0f, 3.5f, 0.5f });
        arr.put({ 4.0f, 4.5f, 0.5f });
        arr.put({ 5.0f, 5.5f, 0.5f });
        arr.put({ 6.0f, 6.5f, 0.5f });
        arr.put({ 7.0f, 7.5f, 0.5f });
        arr.put({ 8.0f, 8.5f, 0.5f });
        arr.put({ 9.0f, 9.5f, 0.5f });
        CHECK(arr.size() == 10);

        arr.put({ 0.0f, 10.0f, 1.0f });
        CHECK(arr.size() == 16);

        // clang-format off
        CHECK(arr[0].start == 0.0f);
        CHECK(arr[0].end == 0.2f);
        CHECK(arr[0].delta_t == 1.0f);
        
        CHECK(arr[1].start == 0.2f);
        CHECK(arr[1].end == 0.5f);
        CHECK(arr[1].delta_t == 0.5f);
        
        CHECK(arr[2].start == 0.5f);
        CHECK(arr[2].end == 1.0f);
        CHECK(arr[2].delta_t == 1.0f);
        
        
        CHECK(arr[3].start == 1.0f);
        CHECK(arr[3].end == 1.5f);
        CHECK(arr[3].delta_t == 0.5f);
        
        CHECK(arr[4].start == 1.5f);
        CHECK(arr[4].end == 2.0f);
        CHECK(arr[4].delta_t == 1.0f);
        
        
        CHECK(arr[5].start == 2.0f);
        CHECK(arr[5].end == 2.5f);
        CHECK(arr[5].delta_t == 0.5f);
        
        CHECK(arr[6].start == 2.5f);
        CHECK(arr[6].end == 3.0f);
        CHECK(arr[6].delta_t == 1.0f);
        
        
        CHECK(arr[7].start == 3.0f);
        CHECK(arr[7].end == 3.5f);
        CHECK(arr[7].delta_t == 0.5f);
        
        CHECK(arr[8].start == 3.5f);
        CHECK(arr[8].end == 4.0f);
        CHECK(arr[8].delta_t == 1.0f);
        
        
        CHECK(arr[9].start == 4.0f);
        CHECK(arr[9].end == 4.5f);
        CHECK(arr[9].delta_t == 0.5f);
        
        CHECK(arr[10].start == 4.5f);
        CHECK(arr[10].end == 5.0f);
        CHECK(arr[10].delta_t == 1.0f);
        
        
        CHECK(arr[11].start == 5.0f);
        CHECK(arr[11].end == 5.5f);
        CHECK(arr[11].delta_t == 0.5f);
        
        CHECK(arr[12].start == 5.5f);
        CHECK(arr[12].end == 6.0f);
        CHECK(arr[12].delta_t == 1.0f);
        
        
        CHECK(arr[13].start == 6.0f);
        CHECK(arr[13].end == 6.5f);
        CHECK(arr[13].delta_t == 0.5f);
        
        CHECK(arr[14].start == 6.5f);
        CHECK(arr[14].end == 7.0f);
        CHECK(arr[14].delta_t == 1.0f);
        
        
        CHECK(arr[15].start == 7.0f);
        CHECK(arr[15].end == 7.5f);
        CHECK(arr[15].delta_t == 0.5f);
        // clang-format on
    }

    SECTION("overlapping end not full")
    {
        dgmr::marching_steps::DensityArray<4> arr(0.5f);
        arr.put({ 0.0f, 1.5f, 0.5f });
        CHECK(arr.size() == 1);
        CHECK(arr[0].start == 0.5f);
        CHECK(arr[0].end == 1.5f);
        CHECK(arr[0].delta_t == 0.5f);

        arr.put({ 1.0f, 2.0f, 0.1f });
        CHECK(arr.size() == 2);
        CHECK(arr[0].start == 0.5f);
        CHECK(arr[0].end == 1.0f);
        CHECK(arr[0].delta_t == 0.5f);

        CHECK(arr[1].start == 1.0f);
        CHECK(arr[1].end == 2.0f);
        CHECK(arr[1].delta_t == 0.1f);

        arr.put({ 1.2f, 1.8f, 0.01f });
        CHECK(arr.size() == 4);
        CHECK(arr[0].start == 0.5f);
        CHECK(arr[0].end == 1.0f);
        CHECK(arr[0].delta_t == 0.5f);

        CHECK(arr[1].start == 1.0f);
        CHECK(arr[1].end == 1.2f);
        CHECK(arr[1].delta_t == 0.1f);

        CHECK(arr[2].start == 1.2f);
        CHECK(arr[2].end == 1.8f);
        CHECK(arr[2].delta_t == 0.01f);

        CHECK(arr[3].start == 1.8f);
        CHECK(arr[3].end == 2.0f);
        CHECK(arr[3].delta_t == 0.1f);
    }

    SECTION("splitting in the middle")
    {
        dgmr::marching_steps::DensityArray<4> arr(0.5f);
        arr.put({ 1.0f, 1.5f, 0.5f });
        arr.put({ 2.0f, 2.5f, 0.5f });
        arr.put({ 3.0f, 3.5f, 0.5f });
        arr.put({ 4.0f, 4.5f, 0.5f });

        arr.put({ 2.1f, 2.2f, 0.1f });
        CHECK(arr.size() == 4);

        CHECK(arr[0].start == 1.0f);
        CHECK(arr[0].end == 1.5f);
        CHECK(arr[0].delta_t == 0.5f);

        CHECK(arr[1].start == 2.0f);
        CHECK(arr[1].end == 2.1f);
        CHECK(arr[1].delta_t == 0.5f);

        CHECK(arr[2].start == 2.1f);
        CHECK(arr[2].end == 2.2f);
        CHECK(arr[2].delta_t == 0.1f);

        CHECK(arr[3].start == 2.2f);
        CHECK(arr[3].end == 2.5f);
        CHECK(arr[3].delta_t == 0.5f);
    }

    SECTION("splitting in the middle 2")
    {
        dgmr::marching_steps::DensityArray<16> arr(0.5f);
        arr.put({ 1.0f, 1.5f, 0.5f });
        arr.put({ 2.0f, 2.5f, 0.5f });
        arr.put({ 3.0f, 3.5f, 0.5f });
        arr.put({ 4.0f, 4.5f, 0.5f });

        arr.put({ 2.1f, 2.2f, 0.1f });
        CHECK(arr.size() == 6);

        CHECK(arr[0].start == 1.0f);
        CHECK(arr[0].end == 1.5f);
        CHECK(arr[0].delta_t == 0.5f);

        CHECK(arr[1].start == 2.0f);
        CHECK(arr[1].end == 2.1f);
        CHECK(arr[1].delta_t == 0.5f);

        CHECK(arr[2].start == 2.1f);
        CHECK(arr[2].end == 2.2f);
        CHECK(arr[2].delta_t == 0.1f);

        CHECK(arr[3].start == 2.2f);
        CHECK(arr[3].end == 2.5f);
        CHECK(arr[3].delta_t == 0.5f);

        CHECK(arr[4].start == 3.0f);
        CHECK(arr[4].end == 3.5f);
        CHECK(arr[4].delta_t == 0.5f);

        CHECK(arr[5].start == 4.0f);
        CHECK(arr[5].end == 4.5f);
        CHECK(arr[5].delta_t == 0.5f);
    }
    SECTION("splitting in the middle 3")
    {
        dgmr::marching_steps::DensityArray<4> arr(0.5f);
        arr.put({ 1.0f, 1.5f, 0.5f });
        arr.put({ 2.0f, 2.5f, 0.5f });

        arr.put({ 2.1f, 2.2f, 0.1f });
        CHECK(arr.size() == 4);

        CHECK(arr[0].start == 1.0f);
        CHECK(arr[0].end == 1.5f);
        CHECK(arr[0].delta_t == 0.5f);

        CHECK(arr[1].start == 2.0f);
        CHECK(arr[1].end == 2.1f);
        CHECK(arr[1].delta_t == 0.5f);

        CHECK(arr[2].start == 2.1f);
        CHECK(arr[2].end == 2.2f);
        CHECK(arr[2].delta_t == 0.1f);

        CHECK(arr[3].start == 2.2f);
        CHECK(arr[3].end == 2.5f);
        CHECK(arr[3].delta_t == 0.5f);
    }

    SECTION("new element merges several old ones")
    {
        dgmr::marching_steps::DensityArray<16> arr(0.0f);
        arr.put({ 0.2f, 0.5f, 0.5f });
        arr.put({ 1.0f, 1.5f, 0.5f });
        arr.put({ 2.0f, 2.5f, 0.5f });
        arr.put({ 3.0f, 3.5f, 0.5f });
        arr.put({ 4.0f, 4.5f, 0.5f });
        arr.put({ 5.0f, 5.5f, 0.5f });
        arr.put({ 6.0f, 6.5f, 0.5f });
        CHECK(arr.size() == 7);

        arr.put({ 0.0f, 7.0f, 0.1f });
        CHECK(arr.size() == 1);
        CHECK(arr[0].start == 0.0f);
        CHECK(arr[0].end == 7.0f);
        CHECK(arr[0].delta_t == 0.1f);
    }

    SECTION("new element merges several old ones in the middle")
    {
        dgmr::marching_steps::DensityArray<16> arr(0.0f);
        arr.put({ 0.2f, 0.5f, 0.5f });
        arr.put({ 1.0f, 1.5f, 0.5f });
        arr.put({ 2.0f, 2.5f, 0.5f });
        arr.put({ 3.0f, 3.5f, 0.5f });
        arr.put({ 4.0f, 4.5f, 0.5f });
        arr.put({ 5.0f, 5.5f, 0.5f });
        arr.put({ 6.0f, 6.5f, 0.5f });
        CHECK(arr.size() == 7);

        arr.put({ 0.3f, 2.3f, 0.1f });
        CHECK(arr.size() == 7);

        CHECK(arr[0].start == 0.2f);
        CHECK(arr[0].end == 0.3f);
        CHECK(arr[0].delta_t == 0.5f);

        CHECK(arr[1].start == 0.3f);
        CHECK(arr[1].end == 2.3f);
        CHECK(arr[1].delta_t == 0.1f);

        CHECK(arr[2].start == 2.3f);
        CHECK(arr[2].end == 2.5f);
        CHECK(arr[2].delta_t == 0.5f);

        CHECK(arr[3].start == 3.0f);
        CHECK(arr[3].end == 3.5f);
        CHECK(arr[3].delta_t == 0.5f);

        CHECK(arr[4].start == 4.0f);
        CHECK(arr[4].end == 4.5f);
        CHECK(arr[4].delta_t == 0.5f);

        CHECK(arr[5].start == 5.0f);
        CHECK(arr[5].end == 5.5f);
        CHECK(arr[5].delta_t == 0.5f);

        CHECK(arr[6].start == 6.0f);
        CHECK(arr[6].end == 6.5f);
        CHECK(arr[6].delta_t == 0.5f);
    }

    SECTION("new element merges several old ones in the middle 2")
    {
        dgmr::marching_steps::DensityArray<16> arr(0.0f);
        arr.put({ 0.2f, 0.5f, 0.5f });
        arr.put({ 1.0f, 1.5f, 0.5f });
        arr.put({ 2.0f, 2.5f, 0.5f });
        arr.put({ 3.0f, 3.5f, 0.5f });
        arr.put({ 4.0f, 4.5f, 0.5f });
        arr.put({ 5.0f, 5.5f, 0.5f });
        arr.put({ 6.0f, 6.5f, 0.5f });
        CHECK(arr.size() == 7);

        arr.put({ 0.1f, 2.7f, 0.1f });
        CHECK(arr.size() == 5);

        CHECK(arr[0].start == 0.1f);
        CHECK(arr[0].end == 2.7f);
        CHECK(arr[0].delta_t == 0.1f);

        CHECK(arr[1].start == 3.0f);
        CHECK(arr[1].end == 3.5f);
        CHECK(arr[1].delta_t == 0.5f);

        CHECK(arr[2].start == 4.0f);
        CHECK(arr[2].end == 4.5f);
        CHECK(arr[2].delta_t == 0.5f);

        CHECK(arr[3].start == 5.0f);
        CHECK(arr[3].end == 5.5f);
        CHECK(arr[3].delta_t == 0.5f);

        CHECK(arr[4].start == 6.0f);
        CHECK(arr[4].end == 6.5f);
        CHECK(arr[4].delta_t == 0.5f);
    }

    SECTION("new element merges several old ones in the middle 3")
    {
        dgmr::marching_steps::DensityArray<16> arr(0.0f);
        arr.put({ 0.2f, 0.5f, 0.5f });
        arr.put({ 1.0f, 1.5f, 0.5f });
        arr.put({ 2.0f, 2.5f, 0.5f });
        arr.put({ 3.0f, 3.5f, 0.5f });
        arr.put({ 4.0f, 4.5f, 0.5f });
        arr.put({ 5.0f, 5.5f, 0.5f });
        arr.put({ 6.0f, 6.5f, 0.5f });
        CHECK(arr.size() == 7);

        arr.put({ 0.1f, 5.7f, 0.1f });
        CHECK(arr.size() == 2);

        CHECK(arr[0].start == 0.1f);
        CHECK(arr[0].end == 5.7f);
        CHECK(arr[0].delta_t == 0.1f);

        CHECK(arr[1].start == 6.0f);
        CHECK(arr[1].end == 6.5f);
        CHECK(arr[1].delta_t == 0.5f);
    }

    SECTION("new element merges several old ones in the middle 4")
    {
        dgmr::marching_steps::DensityArray<16> arr(0.0f);
        arr.put({ 0.2f, 0.5f, 0.5f });
        arr.put({ 1.0f, 1.5f, 0.5f });
        arr.put({ 2.0f, 2.5f, 0.5f });
        arr.put({ 3.0f, 3.5f, 0.5f });
        arr.put({ 4.0f, 4.5f, 0.01f });
        arr.put({ 5.0f, 5.5f, 0.5f });
        arr.put({ 6.0f, 6.5f, 0.5f });
        CHECK(arr.size() == 7);

        arr.put({ 0.0f, 7.0f, 0.1f });
        CHECK(arr.size() == 3);
        CHECK(arr[0].start == 0.0f);
        CHECK(arr[0].end == 4.0f);
        CHECK(arr[0].delta_t == 0.1f);

        CHECK(arr[1].start == 4.0f);
        CHECK(arr[1].end == 4.5f);
        CHECK(arr[1].delta_t == 0.01f);

        CHECK(arr[2].start == 4.5f);
        CHECK(arr[2].end == 7.0f);
        CHECK(arr[2].delta_t == 0.1f);
    }

    SECTION("new element fits in without changing the last ones 1")
    {
        dgmr::marching_steps::DensityArray<8> arr(0.0f);
        arr.put({ 0.2f, 0.5f, 0.5f });
        arr.put({ 1.0f, 1.5f, 0.5f });
        arr.put({ 2.0f, 2.5f, 0.5f });
        arr.put({ 3.0f, 3.5f, 0.5f });
        CHECK(arr.size() == 4);

        arr.put({ 0.8f, 1.7f, 0.1f });
        CHECK(arr.size() == 4);
        CHECK(arr[0].start == 0.2f);
        CHECK(arr[0].end == 0.5f);
        CHECK(arr[0].delta_t == 0.5f);

        CHECK(arr[1].start == 0.8f);
        CHECK(arr[1].end == 1.7f);
        CHECK(arr[1].delta_t == 0.1f);

        CHECK(arr[2].start == 2.0f);
        CHECK(arr[2].end == 2.5f);
        CHECK(arr[2].delta_t == 0.5f);

        CHECK(arr[3].start == 3.0f);
        CHECK(arr[3].end == 3.5f);
        CHECK(arr[3].delta_t == 0.5f);
    }

    SECTION("new new element fits in without changing the last ones 2")
    {
        dgmr::marching_steps::DensityArray<4> arr(0.0f);
        arr.put({ 0.2f, 0.5f, 0.5f });
        arr.put({ 1.0f, 1.5f, 0.5f });
        arr.put({ 2.0f, 2.5f, 0.5f });
        arr.put({ 3.0f, 3.5f, 0.5f });
        CHECK(arr.size() == 4);

        arr.put({ 0.8f, 1.7f, 0.1f });
        CHECK(arr.size() == 4);
        CHECK(arr[0].start == 0.2f);
        CHECK(arr[0].end == 0.5f);
        CHECK(arr[0].delta_t == 0.5f);

        CHECK(arr[1].start == 0.8f);
        CHECK(arr[1].end == 1.7f);
        CHECK(arr[1].delta_t == 0.1f);

        CHECK(arr[2].start == 2.0f);
        CHECK(arr[2].end == 2.5f);
        CHECK(arr[2].delta_t == 0.5f);

        CHECK(arr[3].start == 3.0f);
        CHECK(arr[3].end == 3.5f);
        CHECK(arr[3].delta_t == 0.5f);
    }

    SECTION("new element envelopse old with higher delta (used to crash)")
    {
        dgmr::marching_steps::DensityArray<8> arr(0.0f);
        arr.put({ 1, 2, 1 });
        arr.put({ 3, 4, 1 });
        arr.put({ 5, 6, 1 });

        arr.put({ 0.5, 4.5, 2 });

        CHECK(arr.size() == 6);

        CHECK(arr[0].start == 0.5f);
        CHECK(arr[0].end == 1);
        CHECK(arr[0].delta_t == 2);

        CHECK(arr[1].start == 1);
        CHECK(arr[1].end == 2);
        CHECK(arr[1].delta_t == 1);

        CHECK(arr[2].start == 2);
        CHECK(arr[2].end == 3);
        CHECK(arr[2].delta_t == 2);

        CHECK(arr[3].start == 3);
        CHECK(arr[3].end == 4);
        CHECK(arr[3].delta_t == 1);

        CHECK(arr[4].start == 4);
        CHECK(arr[4].end == 4.5f);
        CHECK(arr[4].delta_t == 2);

        CHECK(arr[5].start == 5);
        CHECK(arr[5].end == 6);
        CHECK(arr[5].delta_t == 1);
    }

    SECTION("randomised")
    {
        std::srand(0);
        const auto r = []() {
            return float(std::rand()) / float(RAND_MAX);
        };
        for (auto i = 0u; i < 1000; ++i) {
            const auto smallest = r();
            dgmr::marching_steps::DensityArray<8> arr(smallest);
            std::vector<dgmr::marching_steps::DensityEntry> entries;

            const auto get_real_delta_t_at = [&](float t) {
                auto delta_t = 1.f / 0.f;
                for (const auto& e : entries) {
                    if (e.start <= t && t < e.end)
                        delta_t = stroke::min(delta_t, e.delta_t);
                }
                return delta_t;
            };

            const auto get_computed_delta_t_at = [&](float t) {
                for (auto i = 0u; i < arr.size(); ++i) {
                    const auto& e = arr[i];
                    if (e.start <= t && t < e.end) {
                        return e.delta_t;
                    }
                }
                return 1.0f / 0.0f;
            };

            for (auto j = 0u; j < 100; ++j) {
                if (i == 73256 && j == 6)
                    printf(".\n");
                const auto start = r() * 10.0f;
                const auto end = start + r();
                entries.push_back({ start, end, r() });
                arr.put(entries.back());
                if (arr.size()) {
                    CHECK(arr[0].start >= smallest);
                    CHECK(arr[0].start < arr[0].end);
                }

                // check validity
                for (auto k = 1u; k < arr.size(); ++k) {
                    const auto& last = arr[k - 1];
                    const auto& curr = arr[k];
                    CHECK((last.start != curr.start || last.end != curr.end || last.delta_t != curr.delta_t));
                    CHECK(curr.start < curr.end);
                    CHECK(last.end <= curr.start);
                }

                // check correctness
                if (arr.size() == 0)
                    continue;
                for (auto l = 0u; l < 50; ++l) {
                    const auto t = r() * (arr[arr.size() - 1].end - smallest - 0.00001f) + smallest;
                    const auto r = get_real_delta_t_at(t);
                    const auto c = get_computed_delta_t_at(t);
                    // if (c != r)
                    CHECK(get_real_delta_t_at(t) == get_computed_delta_t_at(t));
                }
            }
        }
    }
}

TEST_CASE("dgmr marching step Array")
{
    SECTION("in order")
    {
        dgmr::marching_steps::Array<4> arr(0.5);
        arr.add(1.5);
        CHECK(arr.size() == 2);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 1.5f);

        arr.add(2.5);
        CHECK(arr.size() == 3);
        CHECK(arr[2] == 2.5f);

        arr.add(3.5);
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 1.5f);
        CHECK(arr[2] == 2.5f);
        CHECK(arr[3] == 3.5f);

        arr.add(4.5);
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 1.5f);
        CHECK(arr[2] == 2.5f);
        CHECK(arr[3] == 3.5f);

        arr.add(5.5);
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 1.5f);
        CHECK(arr[2] == 2.5f);
        CHECK(arr[3] == 3.5f);

        arr.add(6.5);
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 1.5f);
        CHECK(arr[2] == 2.5f);
        CHECK(arr[3] == 3.5f);
    }

    SECTION("out of order")
    {
        dgmr::marching_steps::Array<4> arr(0.5);
        arr.add(2.5f);
        CHECK(arr.size() == 2);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 2.5f);

        arr.add(1.5f);
        CHECK(arr.size() == 3);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 1.5f);
        CHECK(arr[2] == 2.5f);

        arr.add(4.5f);
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 1.5f);
        CHECK(arr[2] == 2.5f);
        CHECK(arr[3] == 4.5f);

        arr.add(3.5f);
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 1.5f);
        CHECK(arr[2] == 2.5f);
        CHECK(arr[3] == 3.5f);

        arr.add(0.6f);
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 1.5f);
        CHECK(arr[3] == 2.5f);

        arr.add(0.7f);
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 0.7f);
        CHECK(arr[3] == 1.5f);

        arr.add(0.8f);
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 0.7f);
        CHECK(arr[3] == 0.8f);
    }

    SECTION("ignore stuff before start")
    {
        dgmr::marching_steps::Array<4> arr(0.5);
        arr.add(1.5f);
        arr.add(2.5f);
        arr.add(3.5f);
        arr.add(0.0f);
        arr.add(0.1f);
        arr.add(0.2f);
        arr.add(0.3f);
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 1.5f);
        CHECK(arr[2] == 2.5f);
        CHECK(arr[3] == 3.5f);
    }

    SECTION("bulk add to empty")
    {
        dgmr::marching_steps::Array<4> arr(0.5);
        arr.add(whack::Array<float, 6> { 0, 1, 2, 3, 4, 5 });
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 1.0f);
        CHECK(arr[2] == 2.0f);
        CHECK(arr[3] == 3.0f);
    }

    SECTION("bulk add to empty 2")
    {
        dgmr::marching_steps::Array<8> arr(0.5);
        arr.add(whack::Array<float, 4> { 0.f, .1f, .2f, .3f });
        CHECK(arr.size() == 1);

        arr.add(whack::Array<float, 4> { 0, 1, 2, 3 });
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 1.0f);
        CHECK(arr[2] == 2.0f);
        CHECK(arr[3] == 3.0f);

        arr.add(whack::Array<float, 4> { 0.0f, 0.6f, 3.5f, 4.5f });
        CHECK(arr.size() == 7);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 1.0f);
        CHECK(arr[3] == 2.0f);
        CHECK(arr[4] == 3.0f);
        CHECK(arr[5] == 3.5f);
        CHECK(arr[6] == 4.5f);

        arr.add(whack::Array<float, 4> { 3.4f, 3.6f, 3.7f, 3.8f });
        CHECK(arr.size() == 8);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 1.0f);
        CHECK(arr[3] == 2.0f);
        CHECK(arr[4] == 3.0f);
        CHECK(arr[5] == 3.4f);
        CHECK(arr[6] == 3.5f);
        CHECK(arr[7] == 3.6f);
    }

    SECTION("bulk add to empty 2")
    {
        dgmr::marching_steps::Array<8> arr(0.5);

        arr.add(whack::Array<float, 2> { 1, 2 });
        CHECK(arr.size() == 3);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 1.0f);
        CHECK(arr[2] == 2.0f);

        arr.add(whack::Array<float, 2> { 0.6f, 3.0f });
        CHECK(arr.size() == 5);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 1.0f);
        CHECK(arr[3] == 2.0f);
        CHECK(arr[4] == 3.0f);

        arr.add(whack::Array<float, 2> { 3.4f, 3.6f });
        CHECK(arr.size() == 7);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 1.0f);
        CHECK(arr[3] == 2.0f);
        CHECK(arr[4] == 3.0f);
        CHECK(arr[5] == 3.4f);
        CHECK(arr[6] == 3.6f);

        arr.add(whack::Array<float, 2> { 3.7f, 3.8f });
        CHECK(arr.size() == 8);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 1.0f);
        CHECK(arr[3] == 2.0f);
        CHECK(arr[4] == 3.0f);
        CHECK(arr[5] == 3.4f);
        CHECK(arr[6] == 3.6f);
        CHECK(arr[7] == 3.7f);

        arr.add(whack::Array<float, 2> { 4.7f, 5.8f });
        CHECK(arr.size() == 8);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 1.0f);
        CHECK(arr[3] == 2.0f);
        CHECK(arr[4] == 3.0f);
        CHECK(arr[5] == 3.4f);
        CHECK(arr[6] == 3.6f);
        CHECK(arr[7] == 3.7f);

        arr.add(whack::Array<float, 2> { 1.7f, 1.8f });
        CHECK(arr.size() == 8);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 1.0f);
        CHECK(arr[3] == 1.7f);
        CHECK(arr[4] == 1.8f);
        CHECK(arr[5] == 2.0f);
        CHECK(arr[6] == 3.0f);
        CHECK(arr[7] == 3.4f);

        arr.add(whack::Array<float, 2> { 0.1f, 0.55f });
        CHECK(arr.size() == 8);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.55f);
        CHECK(arr[2] == 0.6f);
        CHECK(arr[3] == 1.0f);
        CHECK(arr[4] == 1.7f);
        CHECK(arr[5] == 1.8f);
        CHECK(arr[6] == 2.0f);
        CHECK(arr[7] == 3.0f);
    }

    SECTION("bulk add random")
    {
        std::srand(0);
        const auto r = []() {
            return float(std::rand()) / float(RAND_MAX);
        };
        for (auto i = 0u; i < 1000; ++i) {
            const auto smallest = r();
            dgmr::marching_steps::Array<8> arr(smallest);
            for (auto j = 0u; j < 20; ++j) {
                auto values = whack::Array<float, 4> { r() * 0.3f, r() * 0.3f, r() * 0.3f, r() * 0.3f };
                auto s = 0.f;
                for (auto& v : values) {
                    s += v;
                    v = s;
                }
                arr.add(values);
                CHECK(arr[0] == smallest);
                auto last = smallest;
                for (auto k = 0u; k < arr.size(); ++k) {
                    auto d = arr[k];
                    CHECK(d >= last);
                    last = d;
                }
            }
        }
    }

    SECTION("bulk add to empty, previously crashing")
    {
        dgmr::marching_steps::Array<4> arr(5);

        const auto new_arr = whack::Array<float, 6> { 0, 1, 2, 3, 4, 6 };
        arr.add(new_arr);
        CHECK(arr.size() == 2);
        CHECK(arr[0] == 5.0f);
        CHECK(arr[1] == 6.0f);
    }

    SECTION("bulk add to partially empty")
    {
        dgmr::marching_steps::Array<4> arr(0.5);
        arr.add(1.5);
        arr.add(2.5);

        const auto new_arr = whack::Array<float, 6> { 0, 1, 2, 3, 4, 5 };
        arr.add(new_arr);
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 1.0f);
        CHECK(arr[2] == 1.5f);
        CHECK(arr[3] == 2.0f);
    }

    SECTION("bulk add to full")
    {
        dgmr::marching_steps::Array<4> arr(0.5);
        arr.add(1.5);
        arr.add(2.5);
        arr.add(3.5);

        arr.add(whack::Array<float, 6> { 0, 1, 2, 3, 4, 5 });
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 1.0f);
        CHECK(arr[2] == 1.5f);
        CHECK(arr[3] == 2.0f);

        arr.add(whack::Array<float, 6> { .0f, .2f, .4f, .6f, .8f, .9f });
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 0.8f);
        CHECK(arr[3] == 0.9f);

        arr.add(whack::Array<float, 6> { 1.0f, 1.2f, 1.4f, 1.6f, 1.8f, 1.9f });
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 0.8f);
        CHECK(arr[3] == 0.9f);
    }

    SECTION("bulk add to full, previously crashing")
    {
        dgmr::marching_steps::Array<4> arr(5);
        arr.add(5.5);
        arr.add(6.5);
        arr.add(7.5);

        arr.add(whack::Array<float, 6> { 0, 1, 2, 3, 4, 10 });
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 5.0f);
        CHECK(arr[1] == 5.5f);
        CHECK(arr[2] == 6.5f);
        CHECK(arr[3] == 7.5f);

        arr.add(whack::Array<float, 6> { 0, 1, 2, 3, 4, 7 });
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 5.0f);
        CHECK(arr[1] == 5.5f);
        CHECK(arr[2] == 6.5f);
        CHECK(arr[3] == 7.0f);

        arr.add(whack::Array<float, 6> { 0, 1, 2, 3, 6, 7 });
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 5.0f);
        CHECK(arr[1] == 5.5f);
        CHECK(arr[2] == 6.0f);
        CHECK(arr[3] == 6.5f);

        arr.add(whack::Array<float, 6> { 0, 1, 2, 5.1f, 6, 7 });
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 5.0f);
        CHECK(arr[1] == 5.1f);
        CHECK(arr[2] == 5.5f);
        CHECK(arr[3] == 6.0f);

        arr.add(whack::Array<float, 8> { 0, 1, 2, 5.01f, 5.02f, 5.03f, 5.04f, 5.05f });
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 5.0f);
        CHECK(arr[1] == 5.01f);
        CHECK(arr[2] == 5.02f);
        CHECK(arr[3] == 5.03f);
    }
}

TEST_CASE("dgmr marching step bins")
{
    SECTION("no subdivision")
    {
        dgmr::marching_steps::Array<4> arr(0.0);
        arr.add(1.0);
        arr.add(3.0);
        arr.add(13.0);

        const auto bins = dgmr::marching_steps::make_bins<1>(arr);
        CHECK(bins.size() == 3);

        CHECK(bins.begin_of(0) == 0.0f);
        CHECK(bins.end_of(0) == 1.0f);

        CHECK(bins.begin_of(1) == 1.0f);
        CHECK(bins.end_of(1) == 3.0f);

        CHECK(bins.begin_of(2) == 3.0f);
        CHECK(bins.end_of(2) == 13.0f);
    }

    SECTION("2 subdivisions")
    {
        dgmr::marching_steps::Array<4> arr(0.0);
        arr.add(1.0);
        arr.add(3.0);
        arr.add(13.0);

        const auto bins = dgmr::marching_steps::make_bins<2>(arr);
        CHECK(bins.size() == 6);

        CHECK(bins.begin_of(0) == Approx(0.0));
        CHECK(bins.end_of(0) == Approx(0.5));
        CHECK(bins.begin_of(1) == Approx(0.5));
        CHECK(bins.end_of(1) == Approx(1.0));
        CHECK(bins.begin_of(2) == Approx(1.0));
        CHECK(bins.end_of(2) == Approx(2.0));
        CHECK(bins.begin_of(3) == Approx(2.0));
        CHECK(bins.end_of(3) == Approx(3.0));
        CHECK(bins.begin_of(4) == Approx(3.0));
        CHECK(bins.end_of(4) == Approx(8.0));
        CHECK(bins.begin_of(5) == Approx(8.0));
        CHECK(bins.end_of(5) == Approx(13.0));
    }

    SECTION("2 subdivisions, no array elements")
    {
        dgmr::marching_steps::Array<4> arr(0.0);
        arr.add(1.0);

        const auto bins = dgmr::marching_steps::make_bins<2>(arr);
        CHECK(bins.size() == 6);

        CHECK(bins.begin_of(0) == Approx(0.0));
        CHECK(bins.end_of(0) == Approx(0.5));
        CHECK(bins.begin_of(1) == Approx(0.5));
        CHECK(bins.end_of(1) == Approx(1.0));
        CHECK(bins.begin_of(2) == Approx(1.0));
        CHECK(bins.end_of(2) == Approx(1.0));
        CHECK(bins.begin_of(3) == Approx(1.0));
        CHECK(bins.end_of(3) == Approx(1.0));
        CHECK(bins.begin_of(4) == Approx(1.0));
        CHECK(bins.end_of(4) == Approx(1.0));
        CHECK(bins.begin_of(5) == Approx(1.0));
        CHECK(bins.end_of(5) == Approx(1.0));
    }

    SECTION("2 subdivisions, array not full")
    {
        dgmr::marching_steps::Array<4> arr(0.0);

        const auto bins = dgmr::marching_steps::make_bins<2>(arr);
        CHECK(bins.size() == 6);

        CHECK(bins.begin_of(0) == Approx(0.0));
        CHECK(bins.end_of(0) == Approx(0.0));
        CHECK(bins.begin_of(1) == Approx(0.0));
        CHECK(bins.end_of(1) == Approx(0.0));
        CHECK(bins.begin_of(2) == Approx(0.0));
        CHECK(bins.end_of(2) == Approx(0.0));
        CHECK(bins.begin_of(3) == Approx(0.0));
        CHECK(bins.end_of(3) == Approx(0.0));
        CHECK(bins.begin_of(4) == Approx(0.0));
        CHECK(bins.end_of(4) == Approx(0.0));
        CHECK(bins.begin_of(5) == Approx(0.0));
        CHECK(bins.end_of(5) == Approx(0.0));
    }
}
