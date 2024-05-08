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
        arr.add(2.5);
        CHECK(arr.size() == 2);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 2.5f);

        arr.add(1.5);
        CHECK(arr.size() == 3);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 1.5f);
        CHECK(arr[2] == 2.5f);

        arr.add(4.5);
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 1.5f);
        CHECK(arr[2] == 2.5f);
        CHECK(arr[3] == 4.5f);

        arr.add(3.5);
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 1.5f);
        CHECK(arr[2] == 2.5f);
        CHECK(arr[3] == 3.5f);

        arr.add(0.6);
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 1.5f);
        CHECK(arr[3] == 2.5f);

        arr.add(0.7);
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 0.7f);
        CHECK(arr[3] == 1.5f);

        arr.add(0.8);
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 0.7f);
        CHECK(arr[3] == 0.8f);
    }

    SECTION("ignore stuff before start")
    {
        dgmr::marching_steps::Array<4> arr(0.5);
        arr.add(1.5);
        arr.add(2.5);
        arr.add(3.5);
        arr.add(0.0);
        arr.add(0.1);
        arr.add(0.2);
        arr.add(0.3);
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
        arr.add(whack::Array<float, 4> { 0, .1, .2, .3 });
        CHECK(arr.size() == 1);

        arr.add(whack::Array<float, 4> { 0, 1, 2, 3 });
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 1.0f);
        CHECK(arr[2] == 2.0f);
        CHECK(arr[3] == 3.0f);

        arr.add(whack::Array<float, 4> { 0, 0.6, 3.5, 4.5 });
        CHECK(arr.size() == 7);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 1.0f);
        CHECK(arr[3] == 2.0f);
        CHECK(arr[4] == 3.0f);
        CHECK(arr[5] == 3.5f);
        CHECK(arr[6] == 4.5f);

        arr.add(whack::Array<float, 4> { 3.4, 3.6, 3.7, 3.8 });
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

        arr.add(whack::Array<float, 2> { 0.6, 3.0 });
        CHECK(arr.size() == 5);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 1.0f);
        CHECK(arr[3] == 2.0f);
        CHECK(arr[4] == 3.0f);

        arr.add(whack::Array<float, 2> { 3.4, 3.6 });
        CHECK(arr.size() == 7);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 1.0f);
        CHECK(arr[3] == 2.0f);
        CHECK(arr[4] == 3.0f);
        CHECK(arr[5] == 3.4f);
        CHECK(arr[6] == 3.6f);

        arr.add(whack::Array<float, 2> { 3.7, 3.8 });
        CHECK(arr.size() == 8);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 1.0f);
        CHECK(arr[3] == 2.0f);
        CHECK(arr[4] == 3.0f);
        CHECK(arr[5] == 3.4f);
        CHECK(arr[6] == 3.6f);
        CHECK(arr[7] == 3.7f);

        arr.add(whack::Array<float, 2> { 4.7, 5.8 });
        CHECK(arr.size() == 8);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 1.0f);
        CHECK(arr[3] == 2.0f);
        CHECK(arr[4] == 3.0f);
        CHECK(arr[5] == 3.4f);
        CHECK(arr[6] == 3.6f);
        CHECK(arr[7] == 3.7f);

        arr.add(whack::Array<float, 2> { 1.7, 1.8 });
        CHECK(arr.size() == 8);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 1.0f);
        CHECK(arr[3] == 1.7f);
        CHECK(arr[4] == 1.8f);
        CHECK(arr[5] == 2.0f);
        CHECK(arr[6] == 3.0f);
        CHECK(arr[7] == 3.4f);

        arr.add(whack::Array<float, 2> { 0.1, 0.55 });
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
            return std::rand() / float(RAND_MAX);
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

        arr.add(whack::Array<float, 6> { .0, .2, .4, .6, .8, .9 });
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 0.5f);
        CHECK(arr[1] == 0.6f);
        CHECK(arr[2] == 0.8f);
        CHECK(arr[3] == 0.9f);

        arr.add(whack::Array<float, 6> { 1.0, 1.2, 1.4, 1.6, 1.8, 1.9 });
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

        arr.add(whack::Array<float, 6> { 0, 1, 2, 5.1, 6, 7 });
        CHECK(arr.size() == 4);
        CHECK(arr[0] == 5.0f);
        CHECK(arr[1] == 5.1f);
        CHECK(arr[2] == 5.5f);
        CHECK(arr[3] == 6.0f);

        arr.add(whack::Array<float, 8> { 0, 1, 2, 5.01, 5.02, 5.03, 5.04, 5.05 });
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

        CHECK(bins.begin_of(0) == 0.0);
        CHECK(bins.end_of(0) == 1.0);

        CHECK(bins.begin_of(1) == 1.0);
        CHECK(bins.end_of(1) == 3.0);

        CHECK(bins.begin_of(2) == 3.0);
        CHECK(bins.end_of(2) == 13.0);
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
