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
