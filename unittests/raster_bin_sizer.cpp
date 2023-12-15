/*****************************************************************************
 * DGMR
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

#include <stroke/pretty_printers.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <dgmr/raster_bin_sizers.h>

using Catch::Approx;
namespace {
struct RasterBinSizerConfig {
    static constexpr float transmission_threshold = 0.1f;
    static constexpr unsigned n_rasterisation_steps = 4;
    static constexpr float gaussian_relevance_sigma = 1.f;
};
} // namespace

TEMPLATE_TEST_CASE("dgmr raster bin sizer", "", dgmr::math::RasterBinSizer<RasterBinSizerConfig>, dgmr::math::RasterBinSizer_1<RasterBinSizerConfig>)
{
    using RasterBinSizer = TestType;
    SECTION("start empty")
    {
        const RasterBinSizer sizer;
        CHECK(sizer.is_full() == false);
        CHECK(sizer.begin_of(0) == Approx(0));
        CHECK(sizer.end_of(0) == Approx(0));
        CHECK(sizer.begin_of(1) == Approx(0));
        CHECK(sizer.end_of(1) == Approx(0));
        CHECK(sizer.begin_of(2) == Approx(0));
        CHECK(sizer.end_of(2) == Approx(0));
    }

    SECTION("empty sizer doesn't crash")
    {
        RasterBinSizer sizer;
        sizer.finalise();
        CHECK(sizer.is_full() == false);
    }

    SECTION("single opaque gaussian")
    {
        RasterBinSizer sizer = {};
        sizer.add_gaussian(1.52f, 8, 4);
        CHECK(sizer.is_full() == true);
        sizer.finalise();
        CHECK(sizer.is_full() == true);
        CHECK(sizer.begin_of(0) == Approx(0));
        CHECK(sizer.end_of(0) == Approx(sizer.begin_of(1)));
        CHECK(sizer.end_of(1) == Approx(sizer.begin_of(2)));
        CHECK(sizer.end_of(2) == Approx(sizer.begin_of(3)));

        CHECK(sizer.begin_of(0) <= sizer.end_of(0));
        CHECK(sizer.begin_of(1) <= sizer.end_of(1));
        CHECK(sizer.begin_of(2) <= sizer.end_of(2));
        CHECK(sizer.begin_of(3) <= sizer.end_of(3));
    }

    SECTION("bad gaussian order")
    {
        RasterBinSizer sizer = {};
        sizer.add_gaussian(0.25f, 10, 2);
        CHECK(sizer.is_full() == false);
        sizer.add_gaussian(0.25f, 8, 1);
        CHECK(sizer.is_full() == false);
        sizer.add_gaussian(0.25f, 6, 2);
        CHECK(sizer.is_full() == false);
        sizer.add_gaussian(0.25f, 4, 1);
        CHECK(sizer.is_full() == false);
        sizer.add_gaussian(0.25f, 2, 2);
        CHECK(sizer.is_full() == false);
        sizer.add_gaussian(0.25f, 0, 1);
        CHECK(sizer.is_full() == false);
        sizer.finalise();
        CHECK(sizer.is_full() == false);
        CHECK(sizer.begin_of(0) == Approx(0));
        CHECK(sizer.end_of(0) == Approx(sizer.begin_of(1)));
        CHECK(sizer.end_of(1) == Approx(sizer.begin_of(2)));
        CHECK(sizer.end_of(2) == Approx(sizer.begin_of(3)));

        CHECK(sizer.begin_of(0) <= sizer.end_of(0));
        CHECK(sizer.begin_of(1) <= sizer.end_of(1));
        CHECK(sizer.begin_of(2) <= sizer.end_of(2));
        CHECK(sizer.begin_of(3) <= sizer.end_of(3));
    }

    SECTION("only one gaussian at 0.6 opacity")
    {
        RasterBinSizer sizer = {};
        sizer.add_gaussian(0.6f, 10, 2);
        CHECK(sizer.is_full() == false);
        sizer.finalise();
        CHECK(sizer.is_full() == false);
        CHECK(sizer.begin_of(0) == Approx(0));
        CHECK(sizer.end_of(0) == Approx(sizer.begin_of(1)));
        CHECK(sizer.end_of(1) == Approx(sizer.begin_of(2)));
        CHECK(sizer.end_of(2) == Approx(sizer.begin_of(3)));

        CHECK(sizer.begin_of(0) <= sizer.end_of(0));
        CHECK(sizer.begin_of(1) <= sizer.end_of(1));
        CHECK(sizer.begin_of(2) <= sizer.end_of(2));
        CHECK(sizer.begin_of(3) <= sizer.end_of(3));
    }

    SECTION("only one gaussian at 0.3 opacity")
    {
        RasterBinSizer sizer = {};
        sizer.add_gaussian(0.3f, 10, 2);
        sizer.finalise();
        CHECK(sizer.begin_of(0) == Approx(0));
        CHECK(sizer.end_of(0) == Approx(sizer.begin_of(1)));
        CHECK(sizer.end_of(1) == Approx(sizer.begin_of(2)));
        CHECK(sizer.end_of(2) == Approx(sizer.begin_of(3)));

        CHECK(sizer.begin_of(0) <= sizer.end_of(0));
        CHECK(sizer.begin_of(1) <= sizer.end_of(1));
        CHECK(sizer.begin_of(2) <= sizer.end_of(2));
        CHECK(sizer.begin_of(3) <= sizer.end_of(3));
    }

    SECTION("several gaussians, filling up")
    {
        RasterBinSizer sizer = {};
        sizer.add_gaussian(0.35f, 1.3f, 1.f); // 0.739 remains
        CHECK(!sizer.is_full());
        sizer.add_gaussian(0.40f, 4.7f, 0.81f); // 0.483 remains
        CHECK(!sizer.is_full());
        sizer.add_gaussian(0.6, 6.8, 1.f); // 0.239 remains
        CHECK(!sizer.is_full());
        sizer.add_gaussian(0.8, 9.6, 0.25f); // 0.052 remains, which is under 0.1 from the test config.
        CHECK(sizer.is_full());
        sizer.finalise();
        CHECK(sizer.begin_of(0) == Approx(0));
        CHECK(sizer.end_of(0) == Approx(sizer.begin_of(1)));
        CHECK(sizer.end_of(1) == Approx(sizer.begin_of(2)));
        CHECK(sizer.end_of(2) == Approx(sizer.begin_of(3)));

        CHECK(sizer.begin_of(0) <= sizer.end_of(0));
        CHECK(sizer.begin_of(1) <= sizer.end_of(1));
        CHECK(sizer.begin_of(2) <= sizer.end_of(2));
        CHECK(sizer.begin_of(3) <= sizer.end_of(3));
    }

    SECTION("one gap in transmission stops")
    {
        RasterBinSizer sizer = {};
        sizer.add_gaussian(0.7f, 4.7f, 0.81f); // 0.393 remains, jumps over 0.75
        sizer.add_gaussian(0.6, 6.8, 1.f); // 0.195 remains
        sizer.add_gaussian(0.8, 9.6, 0.25f); // 0.042 remains, which is under 0.1 from the test config.
        sizer.finalise();
        CHECK(sizer.begin_of(0) == Approx(0));
        CHECK(sizer.end_of(0) == Approx(sizer.begin_of(1)));
        CHECK(sizer.end_of(1) == Approx(sizer.begin_of(2)));
        CHECK(sizer.end_of(2) == Approx(sizer.begin_of(3)));

        CHECK(sizer.begin_of(0) <= sizer.end_of(0));
        CHECK(sizer.begin_of(1) <= sizer.end_of(1));
        CHECK(sizer.begin_of(2) <= sizer.end_of(2));
        CHECK(sizer.begin_of(3) <= sizer.end_of(3));
    }

    SECTION("one gap in transmission stops, random order")
    {
        RasterBinSizer sizer = {};
        sizer.add_gaussian(0.7f, 4.7f, 0.81f);
        sizer.add_gaussian(0.8, 9.6, 0.25f);
        sizer.add_gaussian(0.6, 6.8, 1.f);
        sizer.finalise();
        CHECK(sizer.begin_of(0) == Approx(0));
        CHECK(sizer.end_of(0) == Approx(sizer.begin_of(1)));
        CHECK(sizer.end_of(1) == Approx(sizer.begin_of(2)));
        CHECK(sizer.end_of(2) == Approx(sizer.begin_of(3)));

        CHECK(sizer.begin_of(0) <= sizer.end_of(0));
        CHECK(sizer.begin_of(1) <= sizer.end_of(1));
        CHECK(sizer.begin_of(2) <= sizer.end_of(2));
        CHECK(sizer.begin_of(3) <= sizer.end_of(3));
    }

    SECTION("two gaps in transmission stops")
    {
        RasterBinSizer sizer = {};
        sizer.add_gaussian(0.6, 6.8, 1.f); // 0.195 remains
        sizer.add_gaussian(0.9, 9.6, 0.25f); // 0.042 remains, which is under 0.1 from the test config.
        sizer.finalise();
        CHECK(sizer.begin_of(0) == Approx(0));
        CHECK(sizer.end_of(0) == Approx(sizer.begin_of(1)));
        CHECK(sizer.end_of(1) == Approx(sizer.begin_of(2)));
        CHECK(sizer.end_of(2) == Approx(sizer.begin_of(3)));

        CHECK(sizer.begin_of(0) <= sizer.end_of(0));
        CHECK(sizer.begin_of(1) <= sizer.end_of(1));
        CHECK(sizer.begin_of(2) <= sizer.end_of(2));
        CHECK(sizer.begin_of(3) <= sizer.end_of(3));
    }

    SECTION("many small, random order")
    {
        RasterBinSizer sizer = {};
        sizer.add_gaussian(0.08, 1.8, 1.33f);
        sizer.add_gaussian(0.05, 4.6, 0.25f);
        sizer.add_gaussian(0.18, 2.8, 1.22f);
        sizer.add_gaussian(0.06, 5.0, 0.33f);
        sizer.add_gaussian(0.08, 3.8, 1.45f);
        sizer.add_gaussian(0.23, 9.6, 0.25f);
        sizer.add_gaussian(0.18, 6.0, 1.20f);
        sizer.add_gaussian(0.05, 4.6, 0.25f);
        sizer.add_gaussian(0.21, 6.8, 1.00f);
        sizer.add_gaussian(0.38, 10.2, 0.25f);
        sizer.add_gaussian(0.26, 10.8, 1.00f);
        sizer.add_gaussian(0.28, 10.3, 0.25f);
        sizer.add_gaussian(0.21, 10.5, 1.00f);
        sizer.add_gaussian(0.37, 10.9, 0.25f);
        sizer.finalise();
        CHECK(sizer.begin_of(0) == Approx(0));
        CHECK(sizer.end_of(0) == Approx(sizer.begin_of(1)));
        CHECK(sizer.end_of(1) == Approx(sizer.begin_of(2)));
        CHECK(sizer.end_of(2) == Approx(sizer.begin_of(3)));

        CHECK(sizer.begin_of(0) <= sizer.end_of(0));
        CHECK(sizer.begin_of(1) <= sizer.end_of(1));
        CHECK(sizer.begin_of(2) <= sizer.end_of(2));
        CHECK(sizer.begin_of(3) <= sizer.end_of(3));
    }
}
