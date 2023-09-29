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
 */

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <dgmr/utils.h>

using Catch::Approx;
namespace {
struct RasterBinSizerConfig {
	static constexpr float transmission_threshold = 0.1f;
	static constexpr unsigned n_rasterisation_steps = 4;
};
// template <int n_dims>
// bool equals(const glm::vec<n_dims, double>& a, const glm::vec<n_dims, double>& b, double scale = 1) {
//	const auto delta = glm::length(a - b);
//	return delta == Approx(0).scale(scale);
// }
}

TEST_CASE("dgmr utils: gaussian_bounds") {

	struct D {
		glm::vec3 gauss_centr;
		glm::vec3 gauss_size;
		glm::quat gauss_rotation;
		float gauss_iso_ellipsoid;
		glm::vec3 query_point;
		stroke::geometry::Aabb1f expected_bounds;
	};
	const auto data = std::array {
		D { glm::vec3(0, 0, 0), glm::vec3(1, 1, 1), glm::quat(1, glm::vec3(0, 0, 0)), 1.f, glm::vec3(0, 0, 0), stroke::geometry::Aabb1f { 0, std::sqrt(0.75f) } },
		D { glm::vec3(0, 0, 0), glm::vec3(1, 1, 1), glm::quat(1, glm::vec3(0, 0, 0)), 2.f, glm::vec3(0, 0, 0), stroke::geometry::Aabb1f { 0, std::sqrt(3.f) } },
		D { glm::vec3(0, 0, 0), glm::vec3(1, 1, 1), glm::quat(0, glm::normalize(glm::vec3(1.65, 0.547, -.157))), 2.f, glm::vec3(0, 0, 0), stroke::geometry::Aabb1f { 0, std::sqrt(3.f) } },
		D { glm::vec3(0, 0, 0), glm::vec3(1, 1, 1), glm::quat(1, glm::normalize(glm::vec3(1.65, 0.547, -.157))), 2.f, glm::vec3(0, 0, 0), stroke::geometry::Aabb1f { 0, std::sqrt(3.f) } },
		D { glm::vec3(0, 0, 0), glm::vec3(1, 1, 1), glm::quat(1, glm::vec3(0, 0, 0)), 1.f, glm::vec3(10, 0, 0), stroke::geometry::Aabb1f { 9.5, std::sqrt(10.5f * 10.5f + 0.25f * 2) } },
		D { glm::vec3(0, 0, 0), glm::vec3(1, 1, 1), glm::quat(1, glm::vec3(0, 0, 0)), 2.f, glm::vec3(0, 10, 0), stroke::geometry::Aabb1f { 9.f, std::sqrt(11.f * 11.f + 1.f * 2) } },
		D { glm::vec3(0, 0, 0), glm::vec3(1, 1, 1), glm::quat(0, glm::vec3(1, 0, 0)), 2.f, glm::vec3(0, 10, 0), stroke::geometry::Aabb1f { 9.f, std::sqrt(11.f * 11.f + 1.f * 2) } },

		// quaternion calculator: https://ninja-calc.mbedded.ninja/calculators/mathematics/geometry/3d-rotations
		D { glm::vec3(0, 0, 0), glm::vec3(1, 10, 1), glm::quat(std::sqrt(0.5f), 0, 0, std::sqrt(0.5f)), 2.f, glm::vec3(11, 0, 0), stroke::geometry::Aabb1f { 1.f, std::sqrt(21.f * 21.f + 1.f * 2) } },
		D { glm::vec3(1, 2, 3), glm::vec3(1, 10, 1), glm::quat(std::sqrt(0.5f), 0, 0, std::sqrt(0.5f)), 2.f, glm::vec3(12, 2, 3), stroke::geometry::Aabb1f { 1.f, std::sqrt(21.f * 21.f + 1.f * 2) } },

	};
	for (const auto& d : data) {
		const auto bounds = dgmr::utils::gaussian_to_point_distance_bounds(d.gauss_centr, d.gauss_size, d.gauss_rotation, d.gauss_iso_ellipsoid, d.query_point);

		CHECK(bounds.min == Approx(d.expected_bounds.min).scale(10));
		CHECK(bounds.max == Approx(d.expected_bounds.max).scale(10));
	}
}

TEST_CASE("dgmr utils: raster bin sizer") {

	SECTION("start empty") {
		const dgmr::utils::RasterBinSizer<RasterBinSizerConfig> sizer;
		CHECK(sizer.begin_of(0) == Approx(0));
		CHECK(sizer.end_of(0) == Approx(0));
		CHECK(sizer.begin_of(1) == Approx(0));
		CHECK(sizer.end_of(1) == Approx(0));
		CHECK(sizer.begin_of(2) == Approx(0));
		CHECK(sizer.end_of(2) == Approx(0));
	}
	SECTION("bin begin and end") {
		const dgmr::utils::RasterBinSizer<RasterBinSizerConfig> sizer = { 1.f, 2.f, 3.f, 4.f };
		CHECK(sizer.begin_of(0) == Approx(0));
		CHECK(sizer.end_of(0) == Approx(1));
		CHECK(sizer.begin_of(1) == Approx(1));
		CHECK(sizer.end_of(1) == Approx(2));
		CHECK(sizer.begin_of(2) == Approx(2));
		CHECK(sizer.end_of(2) == Approx(3));
		CHECK(sizer.begin_of(3) == Approx(3));
		CHECK(sizer.end_of(3) == Approx(4));
	}
	SECTION("single opaque element") {
		dgmr::utils::RasterBinSizer<RasterBinSizerConfig> sizer;
		CHECK(!sizer.is_full());
		sizer.add_opacity(10, 0.99);
		CHECK(sizer.is_full());
		sizer.finalise();
		CHECK(sizer.begin_of(0) == Approx(0));
		CHECK(sizer.end_of(0) == Approx(2.5));
		CHECK(sizer.begin_of(1) == Approx(2.5));
		CHECK(sizer.end_of(1) == Approx(5));
		CHECK(sizer.begin_of(2) == Approx(5));
		CHECK(sizer.end_of(2) == Approx(7.5));
		CHECK(sizer.begin_of(3) == Approx(7.5));
		CHECK(sizer.end_of(3) == Approx(10));
	}
	SECTION("enough elements for every transmission stop") {
		dgmr::utils::RasterBinSizer<RasterBinSizerConfig> sizer;
		sizer.add_opacity(2.3, 0.26); // 0.74 remains
		CHECK(!sizer.is_full());
		sizer.add_opacity(5.6, 0.35); // 0.481 remains
		CHECK(!sizer.is_full());
		sizer.add_opacity(7.8, 0.5); // 0.2405 remains
		CHECK(!sizer.is_full());
		sizer.add_opacity(10.1, 0.8); // 0.0481 remains, which is under 0.1 from the test config.
		CHECK(sizer.is_full());
		sizer.finalise();
		CHECK(sizer.end_of(0) == Approx(2.3));
		CHECK(sizer.end_of(1) == Approx(5.6));
		CHECK(sizer.end_of(2) == Approx(7.8));
		CHECK(sizer.end_of(3) == Approx(10.1));
		CHECK(sizer.max_distance() == Approx(10.1));
	}
	SECTION("one gap in transmission stops") {
		dgmr::utils::RasterBinSizer<RasterBinSizerConfig> sizer;
		sizer.add_opacity(2.4, 0.6); // 0.4 remains, jumps over 0.75
		sizer.add_opacity(5.6, 0.5); // 0.2 remains
		sizer.add_opacity(7.8, 0.9); // 0.02 remains
		sizer.finalise();
		CHECK(sizer.end_of(0) == Approx(1.2));
		CHECK(sizer.end_of(1) == Approx(2.4));
		CHECK(sizer.end_of(2) == Approx(5.6));
		CHECK(sizer.end_of(3) == Approx(7.8));
		CHECK(sizer.max_distance() == Approx(7.8));
	}
	SECTION("two gaps in transmission stops") {
		dgmr::utils::RasterBinSizer<RasterBinSizerConfig> sizer;
		sizer.add_opacity(50, 0.6); // 0.4 remains, jumps over 0.75
		sizer.add_opacity(100, 0.9); // 0.04 remains
		sizer.finalise();
		CHECK(sizer.end_of(0) == Approx(25));
		CHECK(sizer.end_of(1) == Approx(50));
		CHECK(sizer.end_of(2) == Approx(75));
		CHECK(sizer.end_of(3) == Approx(100));
		CHECK(sizer.max_distance() == Approx(100));
	}
}
