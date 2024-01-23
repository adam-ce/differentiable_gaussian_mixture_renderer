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

#include <stroke/pretty_printers.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <dgmr/math.h>

#include "unit_test_utils.h"

namespace {

using namespace dgmr::unittest;
using namespace dgmr::math;

template <dgmr::Formulation formulation>
void check_splat(double filter_kernel_size)
{
    using scalar_t = double;
    using vec3_t = glm::vec<3, scalar_t>;
    using cov3_t = stroke::Cov3<scalar_t>;

    whack::random::HostGenerator<scalar_t> rnd;

    for (int i = 0; i < 10; ++i) {
        const auto cam = random_camera<scalar_t>(&rnd);

        const auto weight = rnd.uniform();
        const auto position = rnd.normal3();
        const auto scales = rnd.uniform3();
        const auto rot = host_random_rot<scalar_t>(&rnd);
        Gaussian2d<scalar_t> g = splat<formulation, scalar_t>(weight, position, scales, rot, cam, scalar_t(filter_kernel_size));
        Gaussian2dAndValueCache<scalar_t> gc = splat_with_cache<formulation, scalar_t>(weight, position, scales, rot, cam, scalar_t(filter_kernel_size));
        CHECK(g.weight == gc.gaussian.weight);
        CHECK(g.centroid == gc.gaussian.centroid);
        CHECK(g.cov == gc.gaussian.cov);
    }
}

} // namespace

TEST_CASE("dgmr utils: splat vs splat_with_cache (with opacity, filter kernel size 0)")
{
    check_splat<dgmr::Formulation::Opacity>(0);
}
TEST_CASE("dgmr utils: splat vs splat_with_cache (with opacity, filter kernel size 0.3)")
{
    check_splat<dgmr::Formulation::Opacity>(0.3);
}
TEST_CASE("dgmr utils: splat vs splat_with_cache (with mass, filter kernel size 0)")
{
    check_splat<dgmr::Formulation::Mass>(0);
}
TEST_CASE("dgmr utils: splat vs splat_with_cache (with mass, filter kernel size 0.3)")
{
    check_splat<dgmr::Formulation::Mass>(0.3);
}
TEST_CASE("dgmr utils: splat vs splat_with_cache (with density, filter kernel size 0)")
{
    check_splat<dgmr::Formulation::Density>(0);
}
TEST_CASE("dgmr utils: splat vs splat_with_cache (with density, filter kernel size 0.3)")
{
    check_splat<dgmr::Formulation::Density>(0.3);
}
TEST_CASE("dgmr utils: splat vs splat_with_cache (with Ots, filter kernel size 0)")
{
    check_splat<dgmr::Formulation::Ots>(0);
}
TEST_CASE("dgmr utils: splat vs splat_with_cache (with Ots, filter kernel size 0.3)")
{
    check_splat<dgmr::Formulation::Ots>(0.3);
}

TEST_CASE("dgmr utils: rotation matrix for rotating z into given direction")
{
    CHECK(dgmr::math::rotation_matrix_from(glm::vec3(1, 0, 0)) * glm::vec3(0, 0, 1) == glm::vec3(1, 0, 0));
    CHECK(dgmr::math::rotation_matrix_from(glm::vec3(0, 1, 0)) * glm::vec3(0, 0, 1) == glm::vec3(0, 1, 0));
    CHECK(dgmr::math::rotation_matrix_from(glm::vec3(0, 0, 1)) * glm::vec3(0, 0, 1) == glm::vec3(0, 0, 1));
    CHECK(dgmr::math::rotation_matrix_from(glm::vec3(-1, 0, 0)) * glm::vec3(0, 0, 1) == glm::vec3(-1, 0, 0));
    CHECK(dgmr::math::rotation_matrix_from(glm::vec3(0, -1, 0)) * glm::vec3(0, 0, 1) == glm::vec3(0, -1, 0));
    CHECK(dgmr::math::rotation_matrix_from(glm::vec3(0, 0, -1)) * glm::vec3(0, 0, 1) == glm::vec3(0, 0, -1));

    CHECK(dgmr::math::rotation_matrix_from(glm::normalize(glm::vec3(-0.8, .7, 0.2))) * glm::vec3(0, 0, 1) == glm::normalize(glm::vec3(-0.8, .7, 0.2)));
    CHECK(dgmr::math::rotation_matrix_from(glm::normalize(glm::vec3(0.8, -.7, -0.2))) * glm::vec3(0, 0, 1) == glm::normalize(glm::vec3(0.8, -.7, -0.2)));
    CHECK(dgmr::math::rotation_matrix_from(glm::normalize(glm::vec3(0.0, .7, -0.2))) * glm::vec3(0, 0, 1) == glm::normalize(glm::vec3(0.0, .7, -0.2)));
}
TEST_CASE("dgmr utils: oriented filter kernel")
{
    CHECK(dgmr::math::orient_filter_kernel<float>({ .direction = glm::vec3(1, 0, 0), .kernel_scales = glm::vec3(0, 0, 1) }) == stroke::Cov3(1.f, 0.f, 0.f, 0.f, 0.f, 0.f));
    CHECK(dgmr::math::orient_filter_kernel<float>({ .direction = glm::vec3(1, 0, 0), .kernel_scales = glm::vec3(1, 1, 0) }) == stroke::Cov3(0.f, 0.f, 0.f, 1.f, 0.f, 1.f));
    CHECK(dgmr::math::orient_filter_kernel<float>({ .direction = glm::vec3(0, 1, 0), .kernel_scales = glm::vec3(0, 0, 1) }) == stroke::Cov3(0.f, 0.f, 0.f, 1.f, 0.f, 0.f));
    CHECK(dgmr::math::orient_filter_kernel<float>({ .direction = glm::vec3(0, 1, 0), .kernel_scales = glm::vec3(1, 1, 0) }) == stroke::Cov3(1.f, 0.f, 0.f, 0.f, 0.f, 1.f));
    CHECK(dgmr::math::orient_filter_kernel<float>({ .direction = glm::vec3(0, 0, 1), .kernel_scales = glm::vec3(0, 0, 1) }) == stroke::Cov3(0.f, 0.f, 0.f, 0.f, 0.f, 1.f));
    CHECK(dgmr::math::orient_filter_kernel<float>({ .direction = glm::vec3(0, 0, 1), .kernel_scales = glm::vec3(1, 1, 0) }) == stroke::Cov3(1.f, 0.f, 0.f, 1.f, 0.f, 0.f));

    CHECK(dgmr::math::orient_filter_kernel<float>({ .direction = glm::vec3(1, 0, 0), .kernel_scales = glm::vec3(0, 0, 10) }) == stroke::Cov3(100.f, 0.f, 0.f, 0.f, 0.f, 0.f));
    CHECK(dgmr::math::orient_filter_kernel<float>({ .direction = glm::vec3(1, 0, 0), .kernel_scales = glm::vec3(10, 10, 0) }) == stroke::Cov3(0.f, 0.f, 0.f, 100.f, 0.f, 100.f));
    CHECK(dgmr::math::orient_filter_kernel<float>({ .direction = glm::vec3(0, 1, 0), .kernel_scales = glm::vec3(0, 0, 10) }) == stroke::Cov3(0.f, 0.f, 0.f, 100.f, 0.f, 0.f));
    CHECK(dgmr::math::orient_filter_kernel<float>({ .direction = glm::vec3(0, 1, 0), .kernel_scales = glm::vec3(10, 10, 0) }) == stroke::Cov3(100.f, 0.f, 0.f, 0.f, 0.f, 100.f));
    CHECK(dgmr::math::orient_filter_kernel<float>({ .direction = glm::vec3(0, 0, 1), .kernel_scales = glm::vec3(0, 0, 10) }) == stroke::Cov3(0.f, 0.f, 0.f, 0.f, 0.f, 100.f));
    CHECK(dgmr::math::orient_filter_kernel<float>({ .direction = glm::vec3(0, 0, 1), .kernel_scales = glm::vec3(10, 10, 0) }) == stroke::Cov3(100.f, 0.f, 0.f, 100.f, 0.f, 0.f));
}

TEST_CASE("dgmr utils: gaussian_bounds")
{

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
        const auto bounds = dgmr::math::gaussian_to_point_distance_bounds(d.gauss_centr, d.gauss_size, d.gauss_rotation, d.gauss_iso_ellipsoid, d.query_point);

        CHECK(bounds.min == Catch::Approx(d.expected_bounds.min).scale(10));
        CHECK(bounds.max == Catch::Approx(d.expected_bounds.max).scale(10));
    }
}

TEST_CASE("dgmr utils: larger 2") {
    const auto check = [](glm::ivec2 v) {
        return v == glm::ivec2(2, 3) || v == glm::ivec2(3, 2);
    };

    CHECK(check(dgmr::math::larger2(glm::ivec3(1, 2, 3))));
    CHECK(check(dgmr::math::larger2(glm::ivec3(1, 3, 2))));
    CHECK(check(dgmr::math::larger2(glm::ivec3(2, 3, 1))));
    CHECK(check(dgmr::math::larger2(glm::ivec3(2, 1, 3))));
    CHECK(check(dgmr::math::larger2(glm::ivec3(3, 2, 1))));
    CHECK(check(dgmr::math::larger2(glm::ivec3(3, 1, 2))));

}
