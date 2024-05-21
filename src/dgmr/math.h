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

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <stroke/cuda_compat.h>
#include <stroke/gaussian.h>
#include <stroke/geometry.h>
#include <stroke/linalg.h>
#include <stroke/utility.h>
#include <stroke/welford.h>

#include "constants.h"
#include "types.h"

namespace dgmr::math {

template <typename scalar_t>
struct Gaussian2d {
    scalar_t weight;
    glm::vec<2, scalar_t> centroid;
    stroke::Cov2<scalar_t> cov;
};

template <typename scalar_t>
struct Gaussian2dAndValueCache {
    using mat3_t = glm::mat<3, 3, scalar_t>;
    using vec3_t = glm::vec<3, scalar_t>;
    Gaussian2d<scalar_t> gaussian;
    const mat3_t T;
    const vec3_t unclamped_t;
};

template <typename scalar_t>
struct Camera {
    using mat_t = glm::mat<4, 4, scalar_t>;
    mat_t view_matrix = {};
    mat_t view_projection_matrix = {};
    scalar_t focal_x = 0;
    scalar_t focal_y = 0;
    scalar_t tan_fovx = 0;
    scalar_t tan_fovy = 0;
    unsigned fb_width = 0;
    unsigned fb_height = 0;
};

template <typename scalar_t>
STROKE_DEVICES_INLINE stroke::Cov2<scalar_t> affine_transform_and_cut(const stroke::Cov<3, scalar_t>& S, const glm::mat<3, 3, scalar_t>& M)
{
    return {
        M[0][0] * (S[0] * M[0][0] + S[1] * M[1][0] + S[2] * M[2][0]) + M[1][0] * (S[1] * M[0][0] + S[3] * M[1][0] + S[4] * M[2][0]) + M[2][0] * (S[2] * M[0][0] + S[4] * M[1][0] + S[5] * M[2][0]),
        M[0][0] * (S[0] * M[0][1] + S[1] * M[1][1] + S[2] * M[2][1]) + M[1][0] * (S[1] * M[0][1] + S[3] * M[1][1] + S[4] * M[2][1]) + M[2][0] * (S[2] * M[0][1] + S[4] * M[1][1] + S[5] * M[2][1]),
        M[0][1] * (S[0] * M[0][1] + S[1] * M[1][1] + S[2] * M[2][1]) + M[1][1] * (S[1] * M[0][1] + S[3] * M[1][1] + S[4] * M[2][1]) + M[2][1] * (S[2] * M[0][1] + S[4] * M[1][1] + S[5] * M[2][1])
    };
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t integrate_exponential(const glm::vec<2, scalar_t>& scales)
{
    constexpr auto factor = scalar_t(gcem::sqrt(gcem::pow(2 * glm::pi<double>(), double(2))));
    return factor * scales.x * scales.y;
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t integrate_exponential(const glm::vec<3, scalar_t>& scales)
{
    constexpr auto factor = scalar_t(gcem::sqrt(gcem::pow(2 * glm::pi<double>(), double(3))));
    return factor * scales.x * scales.y * scales.z;
}

template <typename scalar_t>
STROKE_DEVICES_INLINE glm::vec<3, scalar_t> project(const glm::vec<3, scalar_t>& point, const glm::mat<4, 4, scalar_t>& projection_matrix)
{
    auto pp = projection_matrix * glm::vec<4, scalar_t>(point, 1);
    pp /= pp.w + scalar_t(0.0000001);
    return glm::vec<3, scalar_t>(pp);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE glm::vec<2, scalar_t> ndc2screen(const glm::vec<3, scalar_t>& point, unsigned width, unsigned height)
{
    const auto ndc2Pix = [](scalar_t v, int S) {
        return ((v + scalar_t(1.0)) * S - scalar_t(1.0)) * scalar_t(0.5);
    };
    return { ndc2Pix(point.x, width), ndc2Pix(point.y, height) };
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t max(const glm::vec<3, scalar_t>& vec)
{
    return stroke::max(vec.x, stroke::max(vec.y, vec.z));
}

template <typename scalar_t>
STROKE_DEVICES_INLINE glm::vec<2, scalar_t> larger2(const glm::vec<3, scalar_t>& vec)
{
    // if (vec[0] < vec[1] && vec[0] < vec[2])
    //     return glm::vec<2, scalar_t>(vec[1], vec[2]);

    // if (vec[1] < vec[0] && vec[1] < vec[2])
    //     return glm::vec<2, scalar_t>(vec[0], vec[2]);

    // return glm::vec<2, scalar_t>(vec[0], vec[1]);

    if (vec[0] < vec[1]) {
        if (vec[0] < vec[2])
            return glm::vec<2, scalar_t>(vec[1], vec[2]);
    }
    else {
        if (vec[1] < vec[2])
            return glm::vec<2, scalar_t>(vec[0], vec[2]);
    }

    return glm::vec<2, scalar_t>(vec[0], vec[1]);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE glm::vec<2, scalar_t> smaller2(const glm::vec<3, scalar_t>& vec)
{
    // if (vec[0] < vec[1] && vec[0] < vec[2])
    //     return glm::vec<2, scalar_t>(vec[1], vec[2]);

    // if (vec[1] < vec[0] && vec[1] < vec[2])
    //     return glm::vec<2, scalar_t>(vec[0], vec[2]);

    // return glm::vec<2, scalar_t>(vec[0], vec[1]);

    if (vec[0] > vec[1]) {
        if (vec[0] > vec[2])
            return glm::vec<2, scalar_t>(vec[1], vec[2]);
    } else {
        if (vec[1] > vec[2])
            return glm::vec<2, scalar_t>(vec[0], vec[2]);
    }

    return glm::vec<2, scalar_t>(vec[0], vec[1]);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE glm::mat<3, 3, scalar_t> make_jakobian(const glm::vec<3, scalar_t>& t, scalar_t l_prime, scalar_t focal_x = 1, scalar_t focal_y = 1)
{
    using mat3_t = glm::mat<3, 3, scalar_t>;
    using mat3_col_t = typename mat3_t::col_type;
    // clang-format off
    return mat3_t(
               mat3_col_t(                 focal_x / t.z,                                  0,      t.x / l_prime),
               mat3_col_t(                             0,                      focal_y / t.z,      t.y / l_prime),
               mat3_col_t(-(focal_x * t.x) / (t.z * t.z),     -(focal_y * t.y) / (t.z * t.z),      t.z / l_prime));
    // clang-format on
}

template <typename scalar_t>
STROKE_DEVICES_INLINE stroke::Cov3<scalar_t> compute_cov(const glm::vec<3, scalar_t>& scale, const glm::qua<scalar_t>& rot)
{
    const auto RS = glm::toMat3(rot) * glm::mat<3, 3, scalar_t>(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
    return stroke::Cov3<scalar_t>(RS * transpose(RS)); // == R * S * S' * R' == R * (S * S) * R' // (RS)' == S'R'
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
using FilteredCovAndWeight = cuda::std::tuple<stroke::Cov<n_dims, scalar_t>, scalar_t>;

// kernel cov comes from a normalised gaussian, i.e., it integrates to 1 and has no explicit weight
template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE FilteredCovAndWeight<n_dims, scalar_t> convolve_unnormalised_with_normalised(const stroke::Cov<n_dims, scalar_t>& cov, const stroke::Cov<n_dims, scalar_t>& kernel_cov)
{
    const auto new_cov = cov + kernel_cov;
    return { stroke::Cov<n_dims, scalar_t>(new_cov), scalar_t(stroke::sqrt(stroke::max(scalar_t(0.000025), scalar_t(det(cov) / det(new_cov))))) };
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

STROKE_DEVICES_INLINE cuda::std::tuple<glm::vec3, glm::bvec3> sh_to_colour(const SHs<3>& sh, int deg, const glm::vec3& direction)
{
    // based on Differentiable Point-Based Radiance Fields for Efficient View Synthesis by Zhang et al. (2022)
    glm::vec3 result = SH_C0 * sh[0];
    if (deg > 0) {
        float x = direction.x;
        float y = direction.y;
        float z = direction.z;
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

        if (deg > 1) {
            float xx = x * x;
            float yy = y * y;
            float zz = z * z;
            float xy = x * y;
            float yz = y * z;
            float xz = x * z;
            result = result + SH_C2[0] * xy * sh[4] + SH_C2[1] * yz * sh[5] + SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] + SH_C2[3] * xz * sh[7] + SH_C2[4] * (xx - yy) * sh[8];

            if (deg > 2) {
                result = result + SH_C3[0] * y * (3.0f * xx - yy) * sh[9] + SH_C3[1] * xy * z * sh[10] + SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] + SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] + SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] + SH_C3[5] * z * (xx - yy) * sh[14] + SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }
    result += 0.5f;

    return { glm::max(result, 0.0f), glm::greaterThan(result, glm::vec3(0)) };
}

template <Formulation formulation, typename scalar_t>
STROKE_DEVICES_INLINE scalar_t weight_to_density(scalar_t weight, const glm::vec<3, scalar_t>& cov3d_scale)
{
    switch (formulation) {
    case Formulation::Opacity: {
        assert(false);
        return 0;
    }
    case Formulation::Mass: {
        return weight / integrate_exponential(cov3d_scale);
    }
    case Formulation::Density: {
        return weight;
    }
    case Formulation::Ots: {
        const auto i2prime = math::integrate_exponential(larger2(cov3d_scale));
        return weight * i2prime / integrate_exponential(cov3d_scale);
    }
    case Formulation::Ols: {
        const auto i2prime = math::integrate_exponential(smaller2(cov3d_scale));
        return weight * i2prime / integrate_exponential(cov3d_scale);
    }
    }
}

template <Formulation formulation, typename scalar_t>
STROKE_DEVICES_INLINE scalar_t weight_to_mass(scalar_t weight, const glm::vec<3, scalar_t>& cov3d_scale)
{
    switch (formulation) {
    case Formulation::Opacity: {
        assert(false);
        return 0;
    }
    case Formulation::Mass: {
        return weight;
    }
    case Formulation::Density: {
        return weight * integrate_exponential(cov3d_scale);
    }
    case Formulation::Ots: {
        const auto i2prime = math::integrate_exponential(larger2(cov3d_scale));
        return weight * i2prime;
    }
    case Formulation::Ols: {
        const auto i2prime = math::integrate_exponential(smaller2(cov3d_scale));
        return weight * i2prime;
    }
    }
}

template <Formulation formulation, typename scalar_t>
STROKE_DEVICES_INLINE Gaussian2d<scalar_t> splat(
    scalar_t weight,
    const glm::vec<3, scalar_t>& centroid,
    const glm::vec<3, scalar_t>& cov3d_scale,
    const glm::qua<scalar_t>& cov3d_rot,
    const Camera<scalar_t>& camera,
    scalar_t filter_kernel_size)
{
    using vec3_t = glm::vec<3, scalar_t>;
    using vec4_t = glm::vec<4, scalar_t>;
    using mat3_t = glm::mat<3, 3, scalar_t>;
    const auto clamp_to_fov = [&](const vec3_t& t) {
        const auto lim_x = scalar_t(1.3) * camera.tan_fovx * t.z;
        const auto lim_y = scalar_t(1.3) * camera.tan_fovy * t.z;
        return vec3_t { stroke::clamp(t.x, -lim_x, lim_x), stroke::clamp(t.y, -lim_y, lim_y), t.z };
    };

    const auto t = clamp_to_fov(vec3_t(camera.view_matrix * vec4_t(centroid, 1.f))); // clamps the size of the jakobian

    // following zwicker et al. "EWA Splatting"

    const auto l_prime = glm::length(t);
    // // clang-format off
    //    const auto S = mat3_t(camera.focal_x,                                     0,                                  0,
    //                                       0,                        camera.focal_y,                                  0,
    //                                       0,                                     0,                                  1);
    // // clang-format on
    // const auto J = make_jakobian(t, l_prime);
    // const auto SJ = S * J;
    // Avoid matrix multiplication S * J:
    const auto SJ = make_jakobian(t, l_prime, camera.focal_x, camera.focal_y);

    const mat3_t W = mat3_t(camera.view_matrix);
    mat3_t T = SJ * W;

    const auto projected_centroid = project(centroid, camera.view_projection_matrix);
    dgmr::math::Gaussian2d<scalar_t> screen_space_gaussian;
    screen_space_gaussian.centroid = ndc2screen(projected_centroid, camera.fb_width, camera.fb_height);

    const auto cov3d = math::compute_cov(cov3d_scale, cov3d_rot);
    screen_space_gaussian.cov = affine_transform_and_cut(cov3d, T);
    const auto filter_kernel = stroke::Cov2<scalar_t>(filter_kernel_size);
    switch (formulation) {
    case Formulation::Opacity: {
        scalar_t aa_weight_factor = 1;
        cuda::std::tie(screen_space_gaussian.cov, aa_weight_factor) = math::convolve_unnormalised_with_normalised(screen_space_gaussian.cov, filter_kernel);
        screen_space_gaussian.weight = weight * aa_weight_factor;
        break;
    }
    case Formulation::Mass: {
        screen_space_gaussian.cov += filter_kernel;
        const auto detSJ = det(SJ); // det(SJ) == det(S) * det(J)

        const auto norm_2d_factor = stroke::gaussian::norm_factor(screen_space_gaussian.cov);
        screen_space_gaussian.weight = weight * detSJ * norm_2d_factor;
        break;
    }
    case Formulation::Density: {
        screen_space_gaussian.cov += filter_kernel;
        const auto detSJ = det(SJ); // det(SJ) == det(S) * det(J)

        const auto i3 = math::integrate_exponential(cov3d_scale);
        const auto i2 = stroke::gaussian::integrate_exponential(screen_space_gaussian.cov);
        screen_space_gaussian.weight = weight * detSJ * i3 / i2;
        break;
    }
    case Formulation::Ots: {
        screen_space_gaussian.cov += filter_kernel;
        const auto detSJ = det(SJ); // det(SJ) == det(S) * det(J)

        const auto i2prime = math::integrate_exponential(larger2(cov3d_scale));
        const auto i2 = stroke::gaussian::integrate_exponential(screen_space_gaussian.cov);
        screen_space_gaussian.weight = weight * detSJ * i2prime / i2;
        break;
    }
    case Formulation::Ols: {
        screen_space_gaussian.cov += filter_kernel;
        const auto detSJ = det(SJ); // det(SJ) == det(S) * det(J)

        const auto i2prime = math::integrate_exponential(smaller2(cov3d_scale));
        const auto i2 = stroke::gaussian::integrate_exponential(screen_space_gaussian.cov);
        screen_space_gaussian.weight = weight * detSJ * i2prime / i2;
        break;
    }
    }

    return screen_space_gaussian;
}

template <typename scalar_t, unsigned N>
STROKE_DEVICES_INLINE cuda::std::tuple<glm::vec<3, scalar_t>, scalar_t>
integrate_bins(glm::vec<3, scalar_t> current_colour, scalar_t current_transparency, const whack::Array<glm::vec<4, scalar_t>, N>& bins)
{
    for (auto k = 0u; k < bins.size(); ++k) {
        const auto eval_t = bins[k];
        current_colour += glm::vec<3, scalar_t>(eval_t) * current_transparency;
        current_transparency *= stroke::exp(-eval_t.w);
    }
    return cuda::std::make_tuple(current_colour, current_transparency);
}

template <typename scalar_t, unsigned N>
STROKE_DEVICES_INLINE void
sample_gaussian(const scalar_t mass, const glm::vec<3, scalar_t>& rgb, const glm::vec<3, scalar_t>& position, const stroke::Cov3<scalar_t>& inv_cov,
    const stroke::Ray<3, scalar_t>& ray, const whack::Array<scalar_t, N>& bin_borders, whack::Array<glm::vec<4, scalar_t>, N - 1>* bins)
{
    namespace gaussian = stroke::gaussian;
    const auto gaussian1d = gaussian::intersect_with_ray_inv_C(position, inv_cov, ray);
    const auto centroid = gaussian1d.centre;
    const auto variance = gaussian1d.C;
    const auto sd = stroke::sqrt(variance);
    const auto inv_sd = 1 / sd;
    const auto mass_on_ray = gaussian1d.weight * mass;

    if (stroke::isnan(gaussian1d.centre))
        return;
    if (mass_on_ray < 1.1f / 255.f || mass_on_ray > 1'000)
        return;
    if (variance <= 0 || stroke::isnan(variance) || stroke::isnan(mass_on_ray) || mass_on_ray > 100'000)
        return; // todo: shouldn't happen any more after implementing AA?

    const auto mass_in_bins = mass_on_ray * gaussian::integrate_normalised_inv_SD(centroid, inv_sd, { bin_borders[0], bin_borders[bin_borders.size() - 1] });

    if (mass_in_bins < 0.0001f) { // performance critical
        return;
    }

    auto cdf_start = gaussian::cdf_inv_SD(centroid, inv_sd, bin_borders[0]);
    for (auto k = 0u; k < bin_borders.size() - 1; ++k) {
        const auto right = bin_borders[k + 1];
        const auto cdf_end = gaussian::cdf_inv_SD(centroid, inv_sd, right);
        const auto mass_in_bin = stroke::max(scalar_t(0), (cdf_end - cdf_start) * mass_on_ray);
        cdf_start = cdf_end;

        if (mass_in_bin < 0.00001f)
            continue;

        (*bins)[k] += glm::vec<4, scalar_t>(rgb * mass_in_bin, mass_in_bin);
    }
}

} // namespace dgmr::math
