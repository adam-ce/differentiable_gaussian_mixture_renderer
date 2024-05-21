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

#include <stroke/grad/gaussian.h>
#include <stroke/grad/linalg.h>
#include <stroke/grad/quaternions.h>
#include <stroke/grad/scalar_functions.h>
#include <stroke/grad/util.h>
#include <stroke/linalg.h>
#include <whack/array.h>

#include "../math.h"

namespace dgmr::math::grad {

template <typename scalar_t>
STROKE_DEVICES_INLINE glm::vec<3, scalar_t> project(const glm::vec<3, scalar_t>& point, const glm::mat<4, 4, scalar_t>& projection_matrix, const glm::vec<3, scalar_t>& grad)
{
    const auto pp = projection_matrix * glm::vec<4, scalar_t>(point, 1);
    auto grad_pp = glm::vec<4, scalar_t> {};
    stroke::grad::divide_a_by_b(pp.x, pp.w + scalar_t(0.0000001), grad.x).addTo(&grad_pp.x, &grad_pp.w);
    stroke::grad::divide_a_by_b(pp.y, pp.w + scalar_t(0.0000001), grad.y).addTo(&grad_pp.y, &grad_pp.w);
    stroke::grad::divide_a_by_b(pp.z, pp.w + scalar_t(0.0000001), grad.z).addTo(&grad_pp.z, &grad_pp.w);

    const auto [grad_mat, grad_vec] = stroke::grad::matvecmul(projection_matrix, glm::vec<4, scalar_t>(point, 1), grad_pp);
    return grad_vec;
}

template <typename scalar_t>
STROKE_DEVICES_INLINE stroke::grad::TwoGrads<stroke::Cov<3, scalar_t>, glm::mat<3, 3, scalar_t>>
affine_transform_and_cut(const stroke::Cov3<scalar_t>& S, const glm::mat<3, 3, scalar_t>& M, const stroke::Cov2<scalar_t>& grad)
{
    const stroke::Cov3<scalar_t> grad_uncut = {
        grad[0], grad[1], 0,
        grad[2], 0,
        0
    };
    return stroke::grad::affine_transform(S, M, grad_uncut);
}

// kernel cov comes from a normalised gaussian, i.e., it integrates to 1 and has no explicit weight
template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE stroke::Cov<n_dims, scalar_t> convolve_unnormalised_with_normalised(
    const stroke::Cov<n_dims, scalar_t>& cov, const stroke::Cov<n_dims, scalar_t>& kernel_cov,
    const stroke::Cov<n_dims, scalar_t>& grad_filtered_cov, scalar_t grad_weight_factor)
{
    const auto new_cov = cov + kernel_cov;
    const auto det_cov = det(cov);
    const auto det_new_cov = det(new_cov);
    const auto factor = scalar_t(det_cov / det_new_cov);
    // return { stroke::Cov<n_dims, scalar_t>(new_cov), float(stroke::sqrt(stroke::max(scalar_t(0.000025), factor))) };
    // max
    if (factor < scalar_t(0.000025))
        return grad_filtered_cov;
    const auto grad_factor = stroke::grad::sqrt(factor, grad_weight_factor);
    const auto [grad_det_cov, grad_det_new_cov] = stroke::grad::divide_a_by_b(det_cov, det_new_cov, grad_factor);
    auto grad_cov = stroke::grad::det(to_glm(cov), grad_det_cov);
    grad_cov += stroke::grad::det(to_glm(new_cov), grad_det_new_cov);
    return stroke::grad::to_symmetric_gradient(grad_cov) + grad_filtered_cov;
}

template <typename scalar_t>
STROKE_DEVICES_INLINE glm::vec<3, scalar_t>
ndc2screen(const glm::vec<3, scalar_t>&, unsigned width, unsigned height, const glm::vec<2, scalar_t>& grad)
{
    //    const auto ndc2Pix = [](scalar_t v, int S) {
    //        return ((v + scalar_t(1.0)) * S - scalar_t(1.0)) * scalar_t(0.5);
    //    };
    const auto grad_ndc2Pix = [](int S, scalar_t grad) {
        return grad * 0.5 * S;
    };
    //    return { ndc2Pix(point.x, width), ndc2Pix(point.y, height) };
    return { grad_ndc2Pix(width, grad.x), grad_ndc2Pix(height, grad.y), 0 };
}

template <typename scalar_t>
STROKE_DEVICES_INLINE stroke::grad::TwoGrads<glm::vec<3, scalar_t>, scalar_t>
make_jakobian(const glm::vec<3, scalar_t>& t, scalar_t l_prime, const glm::mat<3, 3, scalar_t>& incoming_grad, scalar_t focal_x = 1, scalar_t focal_y = 1)
{
    using vec3_t = glm::vec<3, scalar_t>;

    vec3_t grad_t = {};
    scalar_t grad_l_prime = 0;
    scalar_t grad_tz_tz = 0;

    // focal_x / t.z
    grad_t.z = stroke::grad::divide_a_by_b(focal_x, t.z, incoming_grad[0][0]).right();

    // t.x / l_prime
    stroke::grad::divide_a_by_b(t.x, l_prime, incoming_grad[0][2]).addTo(&grad_t.x, &grad_l_prime);

    // focal_y / t.z
    grad_t.z += stroke::grad::divide_a_by_b(focal_y, t.z, incoming_grad[1][1]).right();

    // t.y / l_prime
    stroke::grad::divide_a_by_b(t.y, l_prime, incoming_grad[1][2]).addTo(&grad_t.y, &grad_l_prime);

    // -(focal_x * t.x) / (t.z * t.z)
    scalar_t tmp = 0;
    stroke::grad::divide_a_by_b(-(focal_x * t.x), t.z * t.z, incoming_grad[2][0]).addTo(&tmp, &grad_tz_tz);
    grad_t.x += -tmp * focal_x;

    // -(focal_y * t.y) / (t.z * t.z)
    tmp = 0;
    stroke::grad::divide_a_by_b(-(focal_y * t.y), t.z * t.z, incoming_grad[2][1]).addTo(&tmp, &grad_tz_tz);
    grad_t.y += -tmp * focal_y;

    // t.z / l_prime
    stroke::grad::divide_a_by_b(t.z, l_prime, incoming_grad[2][2]).addTo(&grad_t.z, &grad_l_prime);

    // (t.z * t.z)
    grad_t.z += grad_tz_tz * 2 * t.z;

    return { grad_t, grad_l_prime };
}

template <typename scalar_t>
STROKE_DEVICES_INLINE stroke::grad::TwoGrads<glm::vec<3, scalar_t>, glm::qua<scalar_t>>
compute_cov(const glm::vec<3, scalar_t>& scale, const glm::qua<scalar_t>& rotation, const stroke::Cov<3, scalar_t> incoming_grad)
{
    using vec_t = glm::vec<3, scalar_t>;
    using mat_t = glm::mat<3, 3, scalar_t>;

    const mat_t R = glm::toMat3(rotation);
    const mat_t S = mat_t(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
    const mat_t RS = R * S;
    const mat_t RS_T = transpose(RS);

    auto [grad_RS, grad_RS_T] = stroke::grad::matmul(RS, RS_T, stroke::grad::to_mat_gradient(incoming_grad));
    grad_RS += transpose(grad_RS_T);
    const auto [grad_R, grad_S] = stroke::grad::matmul(R, S, grad_RS);

    const auto grad_rotation = stroke::grad::toMat3(rotation, grad_R);
    const auto grad_scale = vec_t { grad_S[0][0], grad_S[1][1], grad_S[2][2] };

    return { grad_scale, grad_rotation };
}

template <typename scalar_t>
STROKE_DEVICES_INLINE glm::vec<2, scalar_t> integrate_exponential(const glm::vec<2, scalar_t>& scales, scalar_t grad)
{
    constexpr auto factor = scalar_t(gcem::sqrt(gcem::pow(2 * glm::pi<double>(), double(2))));
    // return factor * scales.x * scales.y * scales.z;
    const auto gf = grad * factor;
    return { gf * scales.y, gf * scales.x };
}

template <typename scalar_t>
STROKE_DEVICES_INLINE glm::vec<3, scalar_t> integrate_exponential(const glm::vec<3, scalar_t>& scales, scalar_t grad)
{
    constexpr auto factor = scalar_t(gcem::sqrt(gcem::pow(2 * glm::pi<double>(), double(3))));
    // return factor * scales.x * scales.y * scales.z;
    const auto gf = grad * factor;
    return { gf * scales.y * scales.z, gf * scales.x * scales.z, gf * scales.x * scales.y };
}

template <typename scalar_t>
STROKE_DEVICES_INLINE glm::vec<3, scalar_t> larger2(const glm::vec<3, scalar_t>& vec, glm::vec<2, scalar_t> grad)
{
    if (vec[0] < vec[1]) {
        if (vec[0] < vec[2])
            return glm::vec<3, scalar_t>(0, grad[0], grad[1]);
    } else {
        if (vec[1] < vec[2])
            return glm::vec<3, scalar_t>(grad[0], 0, grad[1]);
    }

    return glm::vec<3, scalar_t>(grad[0], grad[1], 0);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE glm::vec<3, scalar_t> smaller2(const glm::vec<3, scalar_t>& vec, glm::vec<2, scalar_t> grad)
{
    if (vec[0] > vec[1]) {
        if (vec[0] > vec[2])
            return glm::vec<3, scalar_t>(0, grad[0], grad[1]);
    } else {
        if (vec[1] > vec[2])
            return glm::vec<3, scalar_t>(grad[0], 0, grad[1]);
    }

    return glm::vec<3, scalar_t>(grad[0], grad[1], 0);
}

STROKE_DEVICES_INLINE stroke::grad::TwoGrads<SHs<3>, glm::vec3>
sh_to_colour(const SHs<3>& sh, int deg, const glm::vec3& direction, const glm::vec3& incoming_grad, const glm::bvec3& clamped)
{
    // based on 3D Gaussian Splatting for Real-Time Radiance Field Rendering by Kerbl, Kopanas et al. (2023)
    glm::vec3 dL_dRGB = incoming_grad * glm::vec3(clamped);

    glm::vec3 dRGBdx = {};
    glm::vec3 dRGBdy = {};
    glm::vec3 dRGBdz = {};
    float x = direction.x;
    float y = direction.y;
    float z = direction.z;

    SHs<3> dL_dsh = {};

    float dRGBdsh0 = SH_C0;
    dL_dsh[0] = dRGBdsh0 * dL_dRGB;
    if (deg > 0) {
        float dRGBdsh1 = -SH_C1 * y;
        float dRGBdsh2 = SH_C1 * z;
        float dRGBdsh3 = -SH_C1 * x;
        dL_dsh[1] = dRGBdsh1 * dL_dRGB;
        dL_dsh[2] = dRGBdsh2 * dL_dRGB;
        dL_dsh[3] = dRGBdsh3 * dL_dRGB;

        dRGBdx = -SH_C1 * sh[3];
        dRGBdy = -SH_C1 * sh[1];
        dRGBdz = SH_C1 * sh[2];

        if (deg > 1) {
            float xx = x * x;
            float yy = y * y;
            float zz = z * z;
            float xy = x * y;
            float yz = y * z;
            float xz = x * z;

            float dRGBdsh4 = SH_C2[0] * xy;
            float dRGBdsh5 = SH_C2[1] * yz;
            float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
            float dRGBdsh7 = SH_C2[3] * xz;
            float dRGBdsh8 = SH_C2[4] * (xx - yy);
            dL_dsh[4] = dRGBdsh4 * dL_dRGB;
            dL_dsh[5] = dRGBdsh5 * dL_dRGB;
            dL_dsh[6] = dRGBdsh6 * dL_dRGB;
            dL_dsh[7] = dRGBdsh7 * dL_dRGB;
            dL_dsh[8] = dRGBdsh8 * dL_dRGB;

            dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
            dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
            dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

            if (deg > 2) {
                float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
                float dRGBdsh10 = SH_C3[1] * xy * z;
                float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
                float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
                float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
                float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
                dL_dsh[9] = dRGBdsh9 * dL_dRGB;
                dL_dsh[10] = dRGBdsh10 * dL_dRGB;
                dL_dsh[11] = dRGBdsh11 * dL_dRGB;
                dL_dsh[12] = dRGBdsh12 * dL_dRGB;
                dL_dsh[13] = dRGBdsh13 * dL_dRGB;
                dL_dsh[14] = dRGBdsh14 * dL_dRGB;
                dL_dsh[15] = dRGBdsh15 * dL_dRGB;

                dRGBdx += (SH_C3[0] * sh[9] * 3.f * 2.f * xy + SH_C3[1] * sh[10] * yz + SH_C3[2] * sh[11] * -2.f * xy + SH_C3[3] * sh[12] * -3.f * 2.f * xz + SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) + SH_C3[5] * sh[14] * 2.f * xz + SH_C3[6] * sh[15] * 3.f * (xx - yy));
                dRGBdy += (SH_C3[0] * sh[9] * 3.f * (xx - yy) + SH_C3[1] * sh[10] * xz + SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) + SH_C3[3] * sh[12] * -3.f * 2.f * yz + SH_C3[4] * sh[13] * -2.f * xy + SH_C3[5] * sh[14] * -2.f * yz + SH_C3[6] * sh[15] * -3.f * 2.f * xy);
                dRGBdz += (SH_C3[1] * sh[10] * xy + SH_C3[2] * sh[11] * 4.f * 2.f * yz + SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) + SH_C3[4] * sh[13] * 4.f * 2.f * xz + SH_C3[5] * sh[14] * (xx - yy));
            }
        }
    }

    glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));
    return { dL_dsh, dL_ddir };
}

template <dgmr::Formulation formulation, typename scalar_t>
STROKE_DEVICES_INLINE stroke::grad::TwoGrads<scalar_t, glm::vec<3, scalar_t>>
weight_to_mass(scalar_t weight, const glm::vec<3, scalar_t>& cov3d_scale, const scalar_t& incoming_grad)
{
    switch (formulation) {
    case Formulation::Opacity: {
        assert(false);
        return {};
    }
    case Formulation::Mass: {
        return {
            incoming_grad, {}
        };
    }
    case Formulation::Density: {
        const auto intgrl = dgmr::math::integrate_exponential<scalar_t>(cov3d_scale);
        // return weight * intgrl;
        const auto grad_intgrl = incoming_grad * weight;
        const auto grad_cov3d_scale = grad::integrate_exponential<scalar_t>(cov3d_scale, grad_intgrl);
        return { incoming_grad * intgrl, grad_cov3d_scale };
    }
    case Formulation::Ots: {
        const auto l2 = math::larger2(cov3d_scale);
        const auto i2prime = dgmr::math::integrate_exponential<scalar_t>(l2);
        // return weight * i2prime;
        const auto grad_i2prime = incoming_grad * weight;
        const auto grad_l2 = grad::integrate_exponential<scalar_t>(l2, grad_i2prime);
        const auto grad_cov3d_scale = grad::larger2<scalar_t>(cov3d_scale, grad_l2);
        return { incoming_grad * i2prime, grad_cov3d_scale };
    }
    case Formulation::Ols: {
        // const auto i2prime = math::integrate_exponential(smaller2(cov3d_scale));
        // return weight * i2prime;

        const auto s2 = math::smaller2(cov3d_scale);
        const auto i2prime = dgmr::math::integrate_exponential<scalar_t>(s2);
        // return weight * i2prime;
        const auto grad_i2prime = incoming_grad * weight;
        const auto grad_l2 = grad::integrate_exponential<scalar_t>(s2, grad_i2prime);
        const auto grad_cov3d_scale = grad::smaller2<scalar_t>(cov3d_scale, grad_l2);
        return { incoming_grad * i2prime, grad_cov3d_scale };
    }
    }
    return {};
}

template <dgmr::Formulation formulation, typename scalar_t>
STROKE_DEVICES_INLINE stroke::grad::FourGrads<scalar_t, glm::vec<3, scalar_t>, glm::vec<3, scalar_t>, glm::qua<scalar_t>> splat(
    scalar_t weight,
    const glm::vec<3, scalar_t>& centroid,
    const glm::vec<3, scalar_t>& cov3d_scale,
    const glm::qua<scalar_t>& cov3d_rotation,
    const Gaussian2d<scalar_t>& incoming_grad,
    const Camera<scalar_t>& camera,
    scalar_t filter_kernel_size)
{
    using vec3_t = glm::vec<3, scalar_t>;
    using vec4_t = glm::vec<4, scalar_t>;
    using mat3_t = glm::mat<3, 3, scalar_t>;
    using quat_t = glm::qua<scalar_t>;

    const auto clamp_to_fov = [&](const vec3_t& t) {
        const auto lim_x = 1.3f * camera.tan_fovx * t.z;
        const auto lim_y = 1.3f * camera.tan_fovy * t.z;
        return vec3_t { stroke::clamp(t.x, -lim_x, lim_x), stroke::clamp(t.y, -lim_y, lim_y), t.z };
    };
    const auto grad_clamp_to_fov = [&](const vec3_t& t, const vec3_t& grad) {
        const auto lim_x = 1.3f * camera.tan_fovx * t.z;
        const auto lim_y = 1.3f * camera.tan_fovy * t.z;
        return vec3_t { stroke::grad::clamp(t.x, -lim_x, lim_x, grad.x), stroke::grad::clamp(t.y, -lim_y, lim_y, grad.y), grad.z };
    };

    const vec3_t unclamped_t = vec3_t(camera.view_matrix * vec4_t(centroid, scalar_t(1.)));
    const vec3_t t = clamp_to_fov(unclamped_t); // clamps the size of the jakobian

    const scalar_t l_prime = glm::length(t);
    const auto SJ = dgmr::math::make_jakobian(t, l_prime, camera.focal_x, camera.focal_y);

    const mat3_t W = mat3_t(camera.view_matrix);
    const mat3_t T = SJ * W;

    const auto cov3d = math::compute_cov(cov3d_scale, cov3d_rotation);
    const auto unfiltered_screenspace_cov = dgmr::math::affine_transform_and_cut(cov3d, T);
    const auto projected_centroid = dgmr::math::project(centroid, camera.view_projection_matrix);

    // ================================== compute grad, above is computing forward values that could be cached.
    stroke::Cov2<scalar_t> grad_cov2d = {};
    scalar_t grad_weight3d = {};
    mat3_t grad_SJ = {};
    vec3_t grad_cov3d_scale = {};

    const auto filter_kernel = stroke::Cov2<scalar_t>(filter_kernel_size);

    switch (formulation) {
    case Formulation::Opacity: {
        // scalar_t aa_weight_factor = 1;
        // cuda::std::tie(screen_space_gaussian.cov, aa_weight_factor) = math::convolve_unnormalised_with_normalised(screen_space_gaussian.cov, filter_kernel);
        // screen_space_gaussian.weight = weight * aa_weight_factor;

        const auto aa_weight_factor = cuda::std::get<1>(math::convolve_unnormalised_with_normalised(unfiltered_screenspace_cov, filter_kernel));
        grad_weight3d = incoming_grad.weight * aa_weight_factor;
        const auto grad_aa_weight_factor = incoming_grad.weight * weight;

        grad_cov2d = grad::convolve_unnormalised_with_normalised<2, scalar_t>(unfiltered_screenspace_cov, filter_kernel, incoming_grad.cov, grad_aa_weight_factor);
        break;
    }
    case Formulation::Mass: {
        // screen_space_gaussian.cov += filter_kernel;
        const auto detSJ = det(SJ); // det(SJ) == det(S) * det(J)

        const auto norm_2d_factor = stroke::gaussian::norm_factor(unfiltered_screenspace_cov + filter_kernel);
        // screen_space_gaussian.weight = weight * detSJ * norm_2d_factor;
        grad_weight3d = incoming_grad.weight * detSJ * norm_2d_factor;
        const auto grad_detSJ = incoming_grad.weight * weight * norm_2d_factor;
        const auto grad_norm_2d_factor = incoming_grad.weight * weight * detSJ;

        grad_cov2d = incoming_grad.cov;
        grad_cov2d += stroke::grad::gaussian::norm_factor(unfiltered_screenspace_cov + filter_kernel, grad_norm_2d_factor);

        grad_SJ = stroke::grad::det(SJ, grad_detSJ);

        break;
    }
    case Formulation::Density: {
        // screen_space_gaussian.cov += filter_kernel;
        // const auto detSJ = det(SJ); // det(SJ) == det(S) * det(J)

        // const auto i3 = math::integrate_exponential(cov3d_scale);
        // const auto i2 = stroke::gaussian::integrate_exponential(screen_space_gaussian.cov);
        // screen_space_gaussian.weight = weight * detSJ * i3 / i2;

        const auto detSJ = det(SJ); // det(SJ) == det(S) * det(J)

        const auto i3 = math::integrate_exponential(cov3d_scale);
        const auto i2 = stroke::gaussian::integrate_exponential(unfiltered_screenspace_cov + filter_kernel);

        // screen_space_gaussian.weight = weight * detSJ * i3 / i2;
        grad_weight3d = incoming_grad.weight * detSJ * i3 / i2;
        const auto grad_detSJ = incoming_grad.weight * weight * i3 / i2;
        const auto [grad_i3, grad_i2] = stroke::grad::divide_a_by_b(i3, i2, incoming_grad.weight * weight * detSJ);

        grad_cov2d = incoming_grad.cov;
        grad_cov2d += stroke::grad::gaussian::integrate_exponential(unfiltered_screenspace_cov + filter_kernel, grad_i2);

        grad_cov3d_scale = grad::integrate_exponential(cov3d_scale, grad_i3);

        grad_SJ = stroke::grad::det(SJ, grad_detSJ);

        break;
    }
    case Formulation::Ots: {
        // screen_space_gaussian.cov += filter_kernel;
        // const auto detSJ = det(SJ); // det(SJ) == det(S) * det(J)

        // const auto i2prime = math::integrate_exponential(larger2(cov3d_scale));
        // const auto i2 = stroke::gaussian::integrate_exponential(screen_space_gaussian.cov);
        // screen_space_gaussian.weight = weight * detSJ * i2prime / i2;

        const auto detSJ = det(SJ); // det(SJ) == det(S) * det(J)

        const auto larger2scales = math::larger2(cov3d_scale);
        const auto i2prime = math::integrate_exponential(larger2scales);
        const auto i2 = stroke::gaussian::integrate_exponential(unfiltered_screenspace_cov + filter_kernel);

        // screen_space_gaussian.weight = weight * detSJ * i2prime / i2;
        grad_weight3d = incoming_grad.weight * detSJ * i2prime / i2;
        const auto grad_detSJ = incoming_grad.weight * weight * i2prime / i2;
        const auto [grad_i2prime, grad_i2] = stroke::grad::divide_a_by_b(i2prime, i2, incoming_grad.weight * weight * detSJ);

        grad_cov2d = incoming_grad.cov;
        grad_cov2d += stroke::grad::gaussian::integrate_exponential(unfiltered_screenspace_cov + filter_kernel, grad_i2);

        const auto grad_larger2scales = grad::integrate_exponential(larger2scales, grad_i2prime);
        grad_cov3d_scale = grad::larger2(cov3d_scale, grad_larger2scales);

        grad_SJ = stroke::grad::det(SJ, grad_detSJ);

        break;
    }
    case Formulation::Ols: {
        // screen_space_gaussian.cov += filter_kernel;
        // const auto detSJ = det(SJ); // det(SJ) == det(S) * det(J)

        // const auto i2prime = math::integrate_exponential(larger2(cov3d_scale));
        // const auto i2 = stroke::gaussian::integrate_exponential(screen_space_gaussian.cov);
        // screen_space_gaussian.weight = weight * detSJ * i2prime / i2;

        const auto detSJ = det(SJ); // det(SJ) == det(S) * det(J)

        const auto smaller2scales = math::smaller2(cov3d_scale);
        const auto i2prime = math::integrate_exponential(smaller2scales);
        const auto i2 = stroke::gaussian::integrate_exponential(unfiltered_screenspace_cov + filter_kernel);

        // screen_space_gaussian.weight = weight * detSJ * i2prime / i2;
        grad_weight3d = incoming_grad.weight * detSJ * i2prime / i2;
        const auto grad_detSJ = incoming_grad.weight * weight * i2prime / i2;
        const auto [grad_i2prime, grad_i2] = stroke::grad::divide_a_by_b(i2prime, i2, incoming_grad.weight * weight * detSJ);

        grad_cov2d = incoming_grad.cov;
        grad_cov2d += stroke::grad::gaussian::integrate_exponential(unfiltered_screenspace_cov + filter_kernel, grad_i2);

        const auto grad_larger2scales = grad::integrate_exponential(smaller2scales, grad_i2prime);
        grad_cov3d_scale = grad::smaller2(cov3d_scale, grad_larger2scales);

        grad_SJ = stroke::grad::det(SJ, grad_detSJ);

        break;
    }
    }

    //    screen_space_gaussian.cov = affine_transform_and_cut(cov3D, T);
    const auto [grad_cov3d, grad_T] = grad::affine_transform_and_cut(cov3d, T, grad_cov2d);

    //    screen_space_gaussian.centroid = ndc2screen(projected_centroid, camera.fb_width, camera.fb_height);
    const auto grad_projected_centroid = grad::ndc2screen(projected_centroid, camera.fb_width, camera.fb_height, incoming_grad.centroid);

    //    const auto projected_centroid = project(centroid, camera.view_projection_matrix);
    auto grad_centroid = grad::project(centroid, camera.view_projection_matrix, grad_projected_centroid);

    //    mat3_t T = SJ * W;
    mat3_t grad_W = {};
    stroke::grad::matmul(SJ, W, grad_T).addTo(&grad_SJ, &grad_W);

    //    const auto SJ = dgmr::utils::make_jakobian(t, l_prime, camera.focal_x, camera.focal_y);
    auto [grad_t, grad_l_prime] = grad::make_jakobian(t, l_prime, grad_SJ, camera.focal_x, camera.focal_y);

    //    const scalar_t l_prime = glm::length(t);
    grad_t += stroke::grad::length(t, grad_l_prime);

    //    const vec3_t t = clamp_to_fov(unclamped_t); // clamps the size of the jakobian
    const auto grad_unclamped_t = grad_clamp_to_fov(unclamped_t, grad_t);

    //    const vec3_t unclamped_t = vec3_t(camera.view_matrix * vec4_t(centroid, scalar_t(1.)));
    grad_centroid += vec3_t(transpose(camera.view_matrix) * vec4_t(grad_unclamped_t, scalar_t(0.)));

    quat_t grad_cov3d_rotation = {};
    grad::compute_cov(cov3d_scale, cov3d_rotation, grad_cov3d).addTo(&grad_cov3d_scale, &grad_cov3d_rotation);

    // return screen_space_gaussian;
    return { grad_weight3d, grad_centroid, grad_cov3d_scale, grad_cov3d_rotation };
}

template <typename scalar_t, unsigned N>
STROKE_DEVICES_INLINE cuda::std::tuple<glm::vec<3, scalar_t>, scalar_t, whack::Array<glm::vec<4, scalar_t>, N>>
integrate_bins(glm::vec<3, scalar_t> current_colour, scalar_t current_transparency, scalar_t final_transparency, const whack::Array<glm::vec<4, scalar_t>, N>& bins,
    const glm::vec<3, scalar_t>& grad_colour, scalar_t grad_transparency)
{
    using vec3 = glm::vec<3, scalar_t>;
    whack::Array<glm::vec<4, scalar_t>, N> grad_bins = {};
    for (auto k = 0u; k < bins.size(); ++k) {
        const auto eval_t = bins[k];
        const auto transparency_k = stroke::exp(-eval_t.w);
        grad_bins[k].x += current_transparency * grad_colour.x;
        grad_bins[k].y += current_transparency * grad_colour.y;
        grad_bins[k].z += current_transparency * grad_colour.z;

        const auto c_delta = current_colour - vec3(eval_t);
        current_colour = (c_delta) / transparency_k;

        const auto grad_transparency_k = (final_transparency / transparency_k) * grad_transparency + dot(grad_colour, current_colour) * current_transparency;
        grad_bins[k].w -= grad_transparency_k * stroke::exp(-eval_t.w);
        current_transparency *= transparency_k;
    }
    return { current_colour, current_transparency, grad_bins };
}

template <typename scalar_t, unsigned N>
STROKE_DEVICES_INLINE stroke::grad::FourGrads<scalar_t, glm::vec<3, scalar_t>, glm::vec<3, scalar_t>, stroke::Cov3<scalar_t>>
sample_gaussian(const scalar_t mass, const glm::vec<3, scalar_t>& rgb, const glm::vec<3, scalar_t>& position, const stroke::Cov3<scalar_t>& inv_cov,
    const stroke::Ray<3, scalar_t>& ray, const whack::Array<scalar_t, N>& bin_borders, const whack::Array<glm::vec<4, scalar_t>, N - 1>& grad_incoming)
{
    namespace gaussian = stroke::gaussian;
    using vec3 = glm::vec<3, scalar_t>;
    const auto gaussian1d = gaussian::intersect_with_ray_inv_C(position, inv_cov, ray);
    const auto centroid = gaussian1d.centre;
    const auto variance = gaussian1d.C;
    const auto sd = stroke::sqrt(variance);
    const auto inv_sd = 1 / sd;
    const auto mass_on_ray = gaussian1d.weight * mass;

    if (stroke::isnan(gaussian1d.centre))
        return {};
    if (mass_on_ray < 1.1f / 255.f || mass_on_ray > 1'000)
        return {};
    if (variance <= 0 || stroke::isnan(variance) || stroke::isnan(mass_on_ray) || mass_on_ray > 100'000)
        return {}; // todo: shouldn't happen any more after implementing AA?

    const auto mass_in_bins = mass_on_ray * gaussian::integrate_normalised_inv_SD(centroid, inv_sd, { bin_borders[0], bin_borders[bin_borders.size() - 1] });
    if (mass_in_bins < 0.0001f) { // performance critical
        return {};
    }

    scalar_t grad_centroid = 0;
    scalar_t grad_inv_sd = 0;
    scalar_t grad_mass_on_ray = 0;
    vec3 grad_rgb = {};
    auto cdf_start = gaussian::cdf_inv_SD(centroid, inv_sd, bin_borders[0]);
    for (auto k = 0u; k < bin_borders.size() - 1; ++k) {
        // const auto cdf_start = gaussian::cdf_inv_SD(centroid, inv_sd, bin_borders[k]);
        const auto cdf_end = gaussian::cdf_inv_SD(centroid, inv_sd, bin_borders[k + 1]);
        const auto cdf_delta = cdf_end - cdf_start;
        const auto mass_in_bin = stroke::max(scalar_t(0), cdf_delta * mass_on_ray);
        cdf_start = cdf_end;

        if (mass_in_bin < 0.00001f)
            continue;

        // (*bins)[k] += glm::vec<4, scalar_t>(rgb * mass_in_bin, mass_in_bin);
        grad_rgb += vec3(grad_incoming[k]) * mass_in_bin;
        const auto grad_mass_in_bin = dot(vec3(grad_incoming[k]), rgb) + grad_incoming[k].w;
        if (cdf_delta * mass_on_ray <= 0)
            continue;
        grad_mass_on_ray += grad_mass_in_bin * cdf_delta;
        const auto grad_cdf_delta = grad_mass_in_bin * mass_on_ray;
        const auto grad_cdf_start = -grad_cdf_delta;
        const auto grad_cdf_end = grad_cdf_delta;
        stroke::grad::gaussian::cdf_inv_SD(centroid, inv_sd, bin_borders[k], grad_cdf_start).addTo(&grad_centroid, &grad_inv_sd, nullptr);
        stroke::grad::gaussian::cdf_inv_SD(centroid, inv_sd, bin_borders[k + 1], grad_cdf_end).addTo(&grad_centroid, &grad_inv_sd, nullptr);
    }

    const auto grad_gaussian1d_weight = grad_mass_on_ray * mass;
    const auto grad_mass = grad_mass_on_ray * gaussian1d.weight;
    const auto grad_sd = stroke::grad::divide_a_by_b(scalar_t(1), sd, grad_inv_sd).m_right;
    const auto grad_variance = stroke::grad::sqrt(variance, grad_sd);
    const auto grad_gaussian3d_and_ray = stroke::grad::gaussian::intersect_with_ray_inv_C(position, inv_cov, ray, { grad_gaussian1d_weight, grad_centroid, grad_variance });

    return { grad_mass, grad_rgb, grad_gaussian3d_and_ray.m_left, grad_gaussian3d_and_ray.m_middle }; // grad for ray is unused
}

} // namespace dgmr::math::grad
