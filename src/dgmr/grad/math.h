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

template <bool orientation_dependent_density, typename scalar_t>
STROKE_DEVICES_INLINE stroke::grad::ThreeGrads<scalar_t, glm::vec<3, scalar_t>, stroke::Cov3<scalar_t>>
splat_with_cache(scalar_t weight, const glm::vec<3, scalar_t>& centroid, const stroke::Cov3<scalar_t>& cov3D, const Gaussian2d<scalar_t>& incoming_grad, const Gaussian2dAndValueCache<scalar_t>& cache, const Camera<scalar_t>& camera, scalar_t filter_kernel_size)
{
    using vec3_t = glm::vec<3, scalar_t>;
    using vec4_t = glm::vec<4, scalar_t>;
    using mat3_t = glm::mat<3, 3, scalar_t>;
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

    const vec3_t& unclamped_t = cache.unclamped_t;
    const vec3_t t = clamp_to_fov(unclamped_t); // clamps the size of the jakobian

    const scalar_t l_prime = glm::length(t);
    const auto J = dgmr::math::make_jakobian(t, l_prime);
    const auto SJ = dgmr::math::make_jakobian(t, l_prime, camera.focal_x, camera.focal_y);

    const mat3_t W = mat3_t(camera.view_matrix);
    const mat3_t& T = cache.T; // SJ * W

    const auto unfiltered_screenspace_cov = dgmr::math::affine_transform_and_cut(cov3D, T);
    const auto projected_centroid = dgmr::math::project(centroid, camera.view_projection_matrix);
    const auto det_J = det(J);
    assert(det_J > 0);

    // ================================== compute grad, above is computing forward values that could be cached.
    auto incoming_cov_grad = incoming_grad.cov;
    auto grad_weight = incoming_grad.weight;
    auto grad_t = vec3_t(0);
    auto grad_l_prime = scalar_t(0);

    const auto filter_kernel = stroke::Cov2<scalar_t>(filter_kernel_size);
    if (orientation_dependent_density) {
        const auto norm_factor = stroke::gaussian::norm_factor(unfiltered_screenspace_cov + filter_kernel);
        //    screen_space_gaussian.weight = weight * camera.focal_x * camera.focal_y * det(J) * norm_factor; // det(S) == camera.focal_x * camera.focal_y
        grad_weight *= camera.focal_x * camera.focal_y * det_J * norm_factor;
        const auto grad_det_J = incoming_grad.weight * weight * camera.focal_x * camera.focal_y * norm_factor;

        //    const auto det_J = det(J);
        const auto grad_J = stroke::grad::det(J, grad_det_J);

        //    const auto J = dgmr::utils::make_jakobian(t, l_prime);
        grad::make_jakobian<scalar_t>(t, l_prime, grad_J, 1, 1).addTo(&grad_t, &grad_l_prime);

        const auto grad_norm_factor = incoming_grad.weight * weight * camera.focal_x * camera.focal_y * det_J;
        incoming_cov_grad += stroke::grad::gaussian::norm_factor(unfiltered_screenspace_cov + filter_kernel, grad_norm_factor);

    } else {
        // screen_space_gaussian.weight *= aa_weight_factor;
        const auto aa_weight_factor = cuda::std::get<1>(math::convolve_unnormalised_with_normalised(unfiltered_screenspace_cov, filter_kernel));
        grad_weight *= aa_weight_factor;
        const auto grad_aa_weight_factor = incoming_grad.weight * weight;

        incoming_cov_grad = grad::convolve_unnormalised_with_normalised<2, scalar_t>(unfiltered_screenspace_cov, filter_kernel, incoming_cov_grad, grad_aa_weight_factor);
    }

    //    screen_space_gaussian.cov = affine_transform_and_cut(cov3D, T);
    const auto [grad_cov3d, grad_T] = grad::affine_transform_and_cut(cov3D, T, incoming_cov_grad);

    //    screen_space_gaussian.centroid = ndc2screen(projected_centroid, camera.fb_width, camera.fb_height);
    const auto grad_projected_centroid = grad::ndc2screen(projected_centroid, camera.fb_width, camera.fb_height, incoming_grad.centroid);

    //    const auto projected_centroid = project(centroid, camera.view_projection_matrix);
    auto grad_centroid = grad::project(centroid, camera.view_projection_matrix, grad_projected_centroid);

    //    mat3_t T = SJ * W;
    const auto [grad_SJ, grad_W] = stroke::grad::matmul(SJ, W, grad_T);

    //    const auto SJ = dgmr::utils::make_jakobian(t, l_prime, camera.focal_x, camera.focal_y);
    grad::make_jakobian(t, l_prime, grad_SJ, camera.focal_x, camera.focal_y).addTo(&grad_t, &grad_l_prime);

    //    const scalar_t l_prime = glm::length(t);
    grad_t += stroke::grad::length(t, grad_l_prime);

    //    const vec3_t t = clamp_to_fov(unclamped_t); // clamps the size of the jakobian
    const auto grad_unclamped_t = grad_clamp_to_fov(unclamped_t, grad_t);

    //    const vec3_t unclamped_t = vec3_t(camera.view_matrix * vec4_t(centroid, scalar_t(1.)));
    grad_centroid += vec3_t(transpose(camera.view_matrix) * vec4_t(grad_unclamped_t, scalar_t(0.)));

    // return screen_space_gaussian;
    return { grad_weight, grad_centroid, grad_cov3d };
}
} // namespace dgmr::math::grad
