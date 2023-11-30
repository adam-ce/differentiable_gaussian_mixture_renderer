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

#include <stroke/grad/linalg.h>
#include <stroke/grad/scalar_functions.h>
#include <stroke/grad/util.h>
#include <stroke/linalg.h>

#include "../utils.h"

namespace dgmr::utils::grad {

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
    return stroke::grad::from_mat_gradient(grad_cov) + grad_filtered_cov;
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
    scalar_t tmp_grad_cam_focal_times_t = 0;

    // focal_x / t.z
    grad_t.z = stroke::grad::divide_a_by_b(focal_x, t.z, incoming_grad[0][0]).right();

    // (focal_x * t.x) / l_prime
    stroke::grad::divide_a_by_b(focal_x * t.x, l_prime, incoming_grad[0][2]).addTo(&tmp_grad_cam_focal_times_t, &grad_l_prime);
    grad_t.x += tmp_grad_cam_focal_times_t * focal_x;

    // focal_y / t.z
    grad_t.z += stroke::grad::divide_a_by_b(focal_y, t.z, incoming_grad[1][1]).right();

    // (focal_y * t.y) / l_prime
    tmp_grad_cam_focal_times_t = 0;
    stroke::grad::divide_a_by_b(focal_y * t.y, l_prime, incoming_grad[1][2]).addTo(&tmp_grad_cam_focal_times_t, &grad_l_prime);
    grad_t.y += tmp_grad_cam_focal_times_t * focal_y;

    // -(focal_x * t.x) / (t.z * t.z)
    tmp_grad_cam_focal_times_t = 0;
    stroke::grad::divide_a_by_b(-(focal_x * t.x), t.z * t.z, incoming_grad[2][0]).addTo(&tmp_grad_cam_focal_times_t, &grad_tz_tz);
    grad_t.x += -tmp_grad_cam_focal_times_t * focal_x;

    // -(focal_y * t.y) / (t.z * t.z)
    tmp_grad_cam_focal_times_t = 0;
    stroke::grad::divide_a_by_b(-(focal_y * t.y), t.z * t.z, incoming_grad[2][1]).addTo(&tmp_grad_cam_focal_times_t, &grad_tz_tz);
    grad_t.y += -tmp_grad_cam_focal_times_t * focal_y;

    // t.z / l_prime
    stroke::grad::divide_a_by_b(t.z, l_prime, incoming_grad[2][2]).addTo(&grad_t.z, &grad_l_prime);

    // (t.z * t.z)
    grad_t.z += grad_tz_tz * 2 * t.z;

    return { grad_t, grad_l_prime };
}

template <bool orientation_dependent_density, typename scalar_t>
STROKE_DEVICES_INLINE stroke::grad::ThreeGrads<scalar_t, glm::vec<3, scalar_t>, stroke::Cov3<scalar_t>>
splat(scalar_t weight, const glm::vec<3, scalar_t>& centroid, const stroke::Cov3<scalar_t>& cov3D, const Gaussian2d<scalar_t>& incoming_grad, const Camera<scalar_t>& camera, scalar_t filter_kernel_size)
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

    const vec3_t unclamped_t = vec3_t(camera.view_matrix * vec4_t(centroid, scalar_t(1.)));
    const vec3_t t = clamp_to_fov(unclamped_t); // clamps the size of the jakobian

    const scalar_t l_prime = glm::length(t);
    const auto J = dgmr::utils::make_jakobian(t, l_prime);
    const auto SJ = dgmr::utils::make_jakobian(t, l_prime, camera.focal_x, camera.focal_y);

    const mat3_t W = mat3_t(camera.view_matrix);
    const mat3_t T = SJ * W;

    const auto projected_centroid = dgmr::utils::project(centroid, camera.view_projection_matrix);
    const auto det_J = det(J);
    assert(det_J > 0);

    // ================================== compute grad, above is computing forward values that could be cached.
    auto incoming_cov_grad = incoming_grad.cov;
    auto grad_weight = incoming_grad.weight;
    auto grad_t = vec3_t(0);
    auto grad_l_prime = scalar_t(0);

    if (orientation_dependent_density) {
        //    screen_space_gaussian.weight = weight * camera.focal_x * camera.focal_y * det(J); // det(S) == camera.focal_x * camera.focal_y
        grad_weight *= camera.focal_x * camera.focal_y * det_J;
        const auto grad_det_J = incoming_grad.weight * weight * camera.focal_x * camera.focal_y;

        //    const auto det_J = det(J);
        const auto grad_J = stroke::grad::det(J, grad_det_J);

        //    const auto J = dgmr::utils::make_jakobian(t, l_prime);
        grad::make_jakobian<scalar_t>(t, l_prime, grad_J, 1, 1).addTo(&grad_t, &grad_l_prime);
    } else {
        const auto filter_kernel = stroke::Cov2<scalar_t>(filter_kernel_size);
        // screen_space_gaussian.weight *= aa_weight_factor;
        const auto unfiltered_screenspace_cov = dgmr::utils::affine_transform_and_cut(cov3D, T);
        const auto aa_weight_factor = cuda::std::get<1>(utils::convolve_unnormalised_with_normalised(unfiltered_screenspace_cov, filter_kernel));
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
}
