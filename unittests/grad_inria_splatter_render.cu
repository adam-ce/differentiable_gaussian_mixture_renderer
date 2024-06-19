/*****************************************************************************
 * Differentiable Gaussian Mixture Renderer
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

#include <stroke/unittest/gradcheck.h> // must go first (so that pretty printers work)

#include <catch2/catch_test_macros.hpp>
#include "unit_test_utils.h"

#define DGR_USE_EXP_AND_SELF_SHADOWING

namespace {
using scalar_t = double;

template <unsigned n_gaussians>
std::pair<glm::dvec3, scalar_t> forward(
    const glm::dvec2& pixf,
    const std::array<glm::dvec2, n_gaussians> collected_xy, const std::array<glm::dvec4, n_gaussians>& collected_conic_opacity, const std::array<glm::dvec3, n_gaussians>& features,
    const glm::dvec3& bg_color)
{
    scalar_t current_transparency = 1.0;
    glm::vec<3, scalar_t> current_colour = {};
    for (unsigned j = 0u; j < n_gaussians; ++j) {
        const auto xy = collected_xy[j];
        const auto d = xy - pixf;
        const auto con_o = collected_conic_opacity[j];
        scalar_t power = scalar_t(-0.5) * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
        if (power > 0.0f)
            continue;
        const scalar_t G = exp(power);
        const auto pure_eval = con_o.w * G;
#ifdef DGR_USE_EXP_AND_SELF_SHADOWING
        scalar_t effective_eval = (scalar_t(1) - stroke::exp(-pure_eval));
        const auto transparency_k = stroke::exp(-pure_eval);
#else
        scalar_t effective_eval = stroke::min(scalar_t(0.99), pure_eval);
        const auto transparency_k = (1 - effective_eval);
#endif

        if (effective_eval < scalar_t(1.0 / 255.0))
            continue;

        const auto effective_colour = features[j] * effective_eval;
        current_colour += effective_colour * current_transparency;

        current_transparency *= transparency_k;
        if (current_transparency <= 0.0001)
            break;
    }
    return { current_colour + current_transparency * bg_color, current_transparency };
}

template <unsigned n_gaussians>
std::tuple<std::array<glm::dvec2, n_gaussians>, std::array<glm::dvec4, n_gaussians>, std::array<glm::dvec3, n_gaussians>> backward(
    const glm::dvec2& pixf,
    const std::array<glm::dvec2, n_gaussians> collected_xy, const std::array<glm::dvec4, n_gaussians>& collected_conic_opacity, const std::array<glm::dvec3, n_gaussians>& features,
    const glm::dvec3& bg_color,
    const std::pair<glm::dvec3, scalar_t>& forward_out,
    const glm::dvec3& incoming_grad)
{
    const auto final_colour = forward_out.first;
    const auto final_transparency = forward_out.second;
    const glm::dvec3& grad_colour = incoming_grad;
    const auto grad_transparency = dot(grad_colour, bg_color);

    scalar_t current_transparency = 1;
    glm::dvec3 current_colour = final_colour - final_transparency * bg_color;
    std::array<glm::dvec2, n_gaussians> grad_xy = {};
    std::array<glm::dvec4, n_gaussians> grad_conics_opacities = {};
    std::array<glm::dvec3, n_gaussians> grad_features = {};


    for (unsigned j = 0u; j < n_gaussians; ++j) {
        const auto xy = collected_xy[j];
        const auto d = xy - pixf;
        const auto con_o = collected_conic_opacity[j];
        scalar_t power = scalar_t(-0.5) * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
        if (power > 0.0f)
            continue;
        const scalar_t G = exp(power);

        const auto pure_eval = con_o.w * G;
#ifdef DGR_USE_EXP_AND_SELF_SHADOWING
        const auto exp_neg_eval = stroke::exp(-pure_eval);
        const auto effective_eval = 1.f - exp_neg_eval;
        const auto transparency_k = exp_neg_eval;
#else
        scalar_t effective_eval = stroke::min(scalar_t(0.99), pure_eval);
        const auto transparency_k = (1 - effective_eval);
#endif

        if (effective_eval < scalar_t(1.0 / 255.0))
            continue;

        const auto one_over_transparency_k = std::min(scalar_t(255.), 1 / transparency_k);

        const auto effective_colour = features[j] * effective_eval;

        const auto c_delta = current_colour - effective_colour;
        current_colour = c_delta * one_over_transparency_k;

        const auto grad_transparency_k = final_transparency * one_over_transparency_k * grad_transparency + dot(grad_colour, current_colour) * current_transparency;

        const auto grad_effective_colour = grad_colour * current_transparency;

#ifdef DGR_USE_EXP_AND_SELF_SHADOWING
        const auto grad_effective_eval = dot(grad_effective_colour, features[j]);
        const auto grad_exp_neg_eval = grad_transparency_k - grad_effective_eval;
        const auto grad_pure_eval = - grad_exp_neg_eval * stroke::exp(-pure_eval);
#else
        const auto grad_effective_eval = dot(grad_effective_colour, features[j]) - grad_transparency_k;
        const auto grad_pure_eval = (((con_o.w * G) >= scalar_t(0.99)) ? scalar_t(0) : grad_effective_eval);
#endif


        grad_features[j] += grad_effective_colour * effective_eval;

        // Helpful reusable temporary variables
        const scalar_t dL_dG = con_o.w * grad_pure_eval;
        const scalar_t gdx = G * d.x;
        const scalar_t gdy = G * d.y;
        const scalar_t dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
        const scalar_t dG_ddely = -gdy * con_o.z - gdx * con_o.y;

        // Update gradients w.r.t. 2D mean position of the Gaussian
        grad_xy[j].x += dL_dG * dG_ddelx;
        grad_xy[j].y += dL_dG * dG_ddely;

        // Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
        grad_conics_opacities[j].x += -0.5f * gdx * d.x * dL_dG;
        grad_conics_opacities[j].y += -1.0f * gdx * d.y * dL_dG;
        grad_conics_opacities[j].z += -0.5f * gdy * d.y * dL_dG;    // carefull, inria code uses x, y, and w

        // Update gradients w.r.t. opacity of the Gaussian
        grad_conics_opacities[j].w += G * grad_pure_eval;                // carefull, inria code puts the opacity gradient in an extra buffer

        current_transparency *= transparency_k;
        if (current_transparency <= 0.0001)
            break;
    }
    return {grad_xy, grad_conics_opacities, grad_features};
}

}

TEST_CASE("inria splatter render kernel grad")
{

    // const glm::dvec2& pixf,
    // const std::array<glm::dvec2, n_gaussians> collected_xy, const std::array<glm::dvec4, n_gaussians>& collected_conic_opacity, const std::array<glm::dvec3, n_gaussians>& features,
    // const glm::dvec3& bg_color
    using scalar_t = double;
    constexpr auto n_gaussians = 8u;


    whack::random::HostGenerator<scalar_t> rnd;

    const auto rnd_positions = [&]() {
        std::array<glm::dvec2, n_gaussians> retval;
        for (auto i = 0u; i < n_gaussians; ++i) {
            retval[i] = glm::dvec2(rnd.uniform(), rnd.uniform());
        }
        return retval;
    };
    const auto rnd_conics_opacities = [&]() {
        std::array<glm::dvec4, n_gaussians> retval;
        for (auto i = 0u; i < n_gaussians; ++i) {
            retval[i] = glm::dvec4(rnd.uniform() + 1, rnd.uniform(), rnd.uniform() + 1, rnd.uniform());
        }
        return retval;
    };
    const auto rnd_features = [&]() {
        std::array<glm::dvec3, n_gaussians> retval;
        for (auto i = 0u; i < n_gaussians; ++i) {
            retval[i] = glm::dvec3(rnd.uniform(), rnd.uniform(), rnd.uniform());
        }
        return retval;
    };


    for (int i = 0; i < 10; ++i) {
        const auto pixf = glm::dvec2(rnd.uniform(), rnd.uniform());
        const auto bg_color = glm::dvec3(rnd.uniform(), rnd.uniform(), rnd.uniform());

        const auto fun = [pixf, bg_color](const whack::Tensor<scalar_t, 1>& input) {
            const auto [collected_xy, collected_conic_opacity, features] =
                stroke::extract<std::array<glm::dvec2, n_gaussians>, std::array<glm::dvec4, n_gaussians>, std::array<glm::dvec3, n_gaussians>>(input);
            const auto output = forward<n_gaussians>(pixf, collected_xy, collected_conic_opacity, features, bg_color);
            return stroke::pack_tensor<scalar_t>(output.first);
        };

        const auto fun_grad = [pixf, bg_color](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
            const auto [collected_xy, collected_conic_opacity, features] =
                stroke::extract<std::array<glm::dvec2, n_gaussians>, std::array<glm::dvec4, n_gaussians>, std::array<glm::dvec3, n_gaussians>>(input);
            const auto forward_out = forward<n_gaussians>(pixf, collected_xy, collected_conic_opacity, features, bg_color);

            const auto incoming_grad_colour = stroke::extract<glm::dvec3>(grad_output);

            const auto [grad_xy, grad_conics_opas, grad_features] = backward<n_gaussians>(pixf, collected_xy, collected_conic_opacity, features, bg_color,
                forward_out,
                incoming_grad_colour
                );

            return stroke::pack_tensor<scalar_t>(grad_xy, grad_conics_opas, grad_features);
        };

        const auto test_data = stroke::pack_tensor<scalar_t>(
            rnd_positions(),
            rnd_conics_opacities(),
            rnd_features()
            );
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
}
