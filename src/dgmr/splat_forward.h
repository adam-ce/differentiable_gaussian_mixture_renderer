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
#include <stroke/matrix.h>
#include <whack/TensorView.h>
#include <whack/array.h>

namespace dgmr {

struct Statistics {
    unsigned n_rendered = 0;
};

template <int D>
using SHs = whack::Array<glm::vec3, (D + 1) * (D + 1)>;

struct ConicAndOpacity {
    stroke::SymmetricMat<2, float> conic;
    float opacity;
};
static_assert(sizeof(ConicAndOpacity) == 4 * 4);

struct SplatForwardData {
    whack::TensorView<const glm::vec3, 1> gm_centroids;
    whack::TensorView<const SHs<3>, 1> gm_sh_params;
    whack::TensorView<const float, 1> gm_weights;
    whack::TensorView<const glm::vec3, 1> gm_cov_scales;
    whack::TensorView<const glm::quat, 1> gm_cov_rotations;

    whack::TensorView<float, 3> framebuffer;
    glm::mat4 view_matrix = glm::mat4(0);
    glm::mat4 proj_matrix = glm::mat4(0);
    glm::vec3 cam_poition = glm::vec3(0);
    glm::vec3 background = glm::vec3(0);
    int sh_degree = 3;
    int sh_max_coeffs = 16;
    float cov_scale_multiplier = 1.f;
    float tan_fovy = 0.f;
    float tan_fovx = 0.f;
    bool debug = true;
};

Statistics splat_forward(SplatForwardData& forward_data);
} // namespace dgmr
