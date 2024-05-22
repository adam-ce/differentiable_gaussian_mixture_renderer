/*****************************************************************************
 * Alpine Terrain Renderer
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

#pragma once

#include "types.h"
#include <glm/glm.hpp>
#include <whack/TensorView.h>

namespace dgmr::vol_marcher {

struct ForwardData {
    whack::TensorView<const glm::vec3, 1> gm_centroids;
    whack::TensorView<const SHs<3>, 1> gm_sh_params;
    whack::TensorView<const float, 1> gm_weights;
    whack::TensorView<const glm::vec3, 1> gm_cov_scales;
    whack::TensorView<const glm::quat, 1> gm_cov_rotations;

    glm::mat4 view_matrix = glm::mat4(0);
    glm::mat4 proj_matrix = glm::mat4(0);
    glm::vec3 cam_poition = glm::vec3(0);
    glm::vec3 background = glm::vec3(0);
    int sh_degree = 3;
    float cov_scale_multiplier = 1.f;
    float tan_fovy = 0.f;
    float tan_fovx = 0.f;
    float max_depth = 20.f;
    bool debug = true;
    enum class RenderMode {
        Full,
        Bins,
        Depth
    };
    RenderMode debug_render_mode = RenderMode::Full;
    int debug_render_bin = -1;
};

struct config {
    static constexpr float filter_kernel_SD = 0.55f;
    static constexpr unsigned n_large_steps = 4;
    static constexpr unsigned n_small_steps = 32;
    static constexpr unsigned n_steps_per_gaussian = 32;
    static constexpr float transmission_threshold = 0.005f;
    static constexpr float gaussian_relevance_sigma = 3.f;
    static constexpr Formulation gaussian_mixture_formulation = Formulation::Ots;
};

} // namespace dgmr::vol_marcher
