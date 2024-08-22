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

// #define DGMR_TORCH_GRAD_CHECK_CONST_SAMPLES
// #define DGMR_PRINT_G_DENSITIES

namespace dgmr::vol_marcher {

template<typename scalar_t>
struct ForwardData {
    using Vec3 = glm::vec<3, scalar_t>;
    using Mat4 = glm::mat<4, 4, scalar_t>;
    whack::TensorView<const Vec3, 1> gm_centroids;
    whack::TensorView<const SHs<3, scalar_t>, 1> gm_sh_params;
    whack::TensorView<const scalar_t, 1> gm_weights;
    whack::TensorView<const Vec3, 1> gm_cov_scales;
    whack::TensorView<const glm::qua<scalar_t>, 1> gm_cov_rotations;

    Mat4 view_matrix = Mat4(0);
    Mat4 proj_matrix = Mat4(0);
    Vec3 cam_poition = Vec3(0);
    Vec3 background = Vec3(0);
    int sh_degree = 3;
    scalar_t cov_scale_multiplier = 1.f;
    scalar_t tan_fovy = 0.f;
    scalar_t tan_fovx = 0.f;
    scalar_t max_depth = 20.f;
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
    static constexpr unsigned n_large_steps = 16;
    static constexpr unsigned n_small_steps = 128;
    static constexpr unsigned n_steps_per_gaussian = 32;
    static constexpr float transmission_threshold = 0.005f;
    static constexpr float gaussian_relevance_sigma = 3.f;
    static constexpr Formulation gaussian_mixture_formulation = Formulation::Opacity;
};

} // namespace dgmr::vol_marcher
