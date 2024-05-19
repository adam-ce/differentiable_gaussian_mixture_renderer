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

#include <torch/types.h>
#include <whack/TensorView.h>

#include "vol_marcher.h"
#include "vol_marcher_forward.h"

namespace dgmr::vol_marcher {
struct Gradients {
    torch::Tensor gm_centroids;
    torch::Tensor gm_sh_params;
    torch::Tensor gm_weights;
    torch::Tensor gm_cov_scales;
    torch::Tensor gm_cov_rotations;
};

Gradients backward(const ForwardData& forward_data, const ForwardCache& forward_cache, const torch::Tensor& grad);

} // namespace dgmr::vol_marcher
