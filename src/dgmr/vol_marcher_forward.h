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

namespace dgmr::vol_marcher {

struct ForwardCache {
    torch::Tensor rects;
    torch::Tensor rgb;
    torch::Tensor rgb_sh_clamped;
    torch::Tensor depths;
    torch::Tensor points_xy_image;
    torch::Tensor inverse_filtered_cov3d;
    torch::Tensor filtered_masses;
    torch::Tensor tiles_touched;
    torch::Tensor point_offsets;
    torch::Tensor i_ranges;
    torch::Tensor b_point_list;
    torch::Tensor remaining_transparency;
};

ForwardCache forward(vol_marcher::ForwardData& forward_data);

} // namespace dgmr::vol_marcher
