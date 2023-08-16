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

#include "forward.h"
#include "sibr_api.h"

using namespace dgmr;

void sibr_api::render(const SibrForwardData& params) {
	// at the time of writing, it was impossible to create a tensorview in a cpp file and then copy it to a cu file.
	// it was also impossible to make GaussianView.cpp a cuda file due to compiler errors
	ForwardData data;
	data.cam_poition = params.cam_poition;
	data.proj_matrix = params.proj_matrix;
	data.view_matrix = params.view_matrix;
	data.centroids = whack::make_tensor_view<const glm::vec3>(params.centroids, whack::Location::Device, params.gaussian_count);
	data.framebuffer = whack::make_tensor_view<float>(params.framebuffer, whack::Location::Device, 3, params.framebuffer_height, params.framebuffer_width);
	forward(data);
}
