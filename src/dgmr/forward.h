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
#include <whack/TensorView.h>

namespace dgmr {

struct Statistics {
	unsigned n_rendered = 0;
};

struct ForwardData {
	whack::TensorView<const glm::vec3, 1> centroids;
	whack::TensorView<float, 3> framebuffer;
	glm::mat4 view_matrix = glm::mat4(0);
	glm::mat4 proj_matrix = glm::mat4(0);
	glm::vec3 cam_poition = glm::vec3(0);
};

Statistics forward(const ForwardData& forward_data);

}
