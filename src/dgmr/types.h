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

#pragma once

#include <stroke/linalg.h>
#include <whack/array.h>

namespace dgmr {

enum class Formulation : int {
    Opacity, // original
    Mass,
    Density,
    Ots, // Opacity thin side, view dependent, but scaled between 0 and 1 for the shortest axis of the gaussian
    Ols // Opacity long side, view dependent, but scaled between 0 and 1 for the longest axis of the gaussian
};

template <int D, typename scalar_t>
using SHs = whack::Array<glm::vec<3, scalar_t>, (D + 1) * (D + 1)>;
}
