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

#include "sibr_interfaces.h"

// #include <chrono>

// #include "vol_marcher_backward.h"
#include "vol_marcher_forward.h"

void dgmr::sibr_interfaces::splat(SplatForwardData& forward_data)
{
    dgmr::splat_forward(forward_data);
}

void dgmr::sibr_interfaces::vol_march(whack::TensorView<float, 3> framebuffer, vol_marcher::ForwardData<float>& forward_data)
{

    // const auto start_f = std::chrono::system_clock::now();
    const auto cache = dgmr::vol_marcher::forward(framebuffer, forward_data);
    // const auto end_f = std::chrono::system_clock::now();
    // std::cout << "forward took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_f - start_f).count() << "ms" << std::endl;

    // const auto start_b = std::chrono::system_clock::now();
    // const auto gradients = dgmr::vol_marcher::backward(framebuffer, forward_data, cache, torch::ones({ 3, framebuffer.size(1), framebuffer.size(2) }).cuda());
    // const auto end_b = std::chrono::system_clock::now();
    // std::cout << "backward took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_b - start_b).count() << "ms" << std::endl;
    // std::cout << "forward + backward took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_b - start_f).count() << "ms" << std::endl;
}
