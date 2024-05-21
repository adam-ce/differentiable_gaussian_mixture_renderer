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

#include "math.h"
#include <glm/glm.hpp>

namespace dgmr::util {
// my own:
// STROKE_DEVICES_INLINE glm::vec3 clamp_cov_scales(const glm::vec3& cov_scales)
// {
//     const auto max_value = stroke::min(50.f, glm::compMax(cov_scales));
//     const auto min_value = max_value * 0.01f;
//     return glm::clamp(cov_scales, min_value, max_value);
// }
// from inria:
#define CHECK_CUDA(debug)                                                                                              \
    if (debug) {                                                                                                       \
        auto ret = cudaDeviceSynchronize();                                                                            \
        if (ret != cudaSuccess) {                                                                                      \
            std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
            throw std::runtime_error(cudaGetErrorString(ret));                                                         \
        }                                                                                                              \
    }

STROKE_DEVICES_INLINE void getRect(const glm::vec2& p, const glm::ivec2& ext_rect, glm::uvec2* rect_min, glm::uvec2* rect_max, const dim3& render_grid_dim)
{
    *rect_min = {
        min(render_grid_dim.x, max((int)0, (int)((p.x - ext_rect.x) / render_block_width))),
        min(render_grid_dim.y, max((int)0, (int)((p.y - ext_rect.y) / render_block_height)))
    };
    *rect_max = {
        min(render_grid_dim.x, max((int)0, (int)((p.x + ext_rect.x + render_block_width - 1) / render_block_width))),
        min(render_grid_dim.y, max((int)0, (int)((p.y + ext_rect.y + render_block_height - 1) / render_block_height)))
    };
}

// Helper function to find the next-highest bit of the MSB
// on the CPU.
inline uint32_t getHigherMsb(uint32_t n)
{
    uint32_t msb = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1) {
        step /= 2;
        if (n >> msb)
            msb += step;
        else
            msb -= step;
    }
    if (n >> msb)
        msb++;
    return msb;
}
} // namespace dgmr::util
