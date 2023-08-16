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

#include <whack/kernel.h>

dgmr::Statistics dgmr::forward(const ForwardData& datad) {
	constexpr unsigned pixel_block_height = 8;
	constexpr unsigned pixel_block_width = 16;
	auto data = datad;
	const dim3 block_dim = { pixel_block_width, pixel_block_height };
	const dim3 grid_dim = whack::grid_dim_from_total_size({ data.framebuffer.size<2>(), data.framebuffer.size<1>() }, block_dim);
	whack::start_parallel(
		whack::Location::Device, grid_dim, block_dim, WHACK_KERNEL(data) {
			WHACK_UNUSED(whack_gridDim);
			const unsigned p_x = whack_blockIdx.x * whack_blockDim.x + whack_threadIdx.x;
			const unsigned p_y = whack_blockIdx.y * whack_blockDim.y + whack_threadIdx.y;
			const auto fb_width = data.framebuffer.size<2>();
			const auto fb_height = data.framebuffer.size<1>();
			if (p_x >= fb_width || p_y >= fb_height)
				return;
			data.framebuffer(0, p_y, p_x) = float(p_x) / fb_width;
			data.framebuffer(1, p_y, p_x) = float(p_y) / fb_height;
			data.framebuffer(2, p_y, p_x) = 1.0;
		});

	return { 0 };
}
