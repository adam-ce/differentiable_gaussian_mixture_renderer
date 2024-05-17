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
#include "vol_marcher_forward.h"

void dgmr::sibr_interfaces::splat(SplatForwardData& forward_data)
{
    dgmr::splat_forward(forward_data);
}

void dgmr::sibr_interfaces::vol_march(vol_marcher::ForwardData& forward_data)
{
    dgmr::vol_marcher::forward(forward_data);
}
