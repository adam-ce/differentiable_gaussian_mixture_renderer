from torch.utils.cpp_extension import load
import os
import torch.autograd
import types

from .util import *

source_dir = os.path.dirname(__file__)

source_files = [
    source_dir + "/vol_marcher.cu",
    source_dir + "/../src/dgmr/vol_marcher_forward.cu",
    source_dir + "/../src/dgmr/vol_marcher_backward.cu"
]
cpp_binding = load('vol_marcher', source_files,
                   extra_include_paths=extra_include_paths,
                   verbose=True, extra_cflags=cuda_extra_cflags, extra_cuda_cflags=cuda_extra_cuda_cflags, extra_ldflags=["-lpthread"])

class VolMarcher(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                shs: torch.Tensor,
                opacity: torch.Tensor,
                means3D: torch.Tensor,
                scales: torch.Tensor, 
                rotations: torch.Tensor, 
                world_view_transform: torch.Tensor,
                full_proj_transform: torch.Tensor, 
                camera_center: torch.Tensor, 
                bg_color: torch.Tensor, 
                image_width: int,
                image_height: int,
                tanfovx: float,
                tanfovy: float,
                active_sh_degree: int):
        cache = cpp_binding.forward(shs, opacity, means3D, scales, rotations,
                                    world_view_transform, full_proj_transform, camera_center,
                                    bg_color, image_width, image_height, tanfovx, tanfovy,
                                    active_sh_degree)

        ctx.save_for_backward(shs, opacity, means3D, scales, rotations,
                              world_view_transform, full_proj_transform, camera_center,
                              bg_color, torch.tensor(image_width), torch.tensor(image_height), torch.tensor(tanfovx), torch.tensor(tanfovy),
                              torch.tensor(active_sh_degree), *cache)
        return cache[0]

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        shs, opacity, means3D, scales, rotations,\
                world_view_transform, full_proj_transform, camera_center,\
                bg_color, image_width, image_height, tanfovx, tanfovy, active_sh_degree, *cache = ctx.saved_tensors

        grad_sh, grad_weights, grad_positions, grad_scales, grad_rotations = cpp_binding.backward(shs, opacity, means3D, scales, rotations,
                                                      world_view_transform, full_proj_transform, camera_center,
                                                      bg_color, image_width.item(), image_height.item(), tanfovx.item(), tanfovy.item(),
                                                      active_sh_degree.item(),
                                                      grad_output, cache)

        return grad_sh, grad_weights, grad_positions, grad_scales, grad_rotations, None, None, None, None, None, None, None, None, None


apply = VolMarcher.apply
