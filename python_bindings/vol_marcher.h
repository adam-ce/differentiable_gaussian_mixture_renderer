#pragma once

#include <vector>

#include <torch/types.h>

std::vector<torch::Tensor> vol_marcher_forward(
    const torch::Tensor& sh_params,
    const torch::Tensor& weights,
    const torch::Tensor& centroids,
    const torch::Tensor& cov_scales,
    const torch::Tensor& cov_rotations,
    const torch::Tensor& view_matrix,
    const torch::Tensor& proj_matrix,
    const torch::Tensor& cam_position,
    unsigned render_width,
    unsigned render_height,
    float tan_fovx,
    float tan_fovy,
    unsigned sh_degree,
    unsigned sh_max_coeffs);

// std::pair<at::Tensor, at::Tensor> vol_marcher_backward(const torch::Tensor& grad,
//     const torch::Tensor& data, const torch::Tensor& kernels, int n_components_fitting,
//     const torch::Tensor& fitting, const torch::Tensor& cached_pos_covs, const torch::Tensor& nodeobjs, const torch::Tensor& fitting_subtrees);
