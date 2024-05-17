#include <vector>
#include <algorithm>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "../src/dgmr/vol_marcher_forward.h"

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
    unsigned sh_max_coeffs)
{
    at::cuda::OptionalCUDAGuard device_guard;
    assert(device_of(sh_params) == device_of(weights));
    assert(device_of(sh_params) == device_of(centroids));
    assert(device_of(sh_params) == device_of(cov_scales));
    assert(device_of(sh_params) == device_of(cov_rotations));
    assert(device_of(sh_params) == device_of(view_matrix));
    assert(device_of(sh_params) == device_of(proj_matrix));
    assert(device_of(sh_params) == device_of(cam_position));
    if (sh_params.is_cuda()) {
        assert(device_of(sh_params).has_value());
        device_guard.set_device(device_of(sh_params).value());
    }

    dgmr::vol_marcher::ForwardData data;

    dgmr::vol_marcher::forward(data);
}

// std::pair<torch::Tensor, torch::Tensor> convolution_fitting_backward(const torch::Tensor& grad,
//                                                                      const torch::Tensor& data, const torch::Tensor& kernels, int n_components_fitting,
//                                                                      const torch::Tensor& fitting, const torch::Tensor& cached_pos_covs, const torch::Tensor& nodeobjs, const torch::Tensor& fitting_subtrees) {
//     at::cuda::OptionalCUDAGuard device_guard;
//     if (grad.is_cuda()) {
//         assert (device_of(grad).has_value());
//         device_guard.set_device(device_of(grad).value());
//     }

//     convolution_fitting::Config config = {};
//     config.n_components_fitting = unsigned(n_components_fitting);
//     const auto retval = convolution_fitting::backward_impl(grad, data, kernels, convolution_fitting::ForwardOutput{fitting, cached_pos_covs, nodeobjs, fitting_subtrees}, config);
// #ifdef GPE_PROFILER_BUILD
//     cudaDeviceSynchronize();
// #endif
//     return retval;
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &vol_marcher_forward, "vol_marcher_forward");
    // m.def("backward", &convolution_fitting_backward, "convolution_fitting_backward");
}
