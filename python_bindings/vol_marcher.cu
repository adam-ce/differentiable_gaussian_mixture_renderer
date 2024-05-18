#include <vector>
#include <algorithm>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "../src/dgmr/vol_marcher_forward.h"
#include <whack/torch_interop.h>

namespace {
glm::vec3 to_vec3(const torch::Tensor& t)
{
    assert(t.dim() == 1);
    assert(t.size(0) >= 3);
    const auto h = t.cpu();
    return { h[0].item<float>(), h[1].item<float>(), h[2].item<float>() };
}
glm::vec4 to_vec4(const torch::Tensor& t)
{
    assert(t.dim() == 1);
    assert(t.size(0) >= 4);
    const auto h = t.cpu();
    return { h[0].item<float>(), h[1].item<float>(), h[2].item<float>(), h[3].item<float>() };
}
glm::mat4 to_mat4(const torch::Tensor& t)
{
    assert(t.dim() == 2);
    assert(t.size(0) == 4);
    assert(t.size(1) == 4);
    const auto h = t.cpu();
    return glm::transpose(glm::mat4(to_vec4(h[0]), to_vec4(h[1]), to_vec4(h[2]), to_vec4(h[3])));
}
} // namespace

std::vector<torch::Tensor> vol_marcher_forward(
    const torch::Tensor& sh_params,
    const torch::Tensor& weights,
    const torch::Tensor& centroids,
    const torch::Tensor& cov_scales,
    const torch::Tensor& cov_rotations,
    const torch::Tensor& view_matrix,
    const torch::Tensor& proj_matrix,
    const torch::Tensor& cam_position,
    const torch::Tensor& background,
    unsigned render_width,
    unsigned render_height,
    float tan_fovx,
    float tan_fovy,
    unsigned sh_degree)
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

    torch::Tensor framebuffer = torch::empty({ 3, render_height, render_width }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    const auto n_gaussians = weights.size(0);

    dgmr::vol_marcher::ForwardData data;
    data.gm_centroids = whack::make_tensor_view<const glm::vec3>(centroids, n_gaussians);
    data.gm_sh_params = whack::make_tensor_view<const dgmr::SHs<3>>(sh_params, n_gaussians);
    data.gm_weights = whack::make_tensor_view<const float>(weights, n_gaussians);
    data.gm_cov_scales = whack::make_tensor_view<const glm::vec3>(cov_scales, n_gaussians);
    data.gm_cov_rotations = whack::make_tensor_view<const glm::quat>(cov_rotations, n_gaussians);
    data.framebuffer = whack::make_tensor_view<float>(framebuffer, render_height, render_width, 3);

    data.view_matrix = to_mat4(view_matrix);
    data.proj_matrix = to_mat4(proj_matrix);
    data.cam_poition = to_vec3(cam_position);
    data.background = to_vec3(background);
    data.sh_degree = sh_degree;
    data.tan_fovx = tan_fovx;
    data.tan_fovy = tan_fovy;

    const auto cache = dgmr::vol_marcher::forward(data);
    return { framebuffer,
        cache.depths_data,
        cache.filtered_masses_data,
        cache.inverse_filtered_cov3d_data,
        cache.point_offsets_data,
        cache.points_xy_image_data,
        cache.rects_data,
        cache.rgb_data,
        cache.rgb_sh_clamped_data,
        cache.tiles_touched_data };
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
