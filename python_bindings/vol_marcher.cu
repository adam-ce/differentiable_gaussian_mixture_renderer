#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <torch/extension.h>
#include <vector>

#include "../src/dgmr/vol_marcher_forward.h"
#include "../src/dgmr/vol_marcher_backward.h"
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
    return glm::mat4(to_vec4(h[0]), to_vec4(h[1]), to_vec4(h[2]), to_vec4(h[3]));
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
    assert(weights.is_contiguous());
    assert(centroids.is_contiguous());
    assert(cov_scales.is_contiguous());
    assert(cov_rotations.is_contiguous());
    assert(view_matrix.is_contiguous());
    assert(proj_matrix.is_contiguous());
    assert(cam_position.is_contiguous());
    if (sh_params.is_cuda()) {
        assert(device_of(sh_params).has_value());
        device_guard.set_device(device_of(sh_params).value());
    }

    const auto cuda_floa_type = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor framebuffer = torch::zeros({ 3, render_height, render_width }, cuda_floa_type);

    const auto n_gaussians = weights.size(0);

    dgmr::vol_marcher::ForwardData data;
    data.gm_centroids = whack::make_tensor_view<const glm::vec3>(centroids, n_gaussians);
    // python may send only the DC term.
    const auto sh_params_extended = torch::cat({ sh_params, torch::ones({ n_gaussians, 16 - sh_params.size(1), 3 }, cuda_floa_type) }, 1);
    data.gm_sh_params = whack::make_tensor_view<const dgmr::SHs<3>>(sh_params_extended, n_gaussians);
    data.gm_weights = whack::make_tensor_view<const float>(weights, n_gaussians);
    data.gm_cov_scales = whack::make_tensor_view<const glm::vec3>(cov_scales, n_gaussians);
    data.gm_cov_rotations = whack::make_tensor_view<const glm::quat>(cov_rotations, n_gaussians);
    data.framebuffer = whack::make_tensor_view<float>(framebuffer, 3, render_height, render_width);
    data.view_matrix = to_mat4(view_matrix);
    data.proj_matrix = to_mat4(proj_matrix);
    data.cam_poition = to_vec3(cam_position);
    data.background = to_vec3(background);
    data.sh_degree = sh_degree;
    data.tan_fovx = tan_fovx;
    data.tan_fovy = tan_fovy;

    const auto cache = dgmr::vol_marcher::forward(data);
    return { framebuffer,
        cache.rects,
        cache.rgb,
        cache.rgb_sh_clamped,
        cache.depths,
        cache.points_xy_image,
        cache.inverse_filtered_cov3d,
        cache.filtered_masses,
        cache.tiles_touched,
        cache.point_offsets,
        cache.remaining_transparency,
        cache.distance_marched
        };
}

std::vector<torch::Tensor> vol_marcher_backward(
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
    unsigned sh_degree,
    const torch::Tensor& grad_framebuffer,
    const std::vector<torch::Tensor>& cache_vector)
{
    at::cuda::OptionalCUDAGuard device_guard;
    assert(device_of(sh_params) == device_of(weights));
    assert(device_of(sh_params) == device_of(centroids));
    assert(device_of(sh_params) == device_of(cov_scales));
    assert(device_of(sh_params) == device_of(cov_rotations));
    assert(device_of(sh_params) == device_of(view_matrix));
    assert(device_of(sh_params) == device_of(proj_matrix));
    assert(device_of(sh_params) == device_of(cam_position));
    assert(device_of(sh_params) == device_of(grad_framebuffer));
    assert(weights.is_contiguous());
    assert(centroids.is_contiguous());
    assert(cov_scales.is_contiguous());
    assert(cov_rotations.is_contiguous());
    assert(view_matrix.is_contiguous());
    assert(proj_matrix.is_contiguous());
    assert(cam_position.is_contiguous());
    assert(grad_framebuffer.is_contiguous());
    for (const auto& t : cache_vector) {
        assert(t.is_contiguous());
    }
    if (sh_params.is_cuda()) {
        assert(device_of(sh_params).has_value());
        device_guard.set_device(device_of(sh_params).value());
    }
    const auto cuda_floa_type = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto n_gaussians = weights.size(0);

    dgmr::vol_marcher::ForwardData data;
    // python may send only the DC term.
    const auto sh_params_extended = torch::cat({ sh_params, torch::ones({ n_gaussians, 16 - sh_params.size(1), 3 }, cuda_floa_type) }, 1);
    data.gm_sh_params = whack::make_tensor_view<const dgmr::SHs<3>>(sh_params_extended, n_gaussians);
    data.gm_weights = whack::make_tensor_view<const float>(weights, n_gaussians);
    data.gm_centroids = whack::make_tensor_view<const glm::vec3>(centroids, n_gaussians);
    data.gm_cov_scales = whack::make_tensor_view<const glm::vec3>(cov_scales, n_gaussians);
    data.gm_cov_rotations = whack::make_tensor_view<const glm::quat>(cov_rotations, n_gaussians);
    data.view_matrix = to_mat4(view_matrix);
    data.proj_matrix = to_mat4(proj_matrix);
    data.cam_poition = to_vec3(cam_position);
    data.background = to_vec3(background);
    data.sh_degree = sh_degree;
    data.tan_fovx = tan_fovx;
    data.tan_fovy = tan_fovy;

    dgmr::vol_marcher::ForwardCache cache;
    cache.rects = cache_vector.at(1);
    cache.rgb = cache_vector.at(2);
    cache.rgb_sh_clamped = cache_vector.at(3);
    cache.depths = cache_vector.at(4);
    cache.points_xy_image = cache_vector.at(5);
    cache.inverse_filtered_cov3d = cache_vector.at(6);
    cache.filtered_masses = cache_vector.at(7);
    cache.tiles_touched = cache_vector.at(8);
    cache.point_offsets = cache_vector.at(9);
    cache.remaining_transparency = cache_vector.at(10);
    cache.distance_marched = cache_vector.at(11);

    const dgmr::vol_marcher::Gradients retval = dgmr::vol_marcher::backward(data, cache, grad_framebuffer);

    return { retval.gm_sh_params, retval.gm_weights, retval.gm_centroids, retval.gm_cov_scales, retval.gm_cov_rotations };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &vol_marcher_forward, "vol_marcher_forward");
    m.def("backward", &vol_marcher_backward, "vol_marcher_backward");
}
