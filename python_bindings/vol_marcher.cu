#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <torch/extension.h>
#include <vector>
#include <chrono>

#include "../src/dgmr/vol_marcher_forward.h"
#include "../src/dgmr/vol_marcher_backward.h"
#include <whack/torch_interop.h>

namespace {
template <typename scalar_t>
glm::vec<3, scalar_t> to_vec3(const torch::Tensor& t)
{
    assert(t.dim() == 1);
    assert(t.size(0) >= 3);
    const auto h = t.cpu();
    return { h[0].item<scalar_t>(), h[1].item<scalar_t>(), h[2].item<scalar_t>() };
}
template <typename scalar_t>
glm::vec<4, scalar_t> to_vec4(const torch::Tensor& t)
{
    assert(t.dim() == 1);
    assert(t.size(0) >= 4);
    const auto h = t.cpu();
    return { h[0].item<scalar_t>(), h[1].item<scalar_t>(), h[2].item<scalar_t>(), h[3].item<scalar_t>() };
}

template <typename scalar_t>
glm::mat<4, 4, scalar_t> to_mat4(const torch::Tensor& t)
{
    assert(t.dim() == 2);
    assert(t.size(0) == 4);
    assert(t.size(1) == 4);
    const auto h = t.cpu();
    return glm::mat<4, 4, scalar_t>(to_vec4<scalar_t>(h[0]), to_vec4<scalar_t>(h[1]), to_vec4<scalar_t>(h[2]), to_vec4<scalar_t>(h[3]));
}

template<typename scalar_t>
std::vector<torch::Tensor> typed_vol_marcher_forward(
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
    scalar_t tan_fovx,
    scalar_t tan_fovy,
    unsigned sh_degree)
{
    using Vec3 = glm::vec<3, scalar_t>;

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

    const auto torch_float_type = (sizeof(scalar_t) == 4) ? torch::kFloat32 : torch::kFloat64;
    const auto cuda_floa_type = torch::TensorOptions().dtype(torch_float_type).device(torch::kCUDA);
    torch::Tensor framebuffer = torch::zeros({ 3, render_height, render_width }, cuda_floa_type);

    const auto n_gaussians = weights.size(0);

    auto framebuffer_view = whack::make_tensor_view<scalar_t>(framebuffer, 3, render_height, render_width);

    dgmr::vol_marcher::ForwardData<scalar_t> data;
    data.gm_centroids = whack::make_tensor_view<const Vec3>(centroids, n_gaussians);
    // python may send only the DC term.
    const auto sh_params_extended = torch::cat({ sh_params, torch::zeros({ n_gaussians, 16 - sh_params.size(1), 3 }, cuda_floa_type) }, 1);
    data.gm_sh_params = whack::make_tensor_view<const dgmr::SHs<3, scalar_t>>(sh_params_extended, n_gaussians);
    data.gm_weights = whack::make_tensor_view<const scalar_t>(weights, n_gaussians);
    data.gm_cov_scales = whack::make_tensor_view<const Vec3>(cov_scales, n_gaussians);
    data.gm_cov_rotations = whack::make_tensor_view<const glm::qua<scalar_t>>(cov_rotations, n_gaussians);
    data.view_matrix = to_mat4<scalar_t>(view_matrix);
    data.proj_matrix = to_mat4<scalar_t>(proj_matrix);
    data.cam_poition = to_vec3<scalar_t>(cam_position);
    data.background = to_vec3<scalar_t>(background);
    data.sh_degree = sh_degree;
    data.tan_fovx = tan_fovx;
    data.tan_fovy = tan_fovy;

    // const auto start = std::chrono::system_clock::now();
    const auto cache = dgmr::vol_marcher::forward(framebuffer_view, data);
    // const auto end = std::chrono::system_clock::now();
    // std::cout << "forward took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
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
        cache.i_ranges,
        cache.b_point_list,
        cache.remaining_transparency,
        };
}

template<typename scalar_t>
std::vector<torch::Tensor> typed_vol_marcher_backward(
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
    using namespace torch::indexing;
    using Vec3 = glm::vec<3, scalar_t>;

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
    const auto torch_float_type = (sizeof(scalar_t) == 4) ? torch::kFloat32 : torch::kFloat64;
    const auto cuda_floa_type = torch::TensorOptions().dtype(torch_float_type).device(torch::kCUDA);
    const auto n_gaussians = weights.size(0);

    dgmr::vol_marcher::ForwardData<scalar_t> data;
    // python may send only the DC term.
    const auto sh_params_extended = torch::cat({ sh_params, torch::ones({ n_gaussians, 16 - sh_params.size(1), 3 }, cuda_floa_type) }, 1);
    data.gm_sh_params = whack::make_tensor_view<const dgmr::SHs<3, scalar_t>>(sh_params_extended, n_gaussians);
    data.gm_weights = whack::make_tensor_view<const scalar_t>(weights, n_gaussians);
    data.gm_centroids = whack::make_tensor_view<const Vec3>(centroids, n_gaussians);
    data.gm_cov_scales = whack::make_tensor_view<const Vec3>(cov_scales, n_gaussians);
    data.gm_cov_rotations = whack::make_tensor_view<const glm::qua<scalar_t>>(cov_rotations, n_gaussians);
    data.view_matrix = to_mat4<scalar_t>(view_matrix);
    data.proj_matrix = to_mat4<scalar_t>(proj_matrix);
    data.cam_poition = to_vec3<scalar_t>(cam_position);
    data.background = to_vec3<scalar_t>(background);
    data.sh_degree = sh_degree;
    data.tan_fovx = tan_fovx;
    data.tan_fovy = tan_fovy;

    const auto framebuffer = whack::make_tensor_view<const scalar_t>(cache_vector.at(0), 3, grad_framebuffer.size(1), grad_framebuffer.size(2));
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
    cache.i_ranges = cache_vector.at(10);
    cache.b_point_list = cache_vector.at(11);
    cache.remaining_transparency = cache_vector.at(12);

    // const auto start = std::chrono::system_clock::now();
    dgmr::vol_marcher::Gradients retval = dgmr::vol_marcher::backward<scalar_t>(framebuffer, data, cache, grad_framebuffer);
    // const auto end = std::chrono::system_clock::now();
    // std::cout << "backward took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    retval.gm_sh_params = retval.gm_sh_params.reshape({ n_gaussians, 16, 3 }).index({ Slice(), Slice(0, sh_params.size(1)), Slice() });
    return { retval.gm_sh_params, retval.gm_weights, retval.gm_centroids, retval.gm_cov_scales, retval.gm_cov_rotations };
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
    assert(sh_params.scalar_type() == weights.scalar_type());
    assert(centroids.scalar_type() == weights.scalar_type());
    assert(cov_scales.scalar_type() == weights.scalar_type());
    assert(cov_rotations.scalar_type() == weights.scalar_type());
    assert(view_matrix.scalar_type() == weights.scalar_type());
    assert(proj_matrix.scalar_type() == weights.scalar_type());
    assert(cam_position.scalar_type() == weights.scalar_type());
    assert(background.scalar_type() == weights.scalar_type());

    if (weights.scalar_type() == torch::ScalarType::Float) {
        return typed_vol_marcher_forward<float>(sh_params, weights, centroids, cov_scales, cov_rotations,
            view_matrix, proj_matrix, cam_position, background,
            render_width, render_height, tan_fovx, tan_fovy, sh_degree);
    }

    if (weights.scalar_type() == torch::ScalarType::Double) {
        return typed_vol_marcher_forward<double>(sh_params, weights, centroids, cov_scales, cov_rotations,
            view_matrix, proj_matrix, cam_position, background,
            render_width, render_height, tan_fovx, tan_fovy, sh_degree);
    }
    std::cerr << "All input vectors must be of the same type (float32 or float64)" << std::endl;
    assert(false);
    return {};
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
    assert(sh_params.scalar_type() == weights.scalar_type());
    assert(centroids.scalar_type() == weights.scalar_type());
    assert(cov_scales.scalar_type() == weights.scalar_type());
    assert(cov_rotations.scalar_type() == weights.scalar_type());
    assert(view_matrix.scalar_type() == weights.scalar_type());
    assert(proj_matrix.scalar_type() == weights.scalar_type());
    assert(cam_position.scalar_type() == weights.scalar_type());
    assert(background.scalar_type() == weights.scalar_type());

    if (weights.scalar_type() == torch::ScalarType::Float)
        return typed_vol_marcher_backward<float>(sh_params, weights, centroids, cov_scales, cov_rotations,
                                                view_matrix, proj_matrix, cam_position, background,
                                                render_width, render_height, tan_fovx, tan_fovy, sh_degree,
                                                grad_framebuffer, cache_vector);

    if (weights.scalar_type() == torch::ScalarType::Double)
        return typed_vol_marcher_backward<double>(sh_params, weights, centroids, cov_scales, cov_rotations,
                                                view_matrix, proj_matrix, cam_position, background,
                                                render_width, render_height, tan_fovx, tan_fovy, sh_degree,
                                                grad_framebuffer, cache_vector);
    std::cerr << "All input vectors must be of the same type (float32 or float64)" << std::endl;
    assert(false);
    return {};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &vol_marcher_forward, "vol_marcher_forward");
    m.def("backward", &vol_marcher_backward, "vol_marcher_backward");
}
