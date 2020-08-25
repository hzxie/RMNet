/*
 * @Author: Haozhe Xie
 * @Date:   2020-08-24 19:42:33
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2020-08-24 21:08:15
 * @Email:  cshzxie@gmail.com
 *
 * References:
 * - https://github.com/lizhihao6/Forward-Warp
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <vector>

// Define GridSamplerInterpolation
namespace at {
namespace native {
namespace detail {
enum class GridSamplerInterpolation { Bilinear, Nearest };
enum class GridSamplerPadding { Zeros, Border, Reflection };
} // namespace detail
} // namespace native
} // namespace at

using at::native::detail::GridSamplerInterpolation;

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

torch::Tensor
forward_warp_cuda_forward(torch::Tensor im0, torch::Tensor flow,
                          GridSamplerInterpolation interpolation_mode,
                          cudaStream_t stream);

std::vector<torch::Tensor> forward_warp_cuda_backward(
    torch::Tensor grad_output, torch::Tensor im0, torch::Tensor flow,
    GridSamplerInterpolation interpolation_mode, cudaStream_t stream);

torch::Tensor forward_warp_forward(torch::Tensor im0, torch::Tensor flow,
                                   int interpolation_mode) {
  CHECK_INPUT(im0);
  CHECK_INPUT(flow);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return forward_warp_cuda_forward(
      im0, flow, static_cast<GridSamplerInterpolation>(interpolation_mode),
      stream);
}

std::vector<torch::Tensor> forward_warp_backward(torch::Tensor grad_output,
                                                 torch::Tensor im0,
                                                 torch::Tensor flow,
                                                 int interpolation_mode) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(im0);
  CHECK_INPUT(flow);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return forward_warp_cuda_backward(
      grad_output, im0, flow,
      static_cast<GridSamplerInterpolation>(interpolation_mode), stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_warp_forward, "Forward Warp forward (CUDA)");
  m.def("backward", &forward_warp_backward, "Forward Warp backward (CUDA)");
}
