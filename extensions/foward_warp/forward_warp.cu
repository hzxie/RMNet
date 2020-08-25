/*
 * @Author: Haozhe Xie
 * @Date:   2020-08-24 19:42:40
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2020-08-25 09:39:01
 * @Email:  cshzxie@gmail.com
 */

#include <ATen/ATen.h>
#include <torch/torch.h>

#define CUDA_NUM_THREADS 512

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

template <typename scalar_t>
__global__ void
forward_warp_cuda_forward_kernel(scalar_t *im0, scalar_t *flow, scalar_t *im1,
                                 int n_channels, int height, int width,
                                 GridSamplerInterpolation interpolation_mode) {
  int batch_index = blockIdx.x;
  int thread_index = threadIdx.x;
  int stride = blockDim.x;
  int n_pixels = height * width;

  im0 += batch_index * n_channels * n_pixels;
  im1 += batch_index * n_channels * n_pixels;
  flow += batch_index * 2 * n_pixels;

  for (int i = thread_index; i < n_pixels; i += stride) {
    int x0 = i % width;
    int y0 = i / width;
    scalar_t x1 = x0 + flow[i];
    scalar_t y1 = y0 + flow[n_pixels + i];

    // Forward warp: im1[:, x1, y1] = im1[:, x0, y0] (Selected)
    // Backward warp: im1[:, x0, y0] = im1[:, x1, y1]
    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
      int x_floor = static_cast<int>(::floor(x1));
      int y_floor = static_cast<int>(::floor(y1));
      int x_ceil = ::min(x_floor + 1, width - 1);
      int y_ceil = ::min(y_floor + 1, height - 1);

      if (x_floor >= 0 && x_ceil < width && y_floor >= 0 && y_ceil < height) {
        scalar_t nw_k = (x_ceil - x1) * (y_ceil - y1);
        scalar_t ne_k = (x1 - x_floor) * (y_ceil - y1);
        scalar_t sw_k = (x_ceil - x1) * (y1 - y_floor);
        scalar_t se_k = (x1 - x_floor) * (y1 - y_floor);

        for (int j = 0; j < n_channels; ++j) {
          scalar_t *im0_ptr = im0 + j * n_pixels + i;
          scalar_t *im1_ptr = im1 + j * n_pixels + y_ceil * width + x_ceil;

          atomicAdd(im1_ptr, nw_k * (*im0_ptr));
          atomicAdd(im1_ptr + 1, ne_k * (*im0_ptr));
          atomicAdd(im1_ptr + width, sw_k * (*im0_ptr));
          atomicAdd(im1_ptr + width + 1, se_k * (*im0_ptr));
        }
      }
    } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      int x_round = static_cast<int>(::round(x1));
      int y_round = static_cast<int>(::round(y1));
      if (x_round >= 0 && x_round < width && y_round >= 0 && y_round < height) {
        for (int j = 0; j < n_channels; ++j) {
          scalar_t *im0_ptr = im0 + j * n_pixels + i;
          scalar_t *im1_ptr = im1 + j * n_pixels + y_round * width + x_round;
          *im1_ptr = *im0_ptr;
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void forward_warp_cuda_backward_kernel(
    scalar_t *grad_output, scalar_t *im0, scalar_t *flow, scalar_t *im0_grad,
    scalar_t *flow_grad, int n_channels, int height, int width,
    GridSamplerInterpolation interpolation_mode) {
  int batch_index = blockIdx.x;
  int thread_index = threadIdx.x;
  int stride = blockDim.x;
  int n_pixels = height * width;

  grad_output += batch_index * n_channels * n_pixels;
  im0 += batch_index * n_channels * n_pixels;
  im0_grad += batch_index * n_channels * n_pixels;
  flow += batch_index * 2 * n_pixels;
  flow_grad += batch_index * 2 * n_pixels;

  for (int i = thread_index; i < n_pixels; i += stride) {
    int x0 = i % width;
    int y0 = i / width;
    scalar_t x1 = x0 + flow[i];
    scalar_t y1 = y0 + flow[n_pixels + i];

    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
      int x_floor = static_cast<int>(::floor(x1));
      int y_floor = static_cast<int>(::floor(y1));
      int x_ceil = x_floor + 1;
      int y_ceil = y_floor + 1;
      // TODO
    } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      int x_round = static_cast<int>(::round(x1));
      int y_round = static_cast<int>(::round(y1));
      if (x_round >= 0 && x_round < width && y_round >= 0 && y_round < height) {
        for (int j = 0; j < n_channels; ++j) {
          scalar_t *im0_ptr = im0 + j * n_pixels + i;
          scalar_t *im1_ptr = im1 + j * n_pixels + y_round * width + x_round;
          *im0_grad = *grad_output;
        }
      }
    }
  }
}

torch::Tensor
forward_warp_cuda_forward(torch::Tensor im0, torch::Tensor flow,
                          GridSamplerInterpolation interpolation_mode,
                          cudaStream_t stream) {
  torch::Tensor im1 = torch::zeros_like(im0);
  int batch_size = im0.size(0);
  int n_channels = im0.size(1);
  int height = im0.size(2);
  int width = im0.size(3);

  AT_DISPATCH_FLOATING_TYPES(
      im0.scalar_type(), "forward_warp_forward_cuda", ([&] {
        forward_warp_cuda_forward_kernel<scalar_t>
            <<<batch_size, CUDA_NUM_THREADS, 0, stream>>>(
                im0.data_ptr<scalar_t>(), flow.data_ptr<scalar_t>(),
                im1.data_ptr<scalar_t>(), n_channels, height, width,
                interpolation_mode);
      }));

  return im1;
}

std::vector<torch::Tensor> forward_warp_cuda_backward(
    torch::Tensor grad_output, torch::Tensor im0, torch::Tensor flow,
    GridSamplerInterpolation interpolation_mode, cudaStream_t stream) {
  torch::Tensor im0_grad = torch::zeros_like(grad_output);
  torch::Tensor flow_grad = torch::zeros_like(flow);
  int batch_size = im0.size(0);
  int n_channels = im0.size(1);
  int height = im0.size(2);
  int width = im0.size(3);

  AT_DISPATCH_FLOATING_TYPES(
      grad_output.scalar_type(), "forward_warp_backward_cuda", ([&] {
        forward_warp_cuda_backward_kernel<scalar_t>
            <<<batch_size, CUDA_NUM_THREADS, 0, stream>>>(
                grad_output.data_ptr<scalar_t>(), im0.data_ptr<scalar_t>(),
                flow.data_ptr<scalar_t>(), im0_grad.data_ptr<scalar_t>(),
                flow_grad.data_ptr<scalar_t>(), n_channels, height, width,
                interpolation_mode);
      }));

  return {im0_grad, flow_grad};
}
