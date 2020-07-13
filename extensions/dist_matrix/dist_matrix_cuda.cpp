/*
 * @Author: Haozhe Xie
 * @Date:   2020-07-09 11:45:06
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2020-07-13 09:57:11
 * @Email:  cshzxie@gmail.com
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

void dist_matrix_cuda_forward(torch::Tensor dist_matrix, cudaStream_t stream);

void dist_matrix_forward(torch::Tensor dist_matrix) {
  CHECK_INPUT(dist_matrix);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return dist_matrix_cuda_forward(dist_matrix, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dist_matrix_forward, "Distance Matrix forward (CUDA)");
}
