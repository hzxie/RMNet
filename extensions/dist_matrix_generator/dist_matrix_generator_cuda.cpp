/*
 * @Author: Haozhe Xie
 * @Date:   2020-08-14 17:20:32
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2020-09-18 14:58:40
 * @Email:  cshzxie@gmail.com
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

torch::Tensor dist_matrix_generator_cuda_forward(torch::Tensor mask,
                                                 torch::Tensor occ_mask,
                                                 float prob_threshold,
                                                 float occ_dist_factor,
                                                 cudaStream_t stream);

torch::Tensor dist_matrix_generator_forward(torch::Tensor mask,
                                            torch::Tensor occ_mask,
                                            float prob_threshold,
                                            float occ_dist_factor) {
  CHECK_INPUT(mask);
  CHECK_INPUT(occ_mask);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return dist_matrix_generator_cuda_forward(mask, occ_mask, prob_threshold,
                                            occ_dist_factor, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dist_matrix_generator_forward,
        "Distance Matrix Generator forward (CUDA)");
}