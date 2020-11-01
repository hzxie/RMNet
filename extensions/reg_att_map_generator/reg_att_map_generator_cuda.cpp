/*
 * @Author: Haozhe Xie
 * @Date:   2020-08-14 17:20:32
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2020-11-01 14:30:12
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

torch::Tensor reg_att_map_generator_cuda_forward(torch::Tensor mask,
                                                 float prob_threshold,
                                                 int n_pts_threshold,
                                                 int dist_threshold,
                                                 cudaStream_t stream);

torch::Tensor reg_att_map_generator_forward(torch::Tensor mask,
                                            float prob_threshold,
                                            int n_pts_threshold,
                                            int dist_threshold) {
  CHECK_INPUT(mask);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return reg_att_map_generator_cuda_forward(
      mask, prob_threshold, n_pts_threshold, dist_threshold, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &reg_att_map_generator_forward,
        "Regional Attention Map Generator forward (CUDA)");
}