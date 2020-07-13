/*
 * @Author: Haozhe Xie
 * @Date:   2020-07-09 11:45:12
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2020-07-13 14:27:13
 * @Email:  cshzxie@gmail.com
 */

#include <cstdio>
#include <cstdlib>
#include <torch/extension.h>

#define CUDA_NUM_THREADS 512

__global__ void dist_matrix_cuda_kernel(float *dist_matrix, int w, int h) {
  int n_pixels = h * w;
  dist_matrix += (blockIdx.x + blockIdx.y * gridDim.x) * n_pixels;

  for (int i = threadIdx.x; i < n_pixels; i += blockDim.x) {
    int r_idx = i / w;
    int c_idx = i % w;

    int diff_x = r_idx - blockIdx.y;
    int diff_y = c_idx - blockIdx.x;
    dist_matrix[i] = diff_x * diff_x + diff_y * diff_y;
  }
}

void dist_matrix_cuda_forward(torch::Tensor dist_matrix, cudaStream_t stream) {
  int h = dist_matrix.size(0);
  int w = dist_matrix.size(1);

  dist_matrix_cuda_kernel<<<dim3(w, h), CUDA_NUM_THREADS, 0, stream>>>(
      dist_matrix.data_ptr<float>(), w, h);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in dist_matrix_cuda_forward: %s\n", cudaGetErrorString(err));
  }
}
