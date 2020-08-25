/*
 * @Author: Haozhe Xie
 * @Date:   2020-08-14 17:22:12
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2020-08-24 21:06:27
 * @Email:  cshzxie@gmail.com
 */

#include <iostream>
#include <torch/types.h>

#define CUDA_NUM_THREADS 512

__global__ void dist_matrix_forward_kernel(int n_objects, int height, int width,
                                           const float prob_threshold,
                                           const float *__restrict__ mask,
                                           int *__restrict__ bboxes,
                                           int *__restrict__ n_points,
                                           float *__restrict__ dist_matrix) {
  int batch_index = blockIdx.x;
  int thread_index = threadIdx.x;
  int stride = blockDim.x;
  int n_pixels = height * width;

  bboxes += batch_index * n_objects * 4;
  n_points += batch_index * n_objects;
  mask += batch_index * n_objects * n_pixels;
  dist_matrix += batch_index * n_objects * n_pixels;

  // Initialize the values for bboxes
  for (int i = 1; i < n_objects; ++i) {
    bboxes[i * 4] = 32767;
    bboxes[i * 4 + 2] = 32767;
  }

  // Get the bounding boxes of objects
  for (int i = 1; i < n_objects; ++i) {
    for (int j = thread_index; j < n_pixels; j += stride) {
      int x = j % width;
      int y = j / width;

      if (mask[i * n_pixels + j] >= prob_threshold) {
        atomicAdd(&n_points[i], 1);
        atomicMin(&bboxes[i * 4], x);     // bbox: x_min
        atomicMax(&bboxes[i * 4 + 1], x); // bbox: x_max
        atomicMin(&bboxes[i * 4 + 2], y); // bbox: y_min
        atomicMax(&bboxes[i * 4 + 3], y); // bbox: y_max
      }
    }
  }
  __syncthreads();

  // Get distance matrix
  for (int i = 1; i < n_objects; ++i) {
    if (n_points[i] == 0) {
      continue;
    }

    for (int j = thread_index; j < n_pixels; j += stride) {
      int x = j % width;
      int y = j / width;
      float diff_x = 0;
      float diff_y = 0;

      if (x < bboxes[i * 4]) {
        diff_x = bboxes[i * 4] - x;
      } else if (x > bboxes[i * 4 + 1]) {
        diff_x = bboxes[i * 4 + 1] - x;
      }
      if (y < bboxes[i * 4 + 2]) {
        diff_y = bboxes[i * 4 + 2] - y;
      } else if (y > bboxes[i * 4 + 3]) {
        diff_y = bboxes[i * 4 + 3] - y;
      }
      dist_matrix[i * n_pixels + j] = diff_x * diff_x + diff_y * diff_y;
    }
  }
}

torch::Tensor dist_matrix_cuda_forward(torch::Tensor mask, float prob_threshold,
                                       cudaStream_t stream) {
  int batch_size = mask.size(0);
  int n_objects = mask.size(1);
  int height = mask.size(2);
  int width = mask.size(3);

  torch::Tensor bboxes =
      torch::zeros({batch_size, n_objects, 4}, torch::CUDA(torch::kInt));
  torch::Tensor n_points =
      torch::zeros({batch_size, n_objects}, torch::CUDA(torch::kInt));
  torch::Tensor dist_matrix = torch::zeros(
      {batch_size, n_objects, height, width}, torch::CUDA(torch::kFloat));

  dist_matrix_forward_kernel<<<batch_size, CUDA_NUM_THREADS, 0, stream>>>(
      n_objects, height, width, prob_threshold, mask.data_ptr<float>(),
      bboxes.data_ptr<int>(), n_points.data_ptr<int>(),
      dist_matrix.data_ptr<float>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "Error in dist_matrix_cuda_forward: "
              << cudaGetErrorString(err);
  }
  return dist_matrix;
}
