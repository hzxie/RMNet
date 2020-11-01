/*
 * @Author: Haozhe Xie
 * @Date:   2020-08-14 17:22:12
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2020-11-01 14:35:20
 * @Email:  cshzxie@gmail.com
 */

#include <iostream>
#include <cmath>
#include <torch/types.h>

#define CUDA_NUM_THREADS 512

__global__ void reg_att_map_generator_cuda_kernel(
    int n_objects, int height, int width, const float prob_threshold,
    int n_pts_threshold, int dist_threshold, const float *__restrict__ mask,
    int *__restrict__ bboxes, int *__restrict__ n_points,
    float *__restrict__ reg_att_map) {
  int batch_index = blockIdx.x;
  int thread_index = threadIdx.x;
  int stride = blockDim.x;
  int n_pixels = height * width;

  bboxes += batch_index * n_objects * 4;
  n_points += batch_index * n_objects;
  mask += batch_index * n_objects * n_pixels;
  reg_att_map += batch_index * n_objects * n_pixels;

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

  // Calculate the distance matrix according to the bounding boxes
  for (int i = 1; i < n_objects; ++i) {
    if (n_points[i] < n_pts_threshold) {
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

      // Determine the values of the attention map
      float dist = std::sqrt(diff_x * diff_x + diff_y * diff_y);
      if (dist > dist_threshold) {
        reg_att_map[i * n_pixels + j] = 0;
      }
    }
  }
}

torch::Tensor reg_att_map_generator_cuda_forward(torch::Tensor mask,
                                                 float prob_threshold,
                                                 int n_pts_threshold,
                                                 int dist_threshold,
                                                 cudaStream_t stream) {
  int batch_size = mask.size(0);
  int n_objects = mask.size(1);
  int height = mask.size(2);
  int width = mask.size(3);

  torch::Tensor bboxes =
      torch::zeros({batch_size, n_objects, 4}, torch::CUDA(torch::kInt));
  torch::Tensor n_points =
      torch::zeros({batch_size, n_objects}, torch::CUDA(torch::kInt));
  torch::Tensor reg_att_map = torch::ones(
      {batch_size, n_objects, height, width}, torch::CUDA(torch::kFloat));

  reg_att_map_generator_cuda_kernel<<<batch_size, CUDA_NUM_THREADS, 0,
                                      stream>>>(
      n_objects, height, width, prob_threshold, n_pts_threshold, dist_threshold,
      mask.data_ptr<float>(), bboxes.data_ptr<int>(), n_points.data_ptr<int>(),
      reg_att_map.data_ptr<float>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "Error in reg_att_map_generator_cuda_forward: "
              << cudaGetErrorString(err);
  }
  return reg_att_map;
}
