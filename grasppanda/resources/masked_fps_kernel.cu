// GPU FPS over an explicit eligibility mask; equal distances prefer the lower row.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <climits>

__global__ void masked_fps_kernel(int n, int count, const float* points,
    const bool* eligible, const int64_t* starts, float* distance, int64_t* output) {
  const int b = blockIdx.x, tid = threadIdx.x;
  points += int64_t(b) * n * 3;
  eligible += int64_t(b) * n;
  distance += int64_t(b) * n;
  output += int64_t(b) * count;
  __shared__ float best_distance[256];
  __shared__ int best_index[256];
  int current = int(starts[b]);
  for (int step = 0; step < count; ++step) {
    if (tid == 0) output[step] = current;
    if (step + 1 == count) return;
    const float x = points[current * 3], y = points[current * 3 + 1], z = points[current * 3 + 2];
    float best = -1.0f;
    int index = INT_MAX;
    for (int k = tid; k < n; k += 256) {
      if (!eligible[k] || distance[k] < 0) continue;
      if (k == current) { distance[k] = -1.0f; continue; }
      const float dx = points[k * 3] - x, dy = points[k * 3 + 1] - y, dz = points[k * 3 + 2] - z;
      const float d = fminf(distance[k], dx * dx + dy * dy + dz * dz);
      distance[k] = d;
      if (d > best || (d == best && k < index)) { best = d; index = k; }
    }
    best_distance[tid] = best;
    best_index[tid] = index;
    __syncthreads();
    for (int stride = 128; stride > 0; stride /= 2) {
      if (tid < stride) {
        const float other = best_distance[tid + stride];
        const int other_index = best_index[tid + stride];
        if (other > best_distance[tid] || (other == best_distance[tid] && other_index < best_index[tid])) {
          best_distance[tid] = other;
          best_index[tid] = other_index;
        }
      }
      __syncthreads();
    }
    current = best_index[0];
    __syncthreads();
  }
}

void masked_fps_launch(const at::Tensor& points, const at::Tensor& eligible,
    const at::Tensor& starts, at::Tensor& distance, at::Tensor& output) {
  masked_fps_kernel<<<points.size(0), 256, 0, at::cuda::getCurrentCUDAStream(points.get_device())>>>(
      points.size(1), output.size(1), points.data_ptr<float>(), eligible.data_ptr<bool>(),
      starts.data_ptr<int64_t>(), distance.data_ptr<float>(), output.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
