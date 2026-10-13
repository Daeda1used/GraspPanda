// CUDA kernels of grasppanda._ops, behind a C interface.
//
// Written for this repository from the operator definitions in grasppanda/ops; see
// common.h for the arithmetic contract. Buffers are device memory owned by PyTorch
// tensors; kernels are queued on the stream PyTorch is using and are not waited for.

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

#include "common.h"

#define GP_EXPORT extern "C" __attribute__((visibility("default")))
#define GP_WARP 0xffffffffu

GP_EXPORT int gp_has_cuda() { return 1; }

GP_EXPORT const char* gp_cuda_error(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}

namespace {

// Makes `device` current for the calling thread while a kernel is queued.
class DeviceScope {
  public:
    explicit DeviceScope(int device) {
        status_ = cudaGetDevice(&previous_);
        if (status_ == cudaSuccess && previous_ != device) status_ = cudaSetDevice(device);
        changed_ = status_ == cudaSuccess && previous_ != device;
    }
    ~DeviceScope() {
        if (changed_) cudaSetDevice(previous_);
    }
    cudaError_t status() const { return status_; }

  private:
    int previous_ = 0;
    bool changed_ = false;
    cudaError_t status_ = cudaSuccess;
};

// Status of the launch that was just queued (reading it also clears it).
inline int launch_status() { return static_cast<int>(cudaGetLastError()); }

// ---------------------------------------------------------- furthest point sampling

struct Candidate {
    float value;
    int index;  // -1: no candidate
};

// True when `a` wins over `b`. NaN beats every number (torch.argmax); among equals the
// lower index wins, or with the kernel tie order the lower (reversed thread, index).
__device__ __forceinline__ bool wins(
    const Candidate& a, const Candidate& b, int tie_break, uint32_t mask, int bits) {
    if (a.index < 0) return false;
    if (b.index < 0) return true;
    const bool a_nan = a.value != a.value, b_nan = b.value != b.value;
    if (a_nan || b_nan) return a_nan && (!b_nan || a.index < b.index);
    if (a.value != b.value) return a.value > b.value;
    if (tie_break == GP_TIE_POINTNET2_CUDA) {
        const uint32_t ra = gp_reversed_thread(static_cast<uint32_t>(a.index), mask, bits);
        const uint32_t rb = gp_reversed_thread(static_cast<uint32_t>(b.index), mask, bits);
        if (ra != rb) return ra < rb;
    }
    return a.index < b.index;
}

__device__ __forceinline__ Candidate warp_winner(
    Candidate best, int tie_break, uint32_t mask, int bits) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        Candidate other;
        other.value = __shfl_down_sync(GP_WARP, best.value, offset);
        other.index = __shfl_down_sync(GP_WARP, best.index, offset);
        if (wins(other, best, tie_break, mask, bits)) best = other;
    }
    return best;  // valid in lane 0
}

// Distance of a point to the picks so far after a new pick at `centre`.
__device__ __forceinline__ float updated_distance(const float4& point, const float4& centre) {
    if (point.w == -1.0f) return -1.0f;  // never picked, see below
    const float d = gp_squared_norm(
        GP_SUB(point.x, centre.x), GP_SUB(point.y, centre.y), GP_SUB(point.z, centre.z));
    return gp_minimum(point.w, d);
}

// One block per cloud; the block walks through all samples. `work` holds x, y, z and
// the running distance of every point. Thread t owns the points k ≡ t (mod blockDim.x).
// The launcher makes blockDim.x a multiple of `priority_threads`, so the points of one
// thread share their rank in the kernel tie order and "the first of my maxima" is the
// right local winner under either tie rule; `wins` then is a total order, and the
// result does not depend on the thread layout.
__global__ void fps_kernel(
    const float* __restrict__ points, float4* __restrict__ work, int64_t* __restrict__ index,
    int count, int num_samples, int skip_near_origin, float near_origin, int tie_break,
    int priority_threads) {
    __shared__ float warp_value[32];
    __shared__ int warp_index[32];
    __shared__ int selected;

    const int thread = threadIdx.x;
    const int threads = blockDim.x;
    points += static_cast<size_t>(blockIdx.x) * count * 3;
    work += static_cast<size_t>(blockIdx.x) * count;
    index += static_cast<size_t>(blockIdx.x) * num_samples;

    int bits = 0;
    while ((1 << (bits + 1)) <= priority_threads) ++bits;
    const uint32_t mask = static_cast<uint32_t>(priority_threads - 1);

    // A distance of -1 marks a point that may never be picked; see fps_cloud in
    // kernels_cpu.cpp.
    for (int k = thread; k < count; k += threads) {
        const float x = points[3 * k], y = points[3 * k + 1], z = points[3 * k + 2];
        const bool ignore = skip_near_origin && gp_squared_norm(x, y, z) <= near_origin;
        work[k] = make_float4(x, y, z, ignore ? -1.0f : INFINITY);
    }
    if (thread == 0) index[0] = 0;
    __syncthreads();
    int current = 0;
    for (int step = 1; step < num_samples; ++step) {
        const float4 centre = work[current];
        float furthest = -INFINITY;
        int furthest_index = -1;
        int first_nan = INT_MAX;
        for (int k = thread; k < count; k += threads) {
            const float value = updated_distance(work[k], centre);
            work[k].w = value;
            if (value > furthest) {  // strict: the first maximum of this thread stays
                furthest = value;
                furthest_index = k;
            } else if (value != value && k < first_nan) {
                first_nan = k;
            }
        }
        Candidate best;
        best.value = first_nan != INT_MAX ? NAN : furthest;
        best.index = first_nan != INT_MAX ? first_nan : furthest_index;
        best = warp_winner(best, tie_break, mask, bits);
        if ((thread & 31) == 0) {
            warp_value[thread >> 5] = best.value;
            warp_index[thread >> 5] = best.index;
        }
        __syncthreads();
        if (thread < 32) {
            const bool present = thread < (threads >> 5);
            best.value = present ? warp_value[thread] : 0.0f;
            best.index = present ? warp_index[thread] : -1;
            best = warp_winner(best, tie_break, mask, bits);
            if (thread == 0) {
                // A NaN maximum equals nothing: the reference's arg-min over the tie
                // order then runs over an all-equal row and returns index 0.
                const bool nan = best.value != best.value;
                selected = (nan && tie_break == GP_TIE_POINTNET2_CUDA) ? 0 : best.index;
                index[step] = selected;
            }
        }
        __syncthreads();
        current = selected;
    }
}

// ------------------------------------------------------------------------- queries
//
// One warp per query. Lane l looks at the support points `base + l`, so one step
// covers 32 consecutive points and `__ballot_sync` returns their verdicts in index
// order. Everything that steers the loop is derived from those shared masks, hence
// identical in all lanes.

__device__ __forceinline__ int64_t warp_query() {
    return (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
}

// Stores the hits of one step and returns the new number found. Hit number r goes to
// slot r; slots that stay free are filled by `pad` once the query is complete.
__device__ __forceinline__ int store_hits(
    int64_t* slots, int nsample, int found, unsigned hit_mask, bool hit, int lane, int point) {
    const int rank = found + __popc(hit_mask & ((1u << lane) - 1u));
    if (hit && rank < nsample) slots[rank] = point;
    const int total = found + __popc(hit_mask);
    return total < nsample ? total : nsample;
}

// Missing slots repeat the first hit; a query without hits returns index 0 everywhere.
__device__ __forceinline__ void pad(int64_t* slots, int nsample, int found, int first, int lane) {
    for (int slot = lane; slot < nsample; slot += 32) {
        if (slot >= found) slots[slot] = first;
    }
}

__global__ void ball_query_kernel(
    const float* __restrict__ query, const float* __restrict__ support,
    int64_t* __restrict__ index, int64_t* __restrict__ hits, int64_t total, int queries,
    int count, float limit, int nsample) {
    const int64_t q = warp_query();
    if (q >= total) return;
    const int lane = threadIdx.x & 31;
    const float* cloud = support + (q / queries) * count * 3;
    const float qx = query[3 * q], qy = query[3 * q + 1], qz = query[3 * q + 2];
    int64_t* slots = index + q * nsample;
    int found = 0, first = 0;
    for (int base = 0; base < count && found < nsample; base += 32) {
        const int j = base + lane;
        bool hit = false;
        if (j < count) {
            const float d = gp_squared_distance(
                GP_SUB(cloud[3 * j], qx), GP_SUB(cloud[3 * j + 1], qy),
                GP_SUB(cloud[3 * j + 2], qz));
            hit = d < limit;
        }
        const unsigned hit_mask = __ballot_sync(GP_WARP, hit);
        if (hit_mask == 0) continue;
        if (found == 0) first = base + __ffs(hit_mask) - 1;
        found = store_hits(slots, nsample, found, hit_mask, hit, lane, j);
    }
    pad(slots, nsample, found, first, lane);
    if (lane == 0) hits[q] = found;
}

__global__ void cylinder_query_kernel(
    const float* __restrict__ query, const float* __restrict__ support,
    const float* __restrict__ rotation, int64_t* __restrict__ index, int64_t* __restrict__ hits,
    int64_t total, int queries, int count, float limit, float hmin, float hmax, int nsample) {
    const int64_t q = warp_query();
    if (q >= total) return;
    const int lane = threadIdx.x & 31;
    const float* cloud = support + (q / queries) * count * 3;
    float frame[9];
    for (int entry = 0; entry < 9; ++entry) frame[entry] = rotation[9 * q + entry];
    const float qx = query[3 * q], qy = query[3 * q + 1], qz = query[3 * q + 2];
    int64_t* slots = index + q * nsample;
    int found = 0, first = 0;
    bool undecided = false;
    for (int base = 0; base < count && found < nsample; base += 32) {
        const int j = base + lane;
        GpCylinderPair pair;
        pair.inside = false;
        pair.undecided = false;
        if (j < count) {
            pair = gp_cylinder_test(
                GP_SUB(cloud[3 * j], qx), GP_SUB(cloud[3 * j + 1], qy),
                GP_SUB(cloud[3 * j + 2], qz), frame, limit, hmin, hmax);
        }
        const unsigned hit_mask = __ballot_sync(GP_WARP, pair.inside);
        const unsigned open_mask = __ballot_sync(GP_WARP, pair.undecided);
        if (open_mask != 0) {
            // An undecided point matters if the scan reaches it, i.e. unless the query
            // is completed by hits that come before it.
            unsigned reached = GP_WARP;
            const int needed = nsample - found;
            if (__popc(hit_mask) >= needed) {
                unsigned last = hit_mask;
                for (int skipped = 1; skipped < needed; ++skipped) last &= last - 1;
                reached = (last & (0u - last)) - 1u;  // the bits below the completing hit
            }
            if (open_mask & reached) {
                undecided = true;
                break;
            }
        }
        if (hit_mask == 0) continue;
        if (found == 0) first = base + __ffs(hit_mask) - 1;
        found = store_hits(slots, nsample, found, hit_mask, pair.inside, lane, j);
    }
    if (undecided) found = first = 0;
    pad(slots, nsample, found, first, lane);
    if (lane == 0) hits[q] = undecided ? -1 : found;
}

// The k nearest in k passes: each pass takes the nearest point that comes after the
// previous pick in the order of a stable sort, so no per-query list is kept. Returns
// squared distances; see gp_knn_cpu.
__global__ void knn_kernel(
    const float* __restrict__ query, const float* __restrict__ support,
    float* __restrict__ distance, int64_t* __restrict__ index, int64_t total, int queries,
    int count, int k) {
    const int64_t q = warp_query();
    if (q >= total) return;
    const int lane = threadIdx.x & 31;
    const float* cloud = support + (q / queries) * count * 3;
    const float qx = query[3 * q], qy = query[3 * q + 1], qz = query[3 * q + 2];
    float previous = 0.0f;
    int previous_index = -1;
    for (int pass = 0; pass < k; ++pass) {
        float nearest = 0.0f;
        int nearest_index = -1;
        for (int j = lane; j < count; j += 32) {
            const float d = gp_squared_distance(
                GP_SUB(qx, cloud[3 * j]), GP_SUB(qy, cloud[3 * j + 1]),
                GP_SUB(qz, cloud[3 * j + 2]));
            if (previous_index >= 0 && !gp_nearer(previous, previous_index, d, j)) continue;
            if (nearest_index < 0 || gp_nearer(d, j, nearest, nearest_index)) {
                nearest = d;
                nearest_index = j;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_down_sync(GP_WARP, nearest, offset);
            const int other_index = __shfl_down_sync(GP_WARP, nearest_index, offset);
            if (other_index >= 0 &&
                (nearest_index < 0 || gp_nearer(other, other_index, nearest, nearest_index))) {
                nearest = other;
                nearest_index = other_index;
            }
        }
        previous = __shfl_sync(GP_WARP, nearest, 0);
        previous_index = __shfl_sync(GP_WARP, nearest_index, 0);
        if (lane == 0) {
            distance[q * k + pass] = previous;
            index[q * k + pass] = previous_index;
        }
    }
}

__global__ void probe_kernel(
    const float* __restrict__ first, const float* __restrict__ second,
    float* __restrict__ squared_norm, float* __restrict__ squared_distance, int64_t count) {
    const int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const float dx = GP_SUB(first[3 * k], second[3 * k]);
    const float dy = GP_SUB(first[3 * k + 1], second[3 * k + 1]);
    const float dz = GP_SUB(first[3 * k + 2], second[3 * k + 2]);
    squared_norm[k] = gp_squared_norm(dx, dy, dz);
    squared_distance[k] = gp_squared_distance(dx, dy, dz);
}

// Four warps, i.e. four queries, per block.
const int kQueryThreads = 128;

inline unsigned query_blocks(int64_t total) {
    return static_cast<unsigned>((total * 32 + kQueryThreads - 1) / kQueryThreads);
}

}  // namespace

GP_EXPORT int gp_fps_cuda(
    const float* points, float* work, int64_t batch, int64_t count, int64_t num_samples,
    int skip_near_origin, double near_origin, int tie_break, int priority_threads,
    int64_t* index, int device, void* stream) {
    DeviceScope scope(device);
    if (scope.status() != cudaSuccess) return static_cast<int>(scope.status());
    // A power of two from one warp up; more threads than points would only idle. Under
    // the kernel tie order it must also be a multiple of `priority_threads` (<= 512).
    int threads = 32;
    while (threads < 1024 && threads * 8 < count) threads *= 2;
    if (tie_break == GP_TIE_POINTNET2_CUDA && threads < priority_threads) {
        threads = priority_threads;
    }
    fps_kernel<<<static_cast<unsigned>(batch), threads, 0, static_cast<cudaStream_t>(stream)>>>(
        points, reinterpret_cast<float4*>(work), index, static_cast<int>(count),
        static_cast<int>(num_samples), skip_near_origin, static_cast<float>(near_origin),
        tie_break, priority_threads);
    return launch_status();
}

GP_EXPORT int gp_ball_query_cuda(
    const float* query, const float* support, int64_t batch, int64_t queries, int64_t count,
    double radius_squared, int64_t nsample, int64_t* index, int64_t* hits, int device,
    void* stream) {
    DeviceScope scope(device);
    if (scope.status() != cudaSuccess) return static_cast<int>(scope.status());
    const int64_t total = batch * queries;
    ball_query_kernel<<<query_blocks(total), kQueryThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        query, support, index, hits, total, static_cast<int>(queries), static_cast<int>(count),
        static_cast<float>(radius_squared), static_cast<int>(nsample));
    return launch_status();
}

GP_EXPORT int gp_cylinder_query_cuda(
    const float* query, const float* support, const float* rotation, int64_t batch,
    int64_t queries, int64_t count, double radius_squared, double hmin, double hmax,
    int64_t nsample, int64_t* index, int64_t* hits, int device, void* stream) {
    DeviceScope scope(device);
    if (scope.status() != cudaSuccess) return static_cast<int>(scope.status());
    const int64_t total = batch * queries;
    cylinder_query_kernel<<<query_blocks(total), kQueryThreads, 0,
                            static_cast<cudaStream_t>(stream)>>>(
        query, support, rotation, index, hits, total, static_cast<int>(queries),
        static_cast<int>(count), static_cast<float>(radius_squared), static_cast<float>(hmin),
        static_cast<float>(hmax), static_cast<int>(nsample));
    return launch_status();
}

GP_EXPORT int gp_knn_cuda(
    const float* query, const float* support, int64_t batch, int64_t queries, int64_t count,
    int64_t k, float* distance, int64_t* index, int device, void* stream) {
    DeviceScope scope(device);
    if (scope.status() != cudaSuccess) return static_cast<int>(scope.status());
    const int64_t total = batch * queries;
    knn_kernel<<<query_blocks(total), kQueryThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        query, support, distance, index, total, static_cast<int>(queries),
        static_cast<int>(count), static_cast<int>(k));
    return launch_status();
}

GP_EXPORT int gp_probe_cuda(
    const float* first, const float* second, int64_t count, float* squared_norm,
    float* squared_distance, int device, void* stream) {
    DeviceScope scope(device);
    if (scope.status() != cudaSuccess) return static_cast<int>(scope.status());
    const unsigned blocks = static_cast<unsigned>((count + 255) / 256);
    probe_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        first, second, squared_norm, squared_distance, count);
    return launch_status();
}
