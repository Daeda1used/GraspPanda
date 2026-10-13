// CPU kernels of grasppanda._ops: plain C++ with OpenMP, behind a C interface.
//
// Written for this repository from the operator definitions in grasppanda/ops; see
// common.h for the arithmetic contract. Every function takes contiguous float32 /
// int64 buffers that the Python side owns, and returns a status code.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include "common.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define GP_EXPORT extern "C" __attribute__((visibility("default")))

// A second copy of a hot function for AVX2, chosen when the library is loaded. Wider
// vectors only: every element still goes through the same IEEE operations (AVX2 does
// not include fused multiply-add).
#if defined(__x86_64__) && defined(__GLIBC__) && defined(__GNUC__) && !defined(__clang__)
#define GP_SIMD_CLONES __attribute__((target_clones("default", "avx2")))
#else
#define GP_SIMD_CLONES
#endif

GP_EXPORT int gp_abi_version() { return GP_ABI_VERSION; }

GP_EXPORT int gp_has_openmp() {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

#ifndef GP_WITH_CUDA
GP_EXPORT int gp_has_cuda() { return 0; }
#endif

// ---------------------------------------------------------- furthest point sampling

namespace {

struct FpsOptions {
    int64_t count;
    int64_t num_samples;
    bool skip_near_origin;
    float near_origin;
    int tie_break;
    int priority_threads;  // threads of the PointNet++ kernel whose tie order is reproduced
};

inline int32_t bits_of(float value) {
    int32_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    return bits;
}

// One cloud. `scratch` holds 4 * count floats: x, y, z and the running distance.
//
// A step is three passes that the compiler vectorises (update the distances, count
// NaNs, take the maximum) and a search for the winner among the maxima. The maximum is
// taken over the bit patterns: without NaN a distance is the marker -1 or lies in
// [+0, +inf] (a sum of squares is never -0), and such values order like their bits read
// as signed integers. An integer maximum vectorises; a floating-point one does not
// without flags that would change results.
GP_SIMD_CLONES
void fps_cloud(const float* points, const FpsOptions& options, int64_t* index, float* scratch) {
    const int64_t count = options.count;
    float* xs = scratch;
    float* ys = scratch + count;
    float* zs = scratch + 2 * count;
    // Squared distance to the selected set; -1 marks a point that may never be picked.
    // A real distance is never negative, so the marker cannot be mistaken for one, and
    // it equals the value the reference gives such a point (`masked_fill(ignore, -1)`).
    float* distance = scratch + 3 * count;
    for (int64_t k = 0; k < count; ++k) {
        xs[k] = points[3 * k];
        ys[k] = points[3 * k + 1];
        zs[k] = points[3 * k + 2];
        const bool ignore =
            options.skip_near_origin && gp_squared_norm(xs[k], ys[k], zs[k]) <= options.near_origin;
        distance[k] = ignore ? -1.0f : std::numeric_limits<float>::infinity();
    }
    int bits = 0;
    while ((1 << (bits + 1)) <= options.priority_threads) ++bits;
    const uint32_t mask = static_cast<uint32_t>(options.priority_threads - 1);

    int64_t current = 0;
    index[0] = 0;
    for (int64_t step = 1; step < options.num_samples; ++step) {
        const float cx = xs[current], cy = ys[current], cz = zs[current];
        for (int64_t k = 0; k < count; ++k) {
            const float before = distance[k];
            const float d = gp_squared_norm(xs[k] - cx, ys[k] - cy, zs[k] - cz);
            distance[k] = before == -1.0f ? before : gp_minimum(before, d);
        }
        int nans = 0;
        int32_t furthest = std::numeric_limits<int32_t>::min();
        for (int64_t k = 0; k < count; ++k) {
            const int32_t value = bits_of(distance[k]);
            furthest = value > furthest ? value : furthest;
            nans += distance[k] != distance[k];
        }
        int64_t best = 0;
        if (nans) {
            // torch.argmax ranks NaN above everything and returns the first one. With the
            // kernel tie order the reference compares against a NaN maximum, nothing is
            // equal to it, and the arg-min over an all-equal row is index 0.
            if (options.tie_break == GP_TIE_LOWEST_INDEX) {
                while (distance[best] == distance[best]) ++best;
            }
        } else if (options.tie_break == GP_TIE_LOWEST_INDEX) {
            while (bits_of(distance[best]) != furthest) ++best;
        } else {
            uint32_t best_rank = std::numeric_limits<uint32_t>::max();
            for (int64_t k = 0; k < count; ++k) {
                if (bits_of(distance[k]) != furthest) continue;
                const uint32_t rank = gp_reversed_thread(static_cast<uint32_t>(k), mask, bits);
                if (rank < best_rank) {  // equal ranks: the lower index, seen first, stays
                    best = k;
                    best_rank = rank;
                }
            }
        }
        current = best;
        index[step] = best;
    }
}

}  // namespace

GP_EXPORT int gp_fps_cpu(
    const float* points, int64_t batch, int64_t count, int64_t num_samples, int skip_near_origin,
    double near_origin, int tie_break, int priority_threads, int64_t* index, int threads) {
    FpsOptions options;
    options.count = count;
    options.num_samples = num_samples;
    options.skip_near_origin = skip_near_origin != 0;
    options.near_origin = static_cast<float>(near_origin);
    options.tie_break = tie_break;
    options.priority_threads = priority_threads;
    std::vector<float> scratch;
    try {
        scratch.resize(static_cast<size_t>(batch) * 4 * count);
    } catch (const std::bad_alloc&) {
        return GP_ERROR_MEMORY;
    }
    (void)threads;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads) if (threads > 1 && batch > 1)
    for (int64_t b = 0; b < batch; ++b) {
        fps_cloud(points + b * count * 3, options, index + b * num_samples,
                  scratch.data() + static_cast<size_t>(b) * 4 * count);
    }
    return GP_OK;
}

// ------------------------------------------------------------------------- queries

namespace {

// First hit fills every slot (the padding rule), later hits overwrite their own.
inline void record_hit(int64_t* slots, int64_t nsample, int64_t found, int64_t point) {
    if (found == 0) {
        for (int64_t slot = 0; slot < nsample; ++slot) slots[slot] = point;
    } else {
        slots[found] = point;
    }
}

}  // namespace

GP_EXPORT int gp_ball_query_cpu(
    const float* query, const float* support, int64_t batch, int64_t queries, int64_t count,
    double radius_squared, int64_t nsample, int64_t* index, int64_t* hits, int threads) {
    const float limit = static_cast<float>(radius_squared);
    const int64_t total = batch * queries;
    (void)threads;
#pragma omp parallel for schedule(dynamic, 16) num_threads(threads) if (threads > 1 && total > 16)
    for (int64_t q = 0; q < total; ++q) {
        const float* cloud = support + (q / queries) * count * 3;
        const float qx = query[3 * q], qy = query[3 * q + 1], qz = query[3 * q + 2];
        int64_t* slots = index + q * nsample;
        int64_t found = 0;
        for (int64_t j = 0; j < count && found < nsample; ++j) {
            const float d = gp_squared_distance(
                cloud[3 * j] - qx, cloud[3 * j + 1] - qy, cloud[3 * j + 2] - qz);
            if (d < limit) {
                record_hit(slots, nsample, found, j);
                ++found;
            }
        }
        if (found == 0) {
            for (int64_t slot = 0; slot < nsample; ++slot) slots[slot] = 0;
        }
        hits[q] = found;
    }
    return GP_OK;
}

GP_EXPORT int gp_cylinder_query_cpu(
    const float* query, const float* support, const float* rotation, int64_t batch,
    int64_t queries, int64_t count, double radius_squared, double hmin, double hmax,
    int64_t nsample, int64_t* index, int64_t* hits, int threads) {
    const float limit = static_cast<float>(radius_squared);
    const float low = static_cast<float>(hmin), high = static_cast<float>(hmax);
    const int64_t total = batch * queries;
    (void)threads;
#pragma omp parallel for schedule(dynamic, 16) num_threads(threads) if (threads > 1 && total > 16)
    for (int64_t q = 0; q < total; ++q) {
        const float* cloud = support + (q / queries) * count * 3;
        const float* frame = rotation + 9 * q;
        const float qx = query[3 * q], qy = query[3 * q + 1], qz = query[3 * q + 2];
        int64_t* slots = index + q * nsample;
        int64_t found = 0;
        bool undecided = false;
        for (int64_t j = 0; j < count && found < nsample; ++j) {
            const GpCylinderPair pair = gp_cylinder_test(
                cloud[3 * j] - qx, cloud[3 * j + 1] - qy, cloud[3 * j + 2] - qz, frame, limit,
                low, high);
            if (pair.undecided) {
                undecided = true;
                break;
            }
            if (pair.inside) {
                record_hit(slots, nsample, found, j);
                ++found;
            }
        }
        if (found == 0 || undecided) {
            for (int64_t slot = 0; slot < nsample; ++slot) slots[slot] = 0;
        }
        hits[q] = undecided ? -1 : found;
    }
    return GP_OK;
}

// Returns *squared* distances: the reference takes the root with PyTorch, whose CPU
// square root is not the correctly rounded one a kernel would compute.
GP_EXPORT int gp_knn_cpu(
    const float* query, const float* support, int64_t batch, int64_t queries, int64_t count,
    int64_t k, float* distance, int64_t* index, int threads) {
    const int64_t total = batch * queries;
    (void)threads;
#pragma omp parallel for schedule(dynamic, 16) num_threads(threads) if (threads > 1 && total > 16)
    for (int64_t q = 0; q < total; ++q) {
        const float* cloud = support + (q / queries) * count * 3;
        const float qx = query[3 * q], qy = query[3 * q + 1], qz = query[3 * q + 2];
        float* nearest = distance + q * k;  // squared distances, ascending
        int64_t* slots = index + q * k;
        int64_t kept = 0;
        for (int64_t j = 0; j < count; ++j) {
            const float d = gp_squared_distance(
                qx - cloud[3 * j], qy - cloud[3 * j + 1], qz - cloud[3 * j + 2]);
            // Kept in the order of a stable ascending sort (gp_nearer); points arrive
            // in index order, so a newcomer goes after its equals.
            int64_t place = kept;
            while (place > 0 && gp_nearer(d, j, nearest[place - 1], slots[place - 1])) --place;
            if (place >= k) continue;
            const int64_t last = kept < k ? kept : k - 1;
            for (int64_t slot = last; slot > place; --slot) {
                nearest[slot] = nearest[slot - 1];
                slots[slot] = slots[slot - 1];
            }
            nearest[place] = d;
            slots[place] = j;
            if (kept < k) ++kept;
        }
    }
    return GP_OK;
}

// Values the self-check compares with PyTorch bit for bit (see common.h).
GP_EXPORT int gp_probe_cpu(
    const float* first, const float* second, int64_t count, float* squared_norm,
    float* squared_distance) {
    for (int64_t k = 0; k < count; ++k) {
        const float dx = first[3 * k] - second[3 * k];
        const float dy = first[3 * k + 1] - second[3 * k + 1];
        const float dz = first[3 * k + 2] - second[3 * k + 2];
        squared_norm[k] = gp_squared_norm(dx, dy, dz);
        squared_distance[k] = gp_squared_distance(dx, dy, dz);
    }
    return GP_OK;
}
