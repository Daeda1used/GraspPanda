// Arithmetic shared by the CPU and the CUDA kernels of grasppanda._ops.
//
// The kernels are held to the results of the pure-PyTorch operators in grasppanda.ops,
// bit for bit, so every floating-point expression below mirrors one expression of the
// reference and says which. Two things would silently break that:
//
// * fused multiply-add. `a * b + c` computed with one rounding differs from the two
//   roundings PyTorch performs. The CUDA build uses the `__f*_rn` intrinsics, which are
//   never fused; the host build is compiled with -ffp-contract=off.
// * the order of a three-term sum. Where the reference spells the order out, so do we.
//   Where it calls `sum(-1)`, the order is the one of PyTorch's reduction on that
//   device: left to right on the CPU, `(x + z) + y` on CUDA. The backend's self-check
//   compares `gp_probe_*` with PyTorch bit for bit before a kernel is used.
#pragma once

#include <cstdint>

#define GP_ABI_VERSION 1

// Tie rules of furthest point sampling (grasppanda.ops.furthest_point_sample).
#define GP_TIE_LOWEST_INDEX 0
#define GP_TIE_POINTNET2_CUDA 1

// Status codes returned by every entry point.
#define GP_OK 0
#define GP_ERROR_MEMORY 1
#define GP_ERROR_CUDA 2

#ifdef __CUDACC__
#define GP_INLINE __device__ __forceinline__
#define GP_MUL(a, b) __fmul_rn((a), (b))
#define GP_ADD(a, b) __fadd_rn((a), (b))
#define GP_SUB(a, b) __fsub_rn((a), (b))
#define GP_ABS(a) fabsf(a)
#else
#include <cmath>
#define GP_INLINE static inline
#define GP_MUL(a, b) ((a) * (b))
#define GP_ADD(a, b) ((a) + (b))
#define GP_SUB(a, b) ((a) - (b))
#define GP_ABS(a) std::fabs(a)
#endif

// `x² + y² + z²` added left to right: `_squared_norm` of grasppanda/ops/sampling.py,
// which is written that way so that it is the same on every device.
GP_INLINE float gp_squared_norm(float x, float y, float z) {
    return GP_ADD(GP_ADD(GP_MUL(x, x), GP_MUL(y, y)), GP_MUL(z, z));
}

// `(a - b).pow(2).sum(-1)` as PyTorch evaluates it on this device (ball query, kNN).
GP_INLINE float gp_squared_distance(float dx, float dy, float dz) {
#ifdef __CUDACC__
    return GP_ADD(GP_ADD(GP_MUL(dx, dx), GP_MUL(dz, dz)), GP_MUL(dy, dy));
#else
    return GP_ADD(GP_ADD(GP_MUL(dx, dx), GP_MUL(dy, dy)), GP_MUL(dz, dz));
#endif
}

// `torch.minimum`: the smaller value, and NaN as soon as one side is NaN.
GP_INLINE float gp_minimum(float current, float candidate) {
    return candidate != candidate ? candidate : (candidate < current ? candidate : current);
}

// Rank of a point in the tie order of the PointNet++ CUDA kernel, without the index:
// the `bits` bits of `index mod threads` reversed (`_kernel_priority` of sampling.py).
GP_INLINE uint32_t gp_reversed_thread(uint32_t index, uint32_t threads_mask, int bits) {
    uint32_t v = index & threads_mask;
    if (bits == 0) return 0;
#ifdef __CUDACC__
    v = __brev(v);
#else
    v = ((v >> 1) & 0x55555555u) | ((v & 0x55555555u) << 1);
    v = ((v >> 2) & 0x33333333u) | ((v & 0x33333333u) << 2);
    v = ((v >> 4) & 0x0F0F0F0Fu) | ((v & 0x0F0F0F0Fu) << 4);
    v = ((v >> 8) & 0x00FF00FFu) | ((v & 0x00FF00FFu) << 8);
    v = (v >> 16) | (v << 16);
#endif
    return v >> (32 - bits);
}

// Order of a stable ascending sort of distances: by value with NaN last, then by index.
GP_INLINE bool gp_nearer(float a, int64_t a_index, float b, int64_t b_index) {
    const bool a_nan = a != a, b_nan = b != b;
    if (a_nan || b_nan) return a_nan == b_nan ? a_index < b_index : b_nan;
    return a < b || (a == b && a_index < b_index);
}

// ----------------------------------------------------------------- cylinder query
//
// The reference rotates the offsets with a batched matrix product, i.e. through a BLAS
// whose rounding is not specified (it differs between MKL and cuBLAS). No kernel can
// reproduce that bit for bit, so the cylinder kernels classify every pair as inside,
// outside or *undecided*: undecided when the value computed here is closer to a
// threshold than the two computations can differ. A query that meets an undecided
// pair before it is complete is handed back (hits = -1) and re-run by the reference.
//
// Bound. For a dot product of three terms, any summation order with or without fused
// multiply-add returns a value within 3u·S/(1-3u) of the exact one, where
// S = Σ|oⱼ·Rⱼₖ| and u = 2⁻²⁴ (Higham, Accuracy and Stability of Numerical Algorithms,
// §3.1). Two such values are therefore within 6u·S(1+ε) of each other; GP_BAND_DOT is
// 8u, plus a few denormals for products that underflow. When S = 0 every product is
// an exact zero and both sides agree exactly, so the band is zero.
#define GP_BAND_DOT 4.76837158e-07f  // 8u = 4 * FLT_EPSILON
#define GP_BAND_TINY 2.24207754e-44f // 16 denormals

struct GpCylinderPair {
    bool inside;
    bool undecided;
};

GP_INLINE GpCylinderPair gp_cylinder_test(
    float dx, float dy, float dz, const float* r, float radius_squared, float hmin, float hmax) {
    float local[3], band[3];
    for (int k = 0; k < 3; ++k) {
        // local = offset · R (row vector times matrix), column k
        const float a = GP_MUL(dx, r[k]), b = GP_MUL(dy, r[3 + k]), c = GP_MUL(dz, r[6 + k]);
        local[k] = GP_ADD(GP_ADD(a, b), c);
        const float magnitude = GP_ABS(a) + GP_ABS(b) + GP_ABS(c);
        band[k] = magnitude > 0.0f ? GP_BAND_DOT * magnitude + GP_BAND_TINY : 0.0f;
    }
    const float axial = local[0];
    const float radial = GP_ADD(GP_MUL(local[1], local[1]), GP_MUL(local[2], local[2]));
    // |y² - y'²| <= band·(2|y| + band) for |y - y'| <= band; the squares and their sum
    // add three more roundings on each side (8u·radial covers them with room to spare).
    float radial_band = 0.0f;
    if (band[1] > 0.0f || band[2] > 0.0f) {
        radial_band = 2.0f * (band[1] * (2.0f * GP_ABS(local[1]) + band[1]) +
                              band[2] * (2.0f * GP_ABS(local[2]) + band[2])) +
                      GP_BAND_DOT * radial + GP_BAND_TINY;
    }
    GpCylinderPair pair;
    // Reference: (radial < radius²) & (axial > hmin) & (axial < hmax). With zero bands
    // the two tests below are that expression and its negation, so nothing is undecided.
    pair.inside = (radius_squared - radial > radial_band) && (axial - hmin > band[0]) &&
                  (hmax - axial > band[0]);
    const bool outside = (radial - radius_squared >= radial_band) || (hmin - axial >= band[0]) ||
                         (axial - hmax >= band[0]);
    pair.undecided = !pair.inside && !outside;
    return pair;
}
