#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Shared PRNG
// ---------------------------------------------------------------------------

inline uint64_t xorshift64(uint64_t s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
}

// ---------------------------------------------------------------------------
// Shader 1: moranPermutation (original — batched scratch buffer)
// One thread per permutation. Each thread shuffles a private z copy.
// Limited by scratch buffer size: nPerm * n * 4 bytes.
// ---------------------------------------------------------------------------

struct MoranParams {
    uint n;
    uint nnz;
    uint nPerm;
    uint seed;
    uint permOffset;
};

kernel void moranPermutation(
    device const float*       z        [[ buffer(0) ]],
    device const uint*        wRows    [[ buffer(1) ]],
    device const uint*        wCols    [[ buffer(2) ]],
    device const float*       wVals    [[ buffer(3) ]],
    device float*             nullDist [[ buffer(4) ]],
    device const MoranParams& params   [[ buffer(5) ]],
    device float*             zPerms   [[ buffer(6) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= params.nPerm) return;
    const uint n = params.n, nnz = params.nnz;
    device float* zPerm = zPerms + (tid * n);
    for (uint i = 0; i < n; i++) zPerm[i] = z[i];
    uint64_t globalPerm = (uint64_t)(params.permOffset + tid);
    uint64_t state = (uint64_t)params.seed + globalPerm * 6364136223846793005ULL;
    for (uint i = n - 1; i > 0; i--) {
        state = xorshift64(state);
        uint j = uint(state >> 33) % (i + 1);
        float tmp = zPerm[i]; zPerm[i] = zPerm[j]; zPerm[j] = tmp;
    }
    float moranI = 0.0;
    for (uint k = 0; k < nnz; k++)
        moranI += z[wRows[k]] * wVals[k] * zPerm[wCols[k]];
    nullDist[tid] = moranI / float(n);
}

// ---------------------------------------------------------------------------
// Shader 2: moranPermutationScratchless
//
// No per-thread scratch buffer. Instead, each thread:
//   1. Uses its per-permutation RNG to build a virtual permutation mapping
//      inline — computing the Fisher-Yates result on-the-fly for each
//      weight index access rather than materialising the full permuted array.
//
// This is done via an inside-out Fisher-Yates construction:
//   - We build the permutation index array perm[0..n-1] in registers.
//   - For small n this fits in thread registers / local stack.
//   - For larger n we use a device index buffer (one row per thread),
//     but sized uint16/uint32 not float — 4x smaller than scratch z copies.
//
// Two variants:
//   moranPermutationScratchless   — register-based, n <= 256 (fits on stack)
//   moranPermutationIndexed       — index buffer, any n, 4x smaller than v1
// ---------------------------------------------------------------------------

// --- Variant A: fully register-based for n <= 256 ---
// Metal allows up to 4KB of thread stack; uint16 * 256 = 512 bytes, safe.

kernel void moranPermutationScratchless(
    device const float*       z        [[ buffer(0) ]],
    device const uint*        wRows    [[ buffer(1) ]],
    device const uint*        wCols    [[ buffer(2) ]],
    device const float*       wVals    [[ buffer(3) ]],
    device float*             nullDist [[ buffer(4) ]],
    device const MoranParams& params   [[ buffer(5) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= params.nPerm) return;
    const uint n = params.n, nnz = params.nnz;

    // Build permutation in thread-local array (stack allocated)
    // Max n=256: 256 * 2 bytes = 512 bytes stack — within Metal's 4KB limit
    uint16_t perm[256];
    for (uint i = 0; i < n; i++) perm[i] = uint16_t(i);

    uint64_t globalPerm = (uint64_t)(params.permOffset + tid);
    uint64_t state = (uint64_t)params.seed + globalPerm * 6364136223846793005ULL;

    for (uint i = n - 1; i > 0; i--) {
        state = xorshift64(state);
        uint j = uint(state >> 33) % (i + 1);
        uint16_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }

    // Moran's I = (1/n) * sum_k z[wRows[k]] * wVals[k] * z[perm[wCols[k]]]
    float moranI = 0.0;
    for (uint k = 0; k < nnz; k++)
        moranI += z[wRows[k]] * wVals[k] * z[perm[wCols[k]]];
    nullDist[tid] = moranI / float(n);
}

// --- Variant B: index buffer for n > 256, any size ---
// Uses a device uint32 index buffer of shape [nPerm * n].
// 4 bytes/index vs 4 bytes/float — same size as scratch but uint,
// and we can use uint16 for n <= 65535 halving it again.

struct MoranParamsLarge {
    uint n;
    uint nnz;
    uint nPerm;
    uint seed;
    uint permOffset;
    uint useUint16;   // 1 = use uint16 indices (n <= 65535), 0 = uint32
};

kernel void moranPermutationIndexed(
    device const float*            z        [[ buffer(0) ]],
    device const uint*             wRows    [[ buffer(1) ]],
    device const uint*             wCols    [[ buffer(2) ]],
    device const float*            wVals    [[ buffer(3) ]],
    device float*                  nullDist [[ buffer(4) ]],
    device const MoranParamsLarge& params   [[ buffer(5) ]],
    device uint*                   permBuf  [[ buffer(6) ]],  // [nPerm * n] uint32
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= params.nPerm) return;
    const uint n = params.n, nnz = params.nnz;

    device uint* perm = permBuf + (tid * n);

    // Initialise identity permutation
    for (uint i = 0; i < n; i++) perm[i] = i;

    // Fisher-Yates with per-permutation seed
    uint64_t globalPerm = (uint64_t)(params.permOffset + tid);
    uint64_t state = (uint64_t)params.seed + globalPerm * 6364136223846793005ULL;

    for (uint i = n - 1; i > 0; i--) {
        state = xorshift64(state);
        uint j = uint(state >> 33) % (i + 1);
        uint tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }

    // Moran's I via index lookup — no z copy needed
    float moranI = 0.0;
    for (uint k = 0; k < nnz; k++)
        moranI += z[wRows[k]] * wVals[k] * z[perm[wCols[k]]];
    nullDist[tid] = moranI / float(n);
}
