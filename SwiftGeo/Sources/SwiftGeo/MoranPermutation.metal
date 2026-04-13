#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Moran's I permutation — Metal compute shader
//
// Each thread handles one permutation independently.
// All threads share the same z vector and sparse weight arrays (read-only).
// Each thread writes its result to nullDist[threadId].
//
// Memory layout:
//   z         — float array [n]         standardized variable
//   wRows     — uint array  [nnz]       COO row indices
//   wCols     — uint array  [nnz]       COO col indices
//   wVals     — float array [nnz]       weight values
//   nullDist  — float array [nPerm]     output null distribution
//   params    — MoranParams             n, nnz, nPerm, seed
// ---------------------------------------------------------------------------

struct MoranParams {
    uint n;
    uint nnz;
    uint nPerm;
    uint seed;
};

// ---------------------------------------------------------------------------
// xorshift64 PRNG — fast, low-overhead, sufficient for permutation tests
// Each thread seeds from global seed + thread id to ensure independence.
// ---------------------------------------------------------------------------
inline uint64_t xorshift64(uint64_t state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

// ---------------------------------------------------------------------------
// Fisher-Yates shuffle in thread-local array
// We copy z into a local buffer, shuffle it, then compute Moran's I.
// Metal doesn't support dynamic stack allocation > ~4KB so we use a fixed
// max size. For larger n we'll use a device buffer approach instead.
// ---------------------------------------------------------------------------
kernel void moranPermutation(
    device const float*        z        [[ buffer(0) ]],
    device const uint*         wRows    [[ buffer(1) ]],
    device const uint*         wCols    [[ buffer(2) ]],
    device const float*        wVals    [[ buffer(3) ]],
    device float*              nullDist [[ buffer(4) ]],
    device const MoranParams&  params   [[ buffer(5) ]],
    device float*              zPerms   [[ buffer(6) ]],  // scratch: [nPerm * n]
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= params.nPerm) return;

    const uint n   = params.n;
    const uint nnz = params.nnz;

    // Each thread gets its own slice of the scratch buffer
    device float* zPerm = zPerms + (tid * n);

    // Copy z into scratch
    for (uint i = 0; i < n; i++) {
        zPerm[i] = z[i];
    }

    // Seed per-thread RNG: mix global seed with thread id
    uint64_t state = (uint64_t)params.seed + (uint64_t)tid * 6364136223846793005ULL;

    // Fisher-Yates shuffle
    for (uint i = n - 1; i > 0; i--) {
        state = xorshift64(state);
        uint j = uint(state >> 33) % (i + 1);
        float tmp  = zPerm[i];
        zPerm[i]   = zPerm[j];
        zPerm[j]   = tmp;
    }

    // Compute spatial lag: wz[i] = sum_j w_ij * zPerm[j]
    // We accumulate directly into the dot product to avoid needing a wz array
    // For small n this is fine; for large n a two-pass approach is better.
    float moranI = 0.0;
    for (uint k = 0; k < nnz; k++) {
        uint i = wRows[k];
        uint j = wCols[k];
        // Accumulate z[i] * w_ij * zPerm[j]
        moranI += z[i] * wVals[k] * zPerm[j];
    }
    moranI /= float(n);

    nullDist[tid] = moranI;
}
