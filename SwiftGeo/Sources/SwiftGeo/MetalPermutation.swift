// MetalPermutation.swift
// Moran's I permutation test via Metal GPU compute.
// Supports batched dispatch for large n where scratch buffer would exceed limits.

import Foundation
import Metal

struct MoranParamsMetal {
    var n: UInt32
    var nnz: UInt32
    var nPerm: UInt32      // permutations in THIS batch
    var seed: UInt32
    var permOffset: UInt32 // global permutation index offset for this batch
}

// MARK: - Shader source (embedded for CLI binary compatibility)

private let metalShaderSource = """
#include <metal_stdlib>
using namespace metal;

struct MoranParams {
    uint n;
    uint nnz;
    uint nPerm;
    uint seed;
    uint permOffset;
};

inline uint64_t xorshift64(uint64_t s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
}

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

    const uint n   = params.n;
    const uint nnz = params.nnz;

    // Each thread gets its own scratch slice
    device float* zPerm = zPerms + (tid * n);
    for (uint i = 0; i < n; i++) zPerm[i] = z[i];

    // Seed derived from global permutation index for cross-batch reproducibility
    uint64_t globalPerm = (uint64_t)(params.permOffset + tid);
    uint64_t state = (uint64_t)params.seed + globalPerm * 6364136223846793005ULL;

    // Fisher-Yates shuffle
    for (uint i = n - 1; i > 0; i--) {
        state = xorshift64(state);
        uint j = uint(state >> 33) % (i + 1);
        float tmp = zPerm[i]; zPerm[i] = zPerm[j]; zPerm[j] = tmp;
    }

    // Moran's I = (1/n) * z' W zPerm
    float moranI = 0.0;
    for (uint k = 0; k < nnz; k++)
        moranI += z[wRows[k]] * wVals[k] * zPerm[wCols[k]];
    nullDist[tid] = moranI / float(n);
}
"""

// MARK: - Engine

public class MetalMoranEngine {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    // Maximum scratch memory per batch (bytes)
    static let maxScratchBytes = 512 * 1024 * 1024  // 512MB

    public init?() {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            print("[Metal] No Metal device available.")
            return nil
        }
        guard let queue = dev.makeCommandQueue() else {
            print("[Metal] Could not create command queue.")
            return nil
        }

        // Load library: try bundle first, fall back to runtime compilation
        let library: MTLLibrary? = {
            if let lib = try? dev.makeDefaultLibrary(bundle: Bundle.module) { return lib }
            if let lib = try? dev.makeDefaultLibrary(bundle: Bundle.main)   { return lib }
            // Runtime compile from embedded source
            if let lib = try? dev.makeLibrary(source: metalShaderSource, options: nil) {
                print("[Metal] Compiled shader from source")
                return lib
            }
            return nil
        }()

        guard let lib = library,
              let fn  = lib.makeFunction(name: "moranPermutation"),
              let pso = try? dev.makeComputePipelineState(function: fn)
        else {
            print("[Metal] Could not load moranPermutation kernel.")
            return nil
        }

        self.device       = dev
        self.commandQueue = queue
        self.pipeline     = pso
        print("[Metal] Device: \(dev.name)")
    }

    // MARK: - Batch size calculation

    /// How many permutations fit in one batch given n observations.
    func batchSize(n: Int) -> Int {
        let bytesPerPerm = n * MemoryLayout<Float>.stride
        let maxPerms = MetalMoranEngine.maxScratchBytes / bytesPerPerm
        // Cap at pipeline maximum threads, floor at 1
        return max(1, min(maxPerms, pipeline.maxTotalThreadsPerThreadgroup * 1024))
    }

    // MARK: - Run (batched)

    public func run(
        z: [Double],
        weights: SparseWeights,
        nPermutations: Int,
        seed: UInt32 = 12345
    ) -> PermutationResult? {

        let n   = weights.n
        let nnz = weights.nnz
        let batch = batchSize(n: n)
        let nBatches = (nPermutations + batch - 1) / batch

        let zF     = z.map { Float($0) }
        let wValsF = weights.values.map { Float($0) }
        let wRowsU = weights.rows.map { UInt32($0) }
        let wColsU = weights.cols.map { UInt32($0) }

        // Allocate fixed GPU buffers (reused across batches)
        guard
            let zBuf = device.makeBuffer(bytes: zF,
                length: MemoryLayout<Float>.stride * n, options: .storageModeShared),
            let rowsBuf = device.makeBuffer(bytes: wRowsU,
                length: MemoryLayout<UInt32>.stride * nnz, options: .storageModeShared),
            let colsBuf = device.makeBuffer(bytes: wColsU,
                length: MemoryLayout<UInt32>.stride * nnz, options: .storageModeShared),
            let valsBuf = device.makeBuffer(bytes: wValsF,
                length: MemoryLayout<Float>.stride * nnz, options: .storageModeShared)
        else {
            print("[Metal] Static buffer allocation failed.")
            return nil
        }

        // Scratch buffer sized for one batch
        let scratchBytes = batch * n * MemoryLayout<Float>.stride
        let scratchMB    = scratchBytes / (1024 * 1024)
        guard let scratchBuf = device.makeBuffer(length: scratchBytes,
                                                 options: .storageModeShared)
        else {
            print("[Metal] Scratch buffer allocation failed (\(scratchMB)MB).")
            return nil
        }

        // Output buffer for one batch
        guard let nullBatchBuf = device.makeBuffer(
            length: MemoryLayout<Float>.stride * batch,
            options: .storageModeShared)
        else {
            print("[Metal] Null batch buffer allocation failed.")
            return nil
        }

        if nBatches > 1 {
            print("[Metal] Batching: \(nPermutations) perms in \(nBatches) batches of ≤\(batch) (scratch \(scratchMB)MB/batch)")
        }

        var nullDist = [Double]()
        nullDist.reserveCapacity(nPermutations)

        let start = Date()

        for batchIdx in 0..<nBatches {
            let offset     = batchIdx * batch
            let thisBatch  = min(batch, nPermutations - offset)

            var params = MoranParamsMetal(
                n: UInt32(n),
                nnz: UInt32(nnz),
                nPerm: UInt32(thisBatch),
                seed: seed,
                permOffset: UInt32(offset)
            )
            guard let paramsBuf = device.makeBuffer(bytes: &params,
                length: MemoryLayout<MoranParamsMetal>.size,
                options: .storageModeShared)
            else { continue }

            guard
                let cmdBuf  = commandQueue.makeCommandBuffer(),
                let encoder = cmdBuf.makeComputeCommandEncoder()
            else { continue }

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(zBuf,        offset: 0, index: 0)
            encoder.setBuffer(rowsBuf,     offset: 0, index: 1)
            encoder.setBuffer(colsBuf,     offset: 0, index: 2)
            encoder.setBuffer(valsBuf,     offset: 0, index: 3)
            encoder.setBuffer(nullBatchBuf,offset: 0, index: 4)
            encoder.setBuffer(paramsBuf,   offset: 0, index: 5)
            encoder.setBuffer(scratchBuf,  offset: 0, index: 6)

            let threadGroupSize = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
            encoder.dispatchThreads(
                MTLSize(width: thisBatch, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
            )
            encoder.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            // Read this batch's results
            let ptr = nullBatchBuf.contents().bindMemory(to: Float.self, capacity: thisBatch)
            for i in 0..<thisBatch {
                nullDist.append(Double(ptr[i]))
            }
        }

        let elapsed  = Date().timeIntervalSince(start)
        let observed = moranI(z: z, weights: weights)
        let absObs   = abs(observed)
        let extreme  = nullDist.filter { abs($0) >= absObs }.count
        let pValue   = Double(extreme) / Double(nPermutations)

        return PermutationResult(
            observed: observed,
            nullDistribution: nullDist,
            pValueTwoSided: pValue,
            elapsedSeconds: elapsed,
            nThreads: device.maxThreadsPerThreadgroup.width
        )
    }
}

// MARK: - Convenience wrapper

public func moranPermutationTestMetal(
    z: [Double],
    weights: SparseWeights,
    nPermutations: Int = 99999,
    seed: UInt32 = 12345
) -> PermutationResult? {
    guard let engine = MetalMoranEngine() else { return nil }
    return engine.run(z: z, weights: weights, nPermutations: nPermutations, seed: seed)
}
