// MetalPermutation.swift
// Moran's I permutation test via Metal GPU compute.
//
// Three shader variants, selected automatically based on n:
//   n <= 256: moranPermutationScratchless  (stack-allocated uint16 perm, single pass)
//   n <= 65535: moranPermutationIndexed    (uint32 index buffer, 4x smaller than float scratch)
//   any n, fallback: moranPermutation      (batched float scratch)

import Foundation
import Metal

// MARK: - Param structs (must match shader layouts exactly)

struct MoranParamsMetal {
    var n: UInt32
    var nnz: UInt32
    var nPerm: UInt32
    var seed: UInt32
    var permOffset: UInt32
}

struct MoranParamsMetalLarge {
    var n: UInt32
    var nnz: UInt32
    var nPerm: UInt32
    var seed: UInt32
    var permOffset: UInt32
    var useUint16: UInt32
}

// MARK: - Embedded shader source

private let metalShaderSource = #"""
#include <metal_stdlib>
using namespace metal;

inline uint64_t xorshift64(uint64_t s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
}

struct MoranParams      { uint n; uint nnz; uint nPerm; uint seed; uint permOffset; };
struct MoranParamsLarge { uint n; uint nnz; uint nPerm; uint seed; uint permOffset; uint useUint16; };

// Variant A: stack-allocated perm, n <= 256, no device scratch at all
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
    uint16_t perm[256];
    for (uint i = 0; i < n; i++) perm[i] = uint16_t(i);
    uint64_t state = (uint64_t)params.seed +
        ((uint64_t)(params.permOffset + tid)) * 6364136223846793005ULL;
    for (uint i = n - 1; i > 0; i--) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        uint j = uint(state >> 33) % (i + 1);
        uint16_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }
    float moranI = 0.0;
    for (uint k = 0; k < nnz; k++)
        moranI += z[wRows[k]] * wVals[k] * z[perm[wCols[k]]];
    nullDist[tid] = moranI / float(n);
}

// Variant B: uint32 index buffer, any n, single pass, 4x smaller than float scratch
kernel void moranPermutationIndexed(
    device const float*            z        [[ buffer(0) ]],
    device const uint*             wRows    [[ buffer(1) ]],
    device const uint*             wCols    [[ buffer(2) ]],
    device const float*            wVals    [[ buffer(3) ]],
    device float*                  nullDist [[ buffer(4) ]],
    device const MoranParamsLarge& params   [[ buffer(5) ]],
    device uint*                   permBuf  [[ buffer(6) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= params.nPerm) return;
    const uint n = params.n, nnz = params.nnz;
    device uint* perm = permBuf + (tid * n);
    for (uint i = 0; i < n; i++) perm[i] = i;
    uint64_t state = (uint64_t)params.seed +
        ((uint64_t)(params.permOffset + tid)) * 6364136223846793005ULL;
    for (uint i = n - 1; i > 0; i--) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        uint j = uint(state >> 33) % (i + 1);
        uint tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }
    float moranI = 0.0;
    for (uint k = 0; k < nnz; k++)
        moranI += z[wRows[k]] * wVals[k] * z[perm[wCols[k]]];
    nullDist[tid] = moranI / float(n);
}

// Variant C: original batched float scratch (fallback)
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
    uint64_t state = (uint64_t)params.seed +
        ((uint64_t)(params.permOffset + tid)) * 6364136223846793005ULL;
    for (uint i = n - 1; i > 0; i--) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        uint j = uint(state >> 33) % (i + 1);
        float tmp = zPerm[i]; zPerm[i] = zPerm[j]; zPerm[j] = tmp;
    }
    float moranI = 0.0;
    for (uint k = 0; k < nnz; k++)
        moranI += z[wRows[k]] * wVals[k] * zPerm[wCols[k]];
    nullDist[tid] = moranI / float(n);
}
"""#

// MARK: - Engine

public class MetalMoranEngine {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let psoScratchless: MTLComputePipelineState  // n <= 256
    private let psoIndexed: MTLComputePipelineState      // any n, uint32 index buf
    private let psoBatched: MTLComputePipelineState      // fallback float scratch

    // Use 40% of recommended working set — conservative enough to leave
    // headroom for macOS and the CPU side of the benchmark on 8GB machines.
    // Evaluated lazily per-device at runtime.
    static func maxMemoryBytes(for device: MTLDevice) -> Int {
        let recommended = device.recommendedMaxWorkingSetSize
        return Int(Double(recommended) * 0.40)
    }

    public init?() {
        guard let dev = MTLCreateSystemDefaultDevice(),
              let queue = dev.makeCommandQueue()
        else { return nil }

        let library: MTLLibrary? = {
            if let lib = try? dev.makeDefaultLibrary(bundle: Bundle.module) { return lib }
            if let lib = try? dev.makeDefaultLibrary(bundle: Bundle.main)   { return lib }
            return try? dev.makeLibrary(source: metalShaderSource, options: nil)
        }()

        guard let lib = library else {
            print("[Metal] Could not load or compile shader library.")
            return nil
        }

        guard
            let fnA = lib.makeFunction(name: "moranPermutationScratchless"),
            let fnB = lib.makeFunction(name: "moranPermutationIndexed"),
            let fnC = lib.makeFunction(name: "moranPermutation"),
            let psoA = try? dev.makeComputePipelineState(function: fnA),
            let psoB = try? dev.makeComputePipelineState(function: fnB),
            let psoC = try? dev.makeComputePipelineState(function: fnC)
        else {
            print("[Metal] Could not create pipeline states.")
            return nil
        }

        self.device           = dev
        self.commandQueue     = queue
        self.psoScratchless   = psoA
        self.psoIndexed       = psoB
        self.psoBatched       = psoC

        print("[Metal] Device: \(dev.name)")
    }

    // MARK: - Shader selection

    enum ShaderMode: String {
        case scratchless = "scratchless (n≤256, stack perm)"
        case indexed     = "indexed (uint32 perm buffer)"
        case batched     = "batched (float scratch, legacy)"
    }

    func selectMode(n: Int, nPerm: Int) -> ShaderMode {
        if n <= 256 { return .scratchless }
        let maxBytes = MetalMoranEngine.maxMemoryBytes(for: device)
        let indexedBytes = nPerm * n * MemoryLayout<UInt32>.stride
        if indexedBytes <= maxBytes { return .indexed }
        return .batched
    }

    // MARK: - Run

    public func run(
        z: [Double],
        weights: SparseWeights,
        nPermutations: Int,
        seed: UInt32 = 12345
    ) -> PermutationResult? {

        let n   = weights.n
        let nnz = weights.nnz
        let mode = selectMode(n: n, nPerm: nPermutations)
        print("[Metal] Shader: \(mode.rawValue), n=\(n), nPerm=\(nPermutations)")

        let zF     = z.map { Float($0) }
        let wValsF = weights.values.map { Float($0) }
        let wRowsU = weights.rows.map { UInt32($0) }
        let wColsU = weights.cols.map { UInt32($0) }

        guard
            let zBuf = device.makeBuffer(bytes: zF,
                length: MemoryLayout<Float>.stride * n, options: .storageModeShared),
            let rowsBuf = device.makeBuffer(bytes: wRowsU,
                length: MemoryLayout<UInt32>.stride * nnz, options: .storageModeShared),
            let colsBuf = device.makeBuffer(bytes: wColsU,
                length: MemoryLayout<UInt32>.stride * nnz, options: .storageModeShared),
            let valsBuf = device.makeBuffer(bytes: wValsF,
                length: MemoryLayout<Float>.stride * nnz, options: .storageModeShared)
        else { return nil }

        switch mode {
        case .scratchless:
            return runScratchless(z: z, zBuf: zBuf, rowsBuf: rowsBuf,
                colsBuf: colsBuf, valsBuf: valsBuf,
                weights: weights, nPermutations: nPermutations, seed: seed)
        case .indexed:
            return runIndexed(z: z, zBuf: zBuf, rowsBuf: rowsBuf,
                colsBuf: colsBuf, valsBuf: valsBuf,
                weights: weights, nPermutations: nPermutations, seed: seed)
        case .batched:
            return runBatched(z: z, zBuf: zBuf, rowsBuf: rowsBuf,
                colsBuf: colsBuf, valsBuf: valsBuf,
                weights: weights, nPermutations: nPermutations, seed: seed)
        }
    }

    // MARK: - Scratchless (n <= 256, single pass, no device scratch)

    private func runScratchless(
        z: [Double], zBuf: MTLBuffer, rowsBuf: MTLBuffer,
        colsBuf: MTLBuffer, valsBuf: MTLBuffer,
        weights: SparseWeights, nPermutations: Int, seed: UInt32
    ) -> PermutationResult? {
        let n = weights.n

        guard let nullBuf = device.makeBuffer(
            length: MemoryLayout<Float>.stride * nPermutations,
            options: .storageModeShared)
        else { return nil }

        var params = MoranParamsMetal(
            n: UInt32(n), nnz: UInt32(weights.nnz),
            nPerm: UInt32(nPermutations), seed: seed, permOffset: 0
        )
        guard let paramsBuf = device.makeBuffer(bytes: &params,
            length: MemoryLayout<MoranParamsMetal>.size,
            options: .storageModeShared)
        else { return nil }

        let start = Date()
        dispatch(pipeline: psoScratchless, nThreads: nPermutations,
                 buffers: [zBuf, rowsBuf, colsBuf, valsBuf, nullBuf, paramsBuf])
        let elapsed = Date().timeIntervalSince(start)

        return buildResult(z: z, weights: weights, nullBuf: nullBuf,
                           nPermutations: nPermutations, elapsed: elapsed)
    }

    // MARK: - Indexed (any n, single pass, uint32 perm buffer)

    private func runIndexed(
        z: [Double], zBuf: MTLBuffer, rowsBuf: MTLBuffer,
        colsBuf: MTLBuffer, valsBuf: MTLBuffer,
        weights: SparseWeights, nPermutations: Int, seed: UInt32
    ) -> PermutationResult? {
        let n = weights.n
        let indexBytes = nPermutations * n * MemoryLayout<UInt32>.stride
        let indexMB    = indexBytes / (1024 * 1024)
        print("[Metal] Index buffer: \(indexMB)MB")

        guard
            let nullBuf  = device.makeBuffer(
                length: MemoryLayout<Float>.stride * nPermutations,
                options: .storageModeShared),
            let permBuf  = device.makeBuffer(
                length: indexBytes,
                options: .storageModeShared)
        else { return nil }

        var params = MoranParamsMetalLarge(
            n: UInt32(n), nnz: UInt32(weights.nnz),
            nPerm: UInt32(nPermutations), seed: seed,
            permOffset: 0, useUint16: 0
        )
        guard let paramsBuf = device.makeBuffer(bytes: &params,
            length: MemoryLayout<MoranParamsMetalLarge>.size,
            options: .storageModeShared)
        else { return nil }

        let start = Date()
        dispatch(pipeline: psoIndexed, nThreads: nPermutations,
                 buffers: [zBuf, rowsBuf, colsBuf, valsBuf, nullBuf, paramsBuf, permBuf])
        let elapsed = Date().timeIntervalSince(start)

        return buildResult(z: z, weights: weights, nullBuf: nullBuf,
                           nPermutations: nPermutations, elapsed: elapsed)
    }

    // MARK: - Batched float scratch (fallback for very large n)

    private func runBatched(
        z: [Double], zBuf: MTLBuffer, rowsBuf: MTLBuffer,
        colsBuf: MTLBuffer, valsBuf: MTLBuffer,
        weights: SparseWeights, nPermutations: Int, seed: UInt32
    ) -> PermutationResult? {
        let n   = weights.n
        let maxBytes = MetalMoranEngine.maxMemoryBytes(for: device)
        let batch = max(1, maxBytes / (n * MemoryLayout<Float>.stride))
        let nBatches = (nPermutations + batch - 1) / batch
        print("[Metal] Batched fallback: \(nBatches) batches of ≤\(batch)")

        let scratchBytes = batch * n * MemoryLayout<Float>.stride
        guard
            let nullBatchBuf = device.makeBuffer(
                length: MemoryLayout<Float>.stride * batch,
                options: .storageModeShared),
            let scratchBuf   = device.makeBuffer(
                length: scratchBytes,
                options: .storageModeShared)
        else { return nil }

        var nullDist = [Double]()
        nullDist.reserveCapacity(nPermutations)
        let start = Date()

        for batchIdx in 0..<nBatches {
            let offset    = batchIdx * batch
            let thisBatch = min(batch, nPermutations - offset)
            var params = MoranParamsMetal(
                n: UInt32(n), nnz: UInt32(weights.nnz),
                nPerm: UInt32(thisBatch), seed: seed,
                permOffset: UInt32(offset)
            )
            guard let paramsBuf = device.makeBuffer(bytes: &params,
                length: MemoryLayout<MoranParamsMetal>.size,
                options: .storageModeShared)
            else { continue }
            dispatch(pipeline: psoBatched, nThreads: thisBatch,
                     buffers: [zBuf, rowsBuf, colsBuf, valsBuf,
                                nullBatchBuf, paramsBuf, scratchBuf])
            let ptr = nullBatchBuf.contents().bindMemory(
                to: Float.self, capacity: thisBatch)
            for i in 0..<thisBatch { nullDist.append(Double(ptr[i])) }
        }

        let elapsed  = Date().timeIntervalSince(start)
        let observed = moranI(z: z, weights: weights)
        let absObs   = abs(observed)
        let extreme  = nullDist.filter { abs($0) >= absObs }.count
        let pValue   = Double(extreme) / Double(nPermutations)
        return PermutationResult(observed: observed, nullDistribution: nullDist,
            pValueTwoSided: pValue, elapsedSeconds: elapsed,
            nThreads: device.maxThreadsPerThreadgroup.width)
    }

    // MARK: - Dispatch helper

    private func dispatch(pipeline: MTLComputePipelineState,
                          nThreads: Int, buffers: [MTLBuffer]) {
        guard
            let cmdBuf  = commandQueue.makeCommandBuffer(),
            let encoder = cmdBuf.makeComputeCommandEncoder()
        else { return }
        encoder.setComputePipelineState(pipeline)
        for (i, buf) in buffers.enumerated() {
            encoder.setBuffer(buf, offset: 0, index: i)
        }
        let tgs = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: nThreads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgs, height: 1, depth: 1)
        )
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    // MARK: - Result builder

    private func buildResult(
        z: [Double], weights: SparseWeights,
        nullBuf: MTLBuffer, nPermutations: Int, elapsed: Double
    ) -> PermutationResult {
        let ptr      = nullBuf.contents().bindMemory(to: Float.self,
                                                     capacity: nPermutations)
        let nullDist = (0..<nPermutations).map { Double(ptr[$0]) }
        let observed = moranI(z: z, weights: weights)
        let absObs   = abs(observed)
        let extreme  = nullDist.filter { abs($0) >= absObs }.count
        let pValue   = Double(extreme) / Double(nPermutations)
        return PermutationResult(
            observed: observed, nullDistribution: nullDist,
            pValueTwoSided: pValue, elapsedSeconds: elapsed,
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
    return engine.run(z: z, weights: weights,
                      nPermutations: nPermutations, seed: seed)
}
