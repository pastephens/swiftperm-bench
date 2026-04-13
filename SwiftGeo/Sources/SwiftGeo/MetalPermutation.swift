// MetalPermutation.swift
// Moran's I permutation test via Metal GPU compute.

import Foundation
import Metal

struct MoranParamsMetal {
    var n: UInt32
    var nnz: UInt32
    var nPerm: UInt32
    var seed: UInt32
}

public class MetalMoranEngine {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    public init?() {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            print("[Metal] No Metal device available.")
            return nil
        }
        guard let queue = dev.makeCommandQueue() else {
            print("[Metal] Could not create command queue.")
            return nil
        }

        // Try multiple library loading strategies for both library and CLI targets
        let library: MTLLibrary? = {
            // Strategy 1: Bundle.module (works for library targets in SwiftPM)
            if let lib = try? dev.makeDefaultLibrary(bundle: Bundle.module) {
                return lib
            }
            // Strategy 2: main bundle (works when embedded in an app)
            if let lib = try? dev.makeDefaultLibrary(bundle: Bundle.main) {
                return lib
            }
            // Strategy 3: locate .metallib next to the executable
            let execURL = URL(fileURLWithPath: CommandLine.arguments[0])
                .deletingLastPathComponent()
            let candidates = [
                execURL.appendingPathComponent("default.metallib"),
                execURL.appendingPathComponent("SwiftGeo_SwiftGeo.bundle/default.metallib"),
                execURL.appendingPathComponent("../lib/SwiftGeo_SwiftGeo.bundle/default.metallib"),
            ]
            for url in candidates {
                if let lib = try? dev.makeLibrary(URL: url) {
                    print("[Metal] Loaded library from: \(url.path)")
                    return lib
                }
            }
            // Strategy 4: compile from source at runtime (fallback for development)
            let metalSrc = """
            #include <metal_stdlib>
            using namespace metal;
            struct MoranParams { uint n; uint nnz; uint nPerm; uint seed; };
            inline uint64_t xorshift64(uint64_t s) {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
            }
            kernel void moranPermutation(
                device const float* z        [[ buffer(0) ]],
                device const uint*  wRows    [[ buffer(1) ]],
                device const uint*  wCols    [[ buffer(2) ]],
                device const float* wVals    [[ buffer(3) ]],
                device float*       nullDist [[ buffer(4) ]],
                device const MoranParams& params [[ buffer(5) ]],
                device float*       zPerms   [[ buffer(6) ]],
                uint tid [[ thread_position_in_grid ]]
            ) {
                if (tid >= params.nPerm) return;
                const uint n = params.n, nnz = params.nnz;
                device float* zPerm = zPerms + (tid * n);
                for (uint i = 0; i < n; i++) zPerm[i] = z[i];
                uint64_t state = (uint64_t)params.seed + (uint64_t)tid * 6364136223846793005ULL;
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
            """
            if let lib = try? dev.makeLibrary(source: metalSrc, options: nil) {
                print("[Metal] Compiled shader from source (development fallback)")
                return lib
            }
            return nil
        }()

        guard let lib = library,
              let function = lib.makeFunction(name: "moranPermutation"),
              let pso = try? dev.makeComputePipelineState(function: function)
        else {
            print("[Metal] Could not load or compile moranPermutation kernel.")
            return nil
        }

        self.device = dev
        self.commandQueue = queue
        self.pipeline = pso
        print("[Metal] Device: \(dev.name)")
    }

    public func run(
        z: [Double],
        weights: SparseWeights,
        nPermutations: Int,
        seed: UInt32 = 12345
    ) -> PermutationResult? {

        let n   = weights.n
        let nnz = weights.nnz

        let zF     = z.map { Float($0) }
        let wValsF = weights.values.map { Float($0) }
        let wRowsU = weights.rows.map { UInt32($0) }
        let wColsU = weights.cols.map { UInt32($0) }

        let scratchBytes = nPermutations * n * MemoryLayout<Float>.stride
        let scratchMB = scratchBytes / (1024 * 1024)
        if scratchBytes > 512 * 1024 * 1024 {
            print("[Metal] Scratch buffer \(scratchMB)MB exceeds limit — use batched mode for n=\(n).")
            return nil
        }

        guard
            let zBuf = device.makeBuffer(bytes: zF,
                length: MemoryLayout<Float>.stride * n, options: .storageModeShared),
            let rowsBuf = device.makeBuffer(bytes: wRowsU,
                length: MemoryLayout<UInt32>.stride * nnz, options: .storageModeShared),
            let colsBuf = device.makeBuffer(bytes: wColsU,
                length: MemoryLayout<UInt32>.stride * nnz, options: .storageModeShared),
            let valsBuf = device.makeBuffer(bytes: wValsF,
                length: MemoryLayout<Float>.stride * nnz, options: .storageModeShared),
            let nullBuf = device.makeBuffer(
                length: MemoryLayout<Float>.stride * nPermutations, options: .storageModeShared),
            let scratchBuf = device.makeBuffer(
                length: scratchBytes, options: .storageModeShared)
        else {
            print("[Metal] Buffer allocation failed.")
            return nil
        }

        var params = MoranParamsMetal(
            n: UInt32(n), nnz: UInt32(nnz),
            nPerm: UInt32(nPermutations), seed: seed
        )
        guard let paramsBuf = device.makeBuffer(bytes: &params,
            length: MemoryLayout<MoranParamsMetal>.size, options: .storageModeShared)
        else { return nil }

        guard
            let cmdBuf  = commandQueue.makeCommandBuffer(),
            let encoder = cmdBuf.makeComputeCommandEncoder()
        else { return nil }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(zBuf,       offset: 0, index: 0)
        encoder.setBuffer(rowsBuf,    offset: 0, index: 1)
        encoder.setBuffer(colsBuf,    offset: 0, index: 2)
        encoder.setBuffer(valsBuf,    offset: 0, index: 3)
        encoder.setBuffer(nullBuf,    offset: 0, index: 4)
        encoder.setBuffer(paramsBuf,  offset: 0, index: 5)
        encoder.setBuffer(scratchBuf, offset: 0, index: 6)

        let threadGroupSize = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: nPermutations, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()

        let start = Date()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        let elapsed = Date().timeIntervalSince(start)

        let nullPtr  = nullBuf.contents().bindMemory(to: Float.self, capacity: nPermutations)
        let nullDist = (0..<nPermutations).map { Double(nullPtr[$0]) }
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

public func moranPermutationTestMetal(
    z: [Double],
    weights: SparseWeights,
    nPermutations: Int = 99999,
    seed: UInt32 = 12345
) -> PermutationResult? {
    guard let engine = MetalMoranEngine() else { return nil }
    return engine.run(z: z, weights: weights, nPermutations: nPermutations, seed: seed)
}
