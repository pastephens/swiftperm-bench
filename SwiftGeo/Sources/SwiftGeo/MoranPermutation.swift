// MoranPermutation.swift
// Core permutation inference for Moran's I statistic.
// Uses Accelerate for fast sparse dot products and concurrent permutation loop.

import Foundation
import Accelerate

// MARK: - Data structures

/// Sparse weight matrix in COO (coordinate) format.
/// Weights are row-standardized (each row sums to 1).
public struct SparseWeights {
    public let rows: [Int32]
    public let cols: [Int32]
    public let values: [Double]
    public let n: Int
    public let nnz: Int

    public init(rows: [Int32], cols: [Int32], values: [Double], n: Int) {
        self.rows = rows
        self.cols = cols
        self.values = values
        self.n = n
        self.nnz = values.count
    }
}

// MARK: - Moran's I

/// Compute Moran's I for a standardized variable z given sparse weights W.
/// I = (1/n) * z' W z
public func moranI(z: [Double], weights: SparseWeights) -> Double {
    var wz = [Double](repeating: 0.0, count: weights.n)
    for k in 0..<weights.nnz {
        wz[Int(weights.rows[k])] += weights.values[k] * z[Int(weights.cols[k])]
    }
    var result = 0.0
    vDSP_dotprD(z, 1, wz, 1, &result, vDSP_Length(weights.n))
    return result / Double(weights.n)
}

// MARK: - Permutation result

public struct PermutationResult {
    public let observed: Double
    public let nullDistribution: [Double]
    public let pValueTwoSided: Double
    public let elapsedSeconds: Double
    public let nThreads: Int

    public var pValueAbove: Double {
        let above = nullDistribution.filter { $0 >= observed }.count
        return Double(above) / Double(nullDistribution.count)
    }
}

// MARK: - Seeded RNG (xorshift64)

/// Fast, seedable PRNG — not cryptographic, sufficient for permutation tests.
struct SeededRNG {
    var state: UInt64

    init(seed: UInt64) {
        self.state = seed == 0 ? 6364136223846793005 : seed
    }

    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }

    mutating func nextInt(upperBound: Int) -> Int {
        return Int(next() % UInt64(upperBound))
    }
}

/// In-place Fisher-Yates shuffle
func fisherYatesShuffle(_ array: inout [Double], rng: inout SeededRNG) {
    for i in stride(from: array.count - 1, through: 1, by: -1) {
        let j = rng.nextInt(upperBound: i + 1)
        if i != j { array.swapAt(i, j) }
    }
}

// MARK: - Permutation test (serial)

public func moranPermutationTestSerial(
    z: [Double],
    weights: SparseWeights,
    nPermutations: Int = 99999,
    seed: UInt64 = 12345
) -> PermutationResult {
    let observed = moranI(z: z, weights: weights)
    let start = Date()

    var nullDist = [Double](repeating: 0.0, count: nPermutations)
    var zPerm = z
    var rng = SeededRNG(seed: seed)

    for p in 0..<nPermutations {
        zPerm = z
        fisherYatesShuffle(&zPerm, rng: &rng)
        nullDist[p] = moranI(z: zPerm, weights: weights)
    }

    let elapsed = Date().timeIntervalSince(start)
    let absObs = abs(observed)
    let extreme = nullDist.filter { abs($0) >= absObs }.count
    let pValue = Double(extreme) / Double(nPermutations)

    return PermutationResult(
        observed: observed,
        nullDistribution: nullDist,
        pValueTwoSided: pValue,
        elapsedSeconds: elapsed,
        nThreads: 1
    )
}

// MARK: - Permutation test (parallel via DispatchQueue.concurrentPerform)

public func moranPermutationTest(
    z: [Double],
    weights: SparseWeights,
    nPermutations: Int = 99999,
    seed: UInt64 = 12345
) -> PermutationResult {
    let observed = moranI(z: z, weights: weights)
    let start = Date()

    // Pre-allocate output buffer — each index written by exactly one thread
    var nullDist = [Double](repeating: 0.0, count: nPermutations)

    // Each permutation gets its own seed derived from the global seed + index,
    // so results are deterministic regardless of thread scheduling order.
    nullDist.withUnsafeMutableBufferPointer { buffer in
        DispatchQueue.concurrentPerform(iterations: nPermutations) { p in
            // Per-permutation seed: mix global seed with permutation index
            let permSeed = seed &+ UInt64(p) &* 6364136223846793005
            var rng = SeededRNG(seed: permSeed)
            var zPerm = z
            fisherYatesShuffle(&zPerm, rng: &rng)
            buffer[p] = moranI(z: zPerm, weights: weights)
        }
    }

    let elapsed = Date().timeIntervalSince(start)
    let absObs = abs(observed)
    let extreme = nullDist.filter { abs($0) >= absObs }.count
    let pValue = Double(extreme) / Double(nPermutations)

    let nThreads = ProcessInfo.processInfo.activeProcessorCount

    return PermutationResult(
        observed: observed,
        nullDistribution: nullDist,
        pValueTwoSided: pValue,
        elapsedSeconds: elapsed,
        nThreads: nThreads
    )
}
