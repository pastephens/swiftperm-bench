// MoranPermutation.swift
// Generic permutation test infrastructure (serial + parallel).
// Statistic functions are in Statistics.swift.

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

// MARK: - Generic permutation test (serial)

public func permutationTestSerial(
    z: [Double],
    weights: SparseWeights,
    nPermutations: Int = 99999,
    seed: UInt64 = 12345,
    statistic: PermutationStatistic
) -> PermutationResult {
    let observed = statistic(z, weights)
    let start = Date()

    var nullDist = [Double](repeating: 0.0, count: nPermutations)
    var zPerm = z
    var rng = SeededRNG(seed: seed)

    for p in 0..<nPermutations {
        zPerm = z
        fisherYatesShuffle(&zPerm, rng: &rng)
        nullDist[p] = statistic(zPerm, weights)
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

// MARK: - Generic permutation test (parallel via DispatchQueue.concurrentPerform)

public func permutationTest(
    z: [Double],
    weights: SparseWeights,
    nPermutations: Int = 99999,
    seed: UInt64 = 12345,
    statistic: PermutationStatistic
) -> PermutationResult {
    let observed = statistic(z, weights)
    let start = Date()

    var nullDist = [Double](repeating: 0.0, count: nPermutations)

    // Per-permutation seed: mix global seed with index for deterministic results
    // regardless of thread scheduling order.
    nullDist.withUnsafeMutableBufferPointer { buffer in
        DispatchQueue.concurrentPerform(iterations: nPermutations) { p in
            let permSeed = seed &+ UInt64(p) &* 6364136223846793005
            var rng = SeededRNG(seed: permSeed)
            var zPerm = z
            fisherYatesShuffle(&zPerm, rng: &rng)
            buffer[p] = statistic(zPerm, weights)
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

// MARK: - Moran's I convenience wrappers (backward compat)

public func moranPermutationTestSerial(
    z: [Double],
    weights: SparseWeights,
    nPermutations: Int = 99999,
    seed: UInt64 = 12345
) -> PermutationResult {
    return permutationTestSerial(z: z, weights: weights,
        nPermutations: nPermutations, seed: seed, statistic: moranI)
}

public func moranPermutationTest(
    z: [Double],
    weights: SparseWeights,
    nPermutations: Int = 99999,
    seed: UInt64 = 12345
) -> PermutationResult {
    return permutationTest(z: z, weights: weights,
        nPermutations: nPermutations, seed: seed, statistic: moranI)
}
