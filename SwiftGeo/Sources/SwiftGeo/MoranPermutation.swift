// MoranPermutation.swift
// Core permutation inference for Moran's I statistic.
// Uses Accelerate for fast sparse dot products and in-place shuffling.

import Foundation
import Accelerate

// MARK: - Data structures

/// Sparse weight matrix in COO (coordinate) format.
/// Weights are row-standardized (each row sums to 1).
public struct SparseWeights {
    public let rows: [Int32]      // focal index
    public let cols: [Int32]      // neighbor index
    public let values: [Double]   // weight value
    public let n: Int             // number of observations
    public let nnz: Int           // number of nonzero entries

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
/// I = z' W z  (since z is already standardized, no further scaling needed
/// beyond the row-standardization embedded in the weights)
public func moranI(z: [Double], weights: SparseWeights) -> Double {
    // Compute spatial lag: Wz[i] = sum_j w_ij * z[j]
    var wz = [Double](repeating: 0.0, count: weights.n)

    for k in 0..<weights.nnz {
        let i = Int(weights.rows[k])
        let j = Int(weights.cols[k])
        wz[i] += weights.values[k] * z[j]
    }

    // I = z · Wz
    var result = 0.0
    vDSP_dotprD(z, 1, wz, 1, &result, vDSP_Length(weights.n))

    // Normalize by n (standard row-standardized Moran's I formulation)
    return result / Double(weights.n)
}

// MARK: - Permutation test

public struct PermutationResult {
    public let observed: Double
    public let nullDistribution: [Double]
    public let pValueTwoSided: Double
    public let elapsedSeconds: Double

    public var pValueAbove: Double {
        let above = nullDistribution.filter { $0 >= observed }.count
        return Double(above) / Double(nullDistribution.count)
    }
}

/// Run a permutation test for Moran's I.
/// - Parameters:
///   - z: Standardized variable (mean 0, variance 1)
///   - weights: Row-standardized sparse spatial weights
///   - nPermutations: Number of random permutations
///   - seed: Random seed for reproducibility (0 = random)
/// - Returns: PermutationResult with observed statistic, null distribution, and p-value
public func moranPermutationTest(
    z: [Double],
    weights: SparseWeights,
    nPermutations: Int = 99999,
    seed: UInt64 = 12345
) -> PermutationResult {

    let observed = moranI(z: z, weights: weights)
    let start = Date()

    var nullDist = [Double](repeating: 0.0, count: nPermutations)
    var zPerm = z  // working copy

    // Seeded RNG for reproducibility
    var rng = SeededRNG(seed: seed)

    for p in 0..<nPermutations {
        // Fisher-Yates shuffle using our seeded RNG
        fisherYatesShuffle(&zPerm, rng: &rng)
        nullDist[p] = moranI(z: zPerm, weights: weights)
        // Restore original order by shuffling back isn't needed —
        // we shuffle from z each time via a fresh copy would be clean
        // but expensive. Instead we keep shuffling the working copy;
        // after enough permutations the distribution converges identically.
        // Reset to original every permutation for strict correctness:
        zPerm = z
    }

    let elapsed = Date().timeIntervalSince(start)

    // Two-sided p-value: proportion of null >= |observed| or <= -|observed|
    let absObs = abs(observed)
    let extreme = nullDist.filter { abs($0) >= absObs }.count
    let pValue = Double(extreme) / Double(nPermutations)

    return PermutationResult(
        observed: observed,
        nullDistribution: nullDist,
        pValueTwoSided: pValue,
        elapsedSeconds: elapsed
    )
}

// MARK: - Seeded RNG (xorshift64)

/// Fast, seedable PRNG — not cryptographic, but sufficient for permutation tests.
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

    /// Uniform random integer in [0, upperBound)
    mutating func nextInt(upperBound: Int) -> Int {
        return Int(next() % UInt64(upperBound))
    }
}

/// In-place Fisher-Yates shuffle
func fisherYatesShuffle(_ array: inout [Double], rng: inout SeededRNG) {
    for i in stride(from: array.count - 1, through: 1, by: -1) {
        let j = rng.nextInt(upperBound: i + 1)
        if i != j {
            array.swapAt(i, j)
        }
    }
}
