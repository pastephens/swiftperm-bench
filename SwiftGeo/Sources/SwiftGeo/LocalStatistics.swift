// LocalStatistics.swift
// Local Moran's I (LISA) — observed values and permutation-based inference
// for all n observations simultaneously.
//
// Formula: Iᵢ = z[i] * Σⱼ w[i,j] * z[j]  (i.e. z[i] * spatialLag[i])
//
// Permutation strategy: unconditional randomization — each permutation shuffles
// all n values, then computes the full vector of local statistics. Consistent
// with the global permutation infrastructure and with esda.moran.Moran_Local.

import Foundation

// MARK: - Result type

public struct LocalPermutationResult {
    public let observed: [Double]       // n local Moran's I values
    public let pValues: [Double]        // n two-sided p-values
    public let elapsedSeconds: Double
    public let nThreads: Int
}

// MARK: - Point estimates

/// Returns the n local Moran's I values: Iᵢ = z[i] * (W·z)[i]
public func localMoranI(z: [Double], weights: SparseWeights) -> [Double] {
    let wz = spatialLag(y: z, weights: weights)
    return (0..<weights.n).map { z[$0] * wz[$0] }
}

/// Returns the n local Geary's C values: Cᵢ = Σⱼ w[i,j] * (z[i] - z[j])²
/// All values are ≥ 0. Low Cᵢ = positive local autocorrelation; high Cᵢ = negative.
public func localGearysC(z: [Double], weights: SparseWeights) -> [Double] {
    var c = [Double](repeating: 0.0, count: weights.n)
    for k in 0..<weights.nnz {
        let i    = Int(weights.rows[k])
        let j    = Int(weights.cols[k])
        let diff = z[i] - z[j]
        c[i] += weights.values[k] * diff * diff
    }
    return c
}

// MARK: - Serial permutation test

public func localPermutationTestSerial(
    z: [Double],
    weights: SparseWeights,
    nPermutations: Int = 9999,
    seed: UInt64 = 12345
) -> LocalPermutationResult {
    let n      = weights.n
    let nnz    = weights.nnz
    let start  = Date()

    let observed = localMoranI(z: z, weights: weights)
    let absObs   = observed.map(abs)

    var counts = [Int](repeating: 0, count: n)
    var wzPerm = [Double](repeating: 0.0, count: n)
    var zPerm  = z
    var rng    = SeededRNG(seed: seed)

    for _ in 0..<nPermutations {
        zPerm = z
        fisherYatesShuffle(&zPerm, rng: &rng)

        for i in 0..<n { wzPerm[i] = 0.0 }
        for k in 0..<nnz {
            wzPerm[Int(weights.rows[k])] += weights.values[k] * zPerm[Int(weights.cols[k])]
        }
        for i in 0..<n {
            if abs(zPerm[i] * wzPerm[i]) >= absObs[i] { counts[i] += 1 }
        }
    }

    let elapsed  = Date().timeIntervalSince(start)
    let pValues  = counts.map { Double($0) / Double(nPermutations) }
    return LocalPermutationResult(observed: observed, pValues: pValues,
                                  elapsedSeconds: elapsed, nThreads: 1)
}

// MARK: - Parallel permutation test

public func localPermutationTest(
    z: [Double],
    weights: SparseWeights,
    nPermutations: Int = 9999,
    seed: UInt64 = 12345
) -> LocalPermutationResult {
    let n        = weights.n
    let nnz      = weights.nnz
    let start    = Date()

    let observed = localMoranI(z: z, weights: weights)
    let absObs   = observed.map(abs)

    let nThreads  = ProcessInfo.processInfo.activeProcessorCount
    let chunkSize = (nPermutations + nThreads - 1) / nThreads

    // Thread-local count arrays — no atomic ops needed
    var threadCounts = [[Int]](repeating: [Int](repeating: 0, count: n), count: nThreads)

    threadCounts.withUnsafeMutableBufferPointer { threadBuf in
        DispatchQueue.concurrentPerform(iterations: nThreads) { threadIdx in
            let pStart = threadIdx * chunkSize
            let pEnd   = min(pStart + chunkSize, nPermutations)
            guard pStart < pEnd else { return }

            var localCounts = [Int](repeating: 0, count: n)
            var wzPerm = [Double](repeating: 0.0, count: n)
            var zPerm  = z
            // Distinct seed per thread for deterministic parallel results
            let threadSeed = seed &+ UInt64(threadIdx) &* 6364136223846793005
            var rng = SeededRNG(seed: threadSeed)

            for _ in pStart..<pEnd {
                zPerm = z
                fisherYatesShuffle(&zPerm, rng: &rng)

                for i in 0..<n { wzPerm[i] = 0.0 }
                for k in 0..<nnz {
                    wzPerm[Int(weights.rows[k])] += weights.values[k] * zPerm[Int(weights.cols[k])]
                }
                for i in 0..<n {
                    if abs(zPerm[i] * wzPerm[i]) >= absObs[i] { localCounts[i] += 1 }
                }
            }
            threadBuf[threadIdx] = localCounts
        }
    }

    // Merge thread-local counts
    var counts = [Int](repeating: 0, count: n)
    for tc in threadCounts { for i in 0..<n { counts[i] += tc[i] } }

    let elapsed = Date().timeIntervalSince(start)
    let pValues = counts.map { Double($0) / Double(nPermutations) }
    return LocalPermutationResult(observed: observed, pValues: pValues,
                                  elapsedSeconds: elapsed, nThreads: nThreads)
}

// MARK: - Local Geary's C permutation tests

public func localGearysCPermutationTestSerial(
    z: [Double],
    weights: SparseWeights,
    nPermutations: Int = 9999,
    seed: UInt64 = 12345
) -> LocalPermutationResult {
    let n     = weights.n
    let nnz   = weights.nnz
    let start = Date()

    let observed = localGearysC(z: z, weights: weights)

    var counts = [Int](repeating: 0, count: n)
    var cPerm  = [Double](repeating: 0.0, count: n)
    var zPerm  = z
    var rng    = SeededRNG(seed: seed)

    for _ in 0..<nPermutations {
        zPerm = z
        fisherYatesShuffle(&zPerm, rng: &rng)

        for i in 0..<n { cPerm[i] = 0.0 }
        for k in 0..<nnz {
            let i    = Int(weights.rows[k])
            let j    = Int(weights.cols[k])
            let diff = zPerm[i] - zPerm[j]
            cPerm[i] += weights.values[k] * diff * diff
        }
        for i in 0..<n {
            if cPerm[i] >= observed[i] { counts[i] += 1 }
        }
    }

    let elapsed = Date().timeIntervalSince(start)
    let pValues = counts.map { Double($0) / Double(nPermutations) }
    return LocalPermutationResult(observed: observed, pValues: pValues,
                                  elapsedSeconds: elapsed, nThreads: 1)
}

public func localGearysCPermutationTest(
    z: [Double],
    weights: SparseWeights,
    nPermutations: Int = 9999,
    seed: UInt64 = 12345
) -> LocalPermutationResult {
    let n        = weights.n
    let nnz      = weights.nnz
    let start    = Date()

    let observed  = localGearysC(z: z, weights: weights)
    let nThreads  = ProcessInfo.processInfo.activeProcessorCount
    let chunkSize = (nPermutations + nThreads - 1) / nThreads

    var threadCounts = [[Int]](repeating: [Int](repeating: 0, count: n), count: nThreads)

    threadCounts.withUnsafeMutableBufferPointer { threadBuf in
        DispatchQueue.concurrentPerform(iterations: nThreads) { threadIdx in
            let pStart = threadIdx * chunkSize
            let pEnd   = min(pStart + chunkSize, nPermutations)
            guard pStart < pEnd else { return }

            var localCounts = [Int](repeating: 0, count: n)
            var cPerm  = [Double](repeating: 0.0, count: n)
            var zPerm  = z
            let threadSeed = seed &+ UInt64(threadIdx) &* 6364136223846793005
            var rng = SeededRNG(seed: threadSeed)

            for _ in pStart..<pEnd {
                zPerm = z
                fisherYatesShuffle(&zPerm, rng: &rng)

                for i in 0..<n { cPerm[i] = 0.0 }
                for k in 0..<nnz {
                    let i    = Int(weights.rows[k])
                    let j    = Int(weights.cols[k])
                    let diff = zPerm[i] - zPerm[j]
                    cPerm[i] += weights.values[k] * diff * diff
                }
                for i in 0..<n {
                    if cPerm[i] >= observed[i] { localCounts[i] += 1 }
                }
            }
            threadBuf[threadIdx] = localCounts
        }
    }

    var counts = [Int](repeating: 0, count: n)
    for tc in threadCounts { for i in 0..<n { counts[i] += tc[i] } }

    let elapsed = Date().timeIntervalSince(start)
    let pValues = counts.map { Double($0) / Double(nPermutations) }
    return LocalPermutationResult(observed: observed, pValues: pValues,
                                  elapsedSeconds: elapsed, nThreads: nThreads)
}
