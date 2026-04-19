// Statistics.swift
// Permutation statistic functions and the PermutationStatistic typealias.
// Each function maps (permuted z, weights) → a scalar test statistic.

import Accelerate

/// A permutation statistic: maps (permuted z, weights) → scalar.
public typealias PermutationStatistic = ([Double], SparseWeights) -> Double

/// Moran's I: I = (1/n) z'Wz
/// z must be standardized (mean=0, std=1).
public func moranI(z: [Double], weights: SparseWeights) -> Double {
    var wz = [Double](repeating: 0.0, count: weights.n)
    for k in 0..<weights.nnz {
        wz[Int(weights.rows[k])] += weights.values[k] * z[Int(weights.cols[k])]
    }
    var result = 0.0
    vDSP_dotprD(z, 1, wz, 1, &result, vDSP_Length(weights.n))
    return result / Double(weights.n)
}

/// Geary's C: C = (n-1)/(2n²) * Σₖ wₖ(z[rowₖ] - z[colₖ])²
/// z must be standardized (mean=0, std=1), so Σz²=n and S₀=n for row-standardized W.
/// C < 1 indicates positive spatial autocorrelation; C > 1 indicates negative.
public func gearysC(z: [Double], weights: SparseWeights) -> Double {
    var sum = 0.0
    for k in 0..<weights.nnz {
        let diff = z[Int(weights.rows[k])] - z[Int(weights.cols[k])]
        sum += weights.values[k] * diff * diff
    }
    let n = Double(weights.n)
    return (n - 1.0) / (2.0 * n * n) * sum
}

/// Getis-Ord G: G = Σₖ wₖ z[rowₖ] z[colₖ] / (Σᵢzᵢ)²
/// z must be positive (non-standardized). The denominator is permutation-invariant.
/// CPU-only (no Metal path); denominator is computed on the original z at call time.
public func getisOrdG(z: [Double], weights: SparseWeights) -> Double {
    var numerator = 0.0
    for k in 0..<weights.nnz {
        numerator += weights.values[k] * z[Int(weights.rows[k])] * z[Int(weights.cols[k])]
    }
    var denom = 0.0
    for v in z { denom += v }
    guard denom != 0 else { return 0 }
    return numerator / (denom * denom)
}

/// Join count: JC = Σₖ wₖ z[rowₖ] z[colₖ]  (z ∈ {0,1} binary)
/// Counts same-class adjacent pairs weighted by W. CPU-only (no Metal path).
public func joinCount(z: [Double], weights: SparseWeights) -> Double {
    var sum = 0.0
    for k in 0..<weights.nnz {
        sum += weights.values[k] * z[Int(weights.rows[k])] * z[Int(weights.cols[k])]
    }
    return sum
}

// MARK: - Spatial lag

/// Serial spatial lag: wy = W · y  (COO sparse matrix-vector multiply)
public func spatialLag(y: [Double], weights: SparseWeights) -> [Double] {
    var wy = [Double](repeating: 0.0, count: weights.n)
    for k in 0..<weights.nnz {
        wy[Int(weights.rows[k])] += weights.values[k] * y[Int(weights.cols[k])]
    }
    return wy
}

/// Parallel spatial lag: converts COO → CSR internally, then parallelizes over rows.
/// Each row's dot product is independent, so no write conflicts.
public func spatialLagParallel(y: [Double], weights: SparseWeights) -> [Double] {
    let n   = weights.n
    let nnz = weights.nnz

    // COO → CSR: count entries per row, build row pointers
    var rowCounts = [Int](repeating: 0, count: n)
    for k in 0..<nnz { rowCounts[Int(weights.rows[k])] += 1 }
    var rowPtrs = [Int](repeating: 0, count: n + 1)
    for i in 0..<n { rowPtrs[i + 1] = rowPtrs[i] + rowCounts[i] }

    var csrCols = [Int32](repeating: 0,   count: nnz)
    var csrVals = [Double](repeating: 0.0, count: nnz)
    var cursor  = rowPtrs  // working offsets into each row's CSR slice
    for k in 0..<nnz {
        let row = Int(weights.rows[k])
        csrCols[cursor[row]] = weights.cols[k]
        csrVals[cursor[row]] = weights.values[k]
        cursor[row] += 1
    }

    var wy = [Double](repeating: 0.0, count: n)
    wy.withUnsafeMutableBufferPointer { buf in
        DispatchQueue.concurrentPerform(iterations: n) { i in
            var s = 0.0
            for pos in rowPtrs[i]..<rowPtrs[i + 1] {
                s += csrVals[pos] * y[Int(csrCols[pos])]
            }
            buf[i] = s
        }
    }
    return wy
}
