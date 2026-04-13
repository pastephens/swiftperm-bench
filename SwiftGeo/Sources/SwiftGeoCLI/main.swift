// main.swift — SwiftGeoCLI
// Usage: SwiftGeoCLI <z_path> <weights_path> <output_path> [n_permutations] [seed]

import Foundation
import SwiftGeo

func main() {
    let args = CommandLine.arguments

    guard args.count >= 4 else {
        fputs("Usage: SwiftGeoCLI <z_path> <weights_path> <output_path> [n_permutations] [seed]\n", stderr)
        exit(1)
    }

    let zPath   = args[1]
    let wPath   = args[2]
    let outPath = args[3]
    let nPerm   = args.count > 4 ? Int(args[4]) ?? 99999 : 99999
    let seed    = args.count > 5 ? UInt64(args[5]) ?? 12345 : 12345

    do {
        print("Reading z vector from \(zPath)...")
        let z = try readZVector(from: zPath)
        print("  n = \(z.count)")

        print("Reading sparse weights from \(wPath)...")
        let weights = try readSparseWeights(from: wPath)
        print("  n = \(weights.n), nnz = \(weights.nnz)")

        print("Running \(nPerm) permutations (seed=\(seed))...")
        let result = moranPermutationTest(
            z: z,
            weights: weights,
            nPermutations: nPerm,
            seed: seed
        )

        print("  Observed Moran's I : \(String(format: "%.6f", result.observed))")
        print("  p-value (2-sided)  : \(String(format: "%.6f", result.pValueTwoSided))")
        print("  Elapsed            : \(String(format: "%.3f", result.elapsedSeconds))s")
        print("  Throughput         : \(String(format: "%.0f", Double(nPerm) / result.elapsedSeconds)) perm/s")

        print("Writing results to \(outPath)...")
        try writeResults(result, to: outPath)
        print("Done.")

    } catch {
        fputs("Error: \(error)\n", stderr)
        exit(1)
    }
}

main()
