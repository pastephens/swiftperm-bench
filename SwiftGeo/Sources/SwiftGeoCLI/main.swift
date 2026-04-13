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
    let nPerm   = args.count > 4 ? Int(args[4])    ?? 99999 : 99999
    let seed    = args.count > 5 ? UInt64(args[5]) ?? 12345 : 12345

    do {
        print("Reading z vector from \(zPath)...")
        let z = try readZVector(from: zPath)
        print("  n = \(z.count)")

        print("Reading sparse weights from \(wPath)...")
        let weights = try readSparseWeights(from: wPath)
        print("  n = \(weights.n), nnz = \(weights.nnz)")

        let cores = ProcessInfo.processInfo.activeProcessorCount
        print("  Active CPU cores: \(cores)")

        // --- Serial ---
        print("\n[Serial] Running \(nPerm) permutations...")
        let serial = moranPermutationTestSerial(
            z: z, weights: weights, nPermutations: nPerm, seed: seed
        )
        printResult(serial, label: "Serial")

        // --- Parallel ---
        print("\n[Parallel] Running \(nPerm) permutations...")
        let parallel = moranPermutationTest(
            z: z, weights: weights, nPermutations: nPerm, seed: seed
        )
        printResult(parallel, label: "Parallel")
        let cpuSpeedup = serial.elapsedSeconds / parallel.elapsedSeconds
        print("  CPU parallel speedup: \(String(format: "%.2f", cpuSpeedup))x")

        // --- Metal ---
        print("\n[Metal] Running \(nPerm) permutations...")
        if let metalResult = moranPermutationTestMetal(
            z: z, weights: weights,
            nPermutations: nPerm, seed: UInt32(seed & 0xFFFFFFFF)
        ) {
            printResult(metalResult, label: "Metal")
            let metalSpeedup = serial.elapsedSeconds / metalResult.elapsedSeconds
            let vsParallel   = parallel.elapsedSeconds / metalResult.elapsedSeconds
            print("  Metal speedup vs serial:   \(String(format: "%.2f", metalSpeedup))x")
            print("  Metal speedup vs parallel: \(String(format: "%.2f", vsParallel))x")

            // Write Metal result as canonical output
            print("\nWriting Metal results to \(outPath)...")
            try writeResults(metalResult, to: outPath)
        } else {
            print("  [Metal unavailable — falling back to parallel CPU result]")
            print("\nWriting parallel results to \(outPath)...")
            try writeResults(parallel, to: outPath)
        }

        print("Done.")

    } catch {
        fputs("Error: \(error)\n", stderr)
        exit(1)
    }
}

func printResult(_ result: PermutationResult, label: String) {
    let nPerm = result.nullDistribution.count
    let throughput = Double(nPerm) / result.elapsedSeconds
    print("  Observed Moran's I : \(String(format: "%.6f", result.observed))")
    print("  p-value (2-sided)  : \(String(format: "%.6f", result.pValueTwoSided))")
    print("  Elapsed            : \(String(format: "%.4f", result.elapsedSeconds))s")
    print("  Throughput         : \(String(format: "%.0f", throughput)) perm/s")
}

main()
