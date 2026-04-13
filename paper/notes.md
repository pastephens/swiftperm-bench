# Paper Notes — Key Framing and Arguments

## Central contribution (computational, not statistical)

The method does not change. The practical barrier to using it correctly does.

Permutation inference for spatial autocorrelation is well-established. The statistical
validity of 999 vs 99,999 permutations is not meaningfully different for most
single-test applications with clear signals. The contribution here is not about
making the test more precise — it is about making the test computationally
accessible at scales and iteration speeds that currently force methodological
compromises.

## The right framing: enabling iteration at scale

**Not:** "run more permutations on the same analysis for higher precision"

**Yes:** "make the analysis feasible at all on larger datasets, and enable
researchers to iterate through many model specifications in a reasonable session"

At n=10,000 with NumPy, 99,999 permutations takes ~17 seconds. Swift parallel
takes ~6 seconds. Metal (where applicable) takes under 1 second at small n.
This is not about p-value precision — it is about whether a researcher can
run 50 model specifications in a morning rather than overnight.

## When more permutations DO matter (narrow but real)

Two genuine cases where 99,999 > 9,999 matters statistically:

1. **Very small p-values / multiple testing correction**
   - Minimum resolvable p with 999 perms: ~0.001
   - Minimum resolvable p with 99,999 perms: ~0.00001
   - Bonferroni-adjusted thresholds in spatial epidemiology with many regions
     commonly require resolution below 0.001 — unreachable with standard counts.

2. **Near the significance threshold**
   - Standard error of p-value estimate near p=0.05:
     - 999 perms:    ±0.007
     - 9,999 perms:  ±0.002
     - 99,999 perms: ±0.0007
   - When observed statistic falls near α=0.05, more permutations reduce
     Monte Carlo noise and give a more stable binary inference decision.

## What permutation tests do vs. confidence intervals

Permutation tests answer: "how often would we see a statistic this extreme
under spatial randomness?" → p-value, hypothesis test.

Confidence intervals answer: "what range of values is consistent with the data?"
→ requires bootstrap (resample with replacement), not permutation (resample
without replacement).

esda's Moran class does NOT produce a CI. It produces:
- p_sim: permutation p-value
- EI_sim, VI_sim, seI_sim: moments of the null distribution
- z_sim, p_z_sim: normal approximation from simulation moments

A bootstrap CI on Moran's I would require a separate operation and a separate
paper argument (that researchers should estimate uncertainty around I itself,
not just test for significance). That is a valid future direction but is not
what this benchmark establishes.

## The p-value precision argument in numbers

At n=5,000, Swift parallel achieves ~34,000 perm/s.

| Permutations | Time    | Min resolvable p |
|---|---|---|
| 999          | 29ms    | 0.001            |
| 9,999        | 0.3s    | 0.0001           |
| 99,999       | 3.2s    | 0.00001          |
| 999,999      | ~32s    | 0.000001         |

With Metal at n=49: 999,999 permutations in ~67ms.

## Suggested paper structure (rough)

1. Introduction — the computational barrier to rigorous permutation inference
2. Background — Moran's I, permutation testing, Apple silicon architecture
3. Implementation — four implementations (NumPy, Numba, Swift CPU, Swift Metal)
4. Results — benchmark across n=49 to n=10,000
5. Discussion — iteration at scale as the primary contribution; p-value
   precision as secondary; limitations (batch dispatch at large n, single
   statistic, macOS only)
6. Conclusion — path to generalisation (arbitrary statistics, Python binding,
   R package)

## Hardware context

All results: Apple M3, 8GB unified memory, 8 CPU cores.
Metal peak: 14.85M perm/s (77.8x over NumPy) at n=49, scratchless stack shader.
Swift parallel peak: 349k perm/s at n=500 (4.5x over NumPy, 2.0x over Numba).
