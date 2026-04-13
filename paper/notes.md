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

---

## Connection to Rey, Stephens & Laura (2017)

The prior paper is a direct intellectual predecessor. Key parallel arguments:

### Structural parallel

Rey et al. (2017) studied the **speed/accuracy tradeoff in optimal map classification**
(Fisher-Jenks) in big data settings, using Monte Carlo simulation to quantify when
sampling-based approximations are acceptable substitutes for full enumeration.

This work studies the **speed/feasibility tradeoff in permutation inference** (Moran's I)
across hardware backends, quantifying when GPU/Swift acceleration enables full
enumeration at scales where Python forces suboptimal permutation counts.

Both papers are fundamentally about: *when does the computational constraint force
a methodological compromise, and how can that constraint be lifted?*

### Three-paper lineage

There are now three papers in the lineage, not two:

1. **Rey, Anselin, Pahle, Kang & Stephens (2013)** — *Parallel optimal choropleth map
   classification in PySAL*. IJGIS. Cited by 32.
   "Refactoring a spatial analysis library to support parallelization... parallel
   implementations of Fisher-Jenks using a multi-core, single desktop environment."
   → The first paper: parallelizing spatial computation on CPU, desktop hardware.

2. **Rey, Stephens & Laura (2017)** — *Sampling and full enumeration strategies
   for Fisher-Jenks in big data settings*. Transactions in GIS. Cited by 32.
   → The second paper: computational tradeoffs when full enumeration is infeasible.

3. **Stephens (2026, this work)** — *swiftperm-bench*.
   → The third paper: GPU acceleration of spatial inference on Apple silicon.

The narrative arc is clean: CPU parallelism (2013) → sampling tradeoffs under
computational constraints (2017) → GPU compute removes the constraints (2026).

Note: The 2013 paper also has 32 citations and was published in IJGIS, the
higher-prestige sibling journal to Transactions in GIS. This gives you two
possible target venues with direct lineage.

### Specific technical parallels

**Spatial autocorrelation as a confounding factor:**
Rey et al. showed that spatial dependence (ρ) directly impacts classification accuracy
via effective sample size reduction. The same issue applies here — spatially
autocorrelated data affects the power of permutation tests, which is one argument
for *more* permutations (not fewer) when ρ is high. Our synthetic datasets vary
spatial autocorrelation, which sets up this connection naturally.

**Hardware context as a constraint:**
Rey et al. used an MPI cluster (dual quad-core Xeon, 24GB, Infiniband) as their
compute environment and noted memory as the binding constraint for Fisher-Jenks
at large n. This paper uses a consumer Apple M3 laptop and shows that unified
memory GPU compute changes the feasibility frontier for a different class of
spatial computation. Nice narrative arc: cluster → laptop, 2013 → 2025.

**Monte Carlo simulation design:**
Rey et al. ran 1,024 realizations per parameter combination across a grid of
n × ρ × k × sample_size values. Our benchmark runs 99,999 permutations per
implementation per dataset size. The methodological lineage (simulation-based
evaluation of spatial computational methods) is direct.

**PySAL as the reference implementation:**
Rey et al. used PySAL's serial Fisher-Jenks as the baseline. This work uses
esda/libpysal's permutation test as the baseline. Same ecosystem, same
reference software community.

### Key citation sentence for introduction

Something like: "Building on earlier work examining computational tradeoffs in
spatial classification [Rey et al. 2017], this paper investigates whether
modern Apple silicon GPU compute can remove the feasibility constraints that
currently limit permutation-based spatial inference at research-relevant
dataset sizes."

### What's different (important to articulate)

Rey et al. asked: "given a fixed hardware budget, how much accuracy do you
sacrifice by sampling?" — the answer guided algorithm selection.

This paper asks: "given a fixed algorithm, how much does hardware matter?" —
the answer guides platform selection and informs what's now computationally
routine vs. what still requires compromise.

---

## Suggested paper structure (rough)

1. **Introduction** — the computational barrier to rigorous permutation inference;
   brief connection to prior work on computational tradeoffs in spatial analysis
2. **Background** — Moran's I, permutation testing, Apple silicon architecture
   and unified memory; why GPU matters differently here than on discrete GPU systems
3. **Implementation** — four implementations (NumPy, Numba, Swift CPU, Swift Metal);
   three Metal shader variants and their selection logic
4. **Results** — benchmark across n=49 to n=10,000; throughput table;
   crossover analysis (Metal vs CPU parallel)
5. **Discussion** — iteration at scale as the primary contribution; p-value
   precision as secondary; effect of spatial autocorrelation on when more
   permutations matter; limitations (batch dispatch at large n, single statistic,
   macOS only)
6. **Conclusion** — path to generalisation (arbitrary statistics, Python binding,
   R package); relevance to the broader spatial analytics ecosystem

---

## Hardware context

All results: Apple M3, 8GB unified memory, 8 CPU cores.
Metal peak: 14.85M perm/s (77.8x over NumPy) at n=49, scratchless stack shader.
Swift parallel peak: 349k perm/s at n=500 (4.5x over NumPy, 2.0x over Numba).

Prior paper hardware: dual quad-core Intel Xeon 2.93GHz, 24GB shared memory,
MPI cluster with Infiniband. Fisher-Jenks at n=15,625 took seconds per realization.
