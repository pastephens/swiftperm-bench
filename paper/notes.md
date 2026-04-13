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

## The crossover problem: when Metal loses to Numba

### The data is unambiguous

Metal wins below n≈1,000. Numba wins above it, and the gap widens:

| n      | Numba    | Metal    | Winner          |
|--------|----------|----------|-----------------|
| 49     | 5.4M/s   | 14.8M/s  | Metal 2.7x      |
| 500    | 173k/s   | 294k/s   | Metal 1.7x      |
| 2,000  | 66k/s    | 44k/s    | Numba 1.5x      |
| 5,000  | 31k/s    | 13k/s    | Numba 2.5x      |
| 10,000 | 17k/s    | 5.4k/s   | Numba 3.0x      |

By n=10,000, Numba is 3x faster than Metal and the gap would continue widening.

### Why this happens — it's structural, not incidental

The fundamental bottleneck at large n is the sparse matrix-vector multiply inside
each permutation: `sum_k w_ij * z[perm[j]]` across the nnz nonzeros. This is a
memory-bound operation with **irregular access patterns** — the permutation indices
scatter-read from z in random order.

GPU compute shaders are not better than CPU for irregular memory access. They are
designed for structured, coalesced reads where many threads access contiguous
memory. The M3 GPU has high bandwidth but its many parallel threads thrash the
cache competing for scattered index lookups into z.

Numba's parallel CPU loop, by contrast, has 8 cores each working on a contiguous
chunk of permutations with reasonable cache locality per thread. The CPU's larger
cache per core handles the irregular access pattern far better.

Unified memory eliminates the CPU↔GPU transfer penalty — but it cannot change the
fundamental memory access pattern problem.

### What this means for the paper

This is a **stronger scientific contribution** than "Metal wins everywhere."
The honest result — identifying the crossover point and characterizing why it
occurs — is more defensible and more useful to the community than an oversold claim.

The narrative becomes:
- Metal dominates for small n typical of many social science spatial datasets (n < 1,000)
- Numba dominates at larger n due to irregular memory access patterns inherent
  to sparse spatial weights computation
- Unified memory eliminates transfer overhead but doesn't change the access pattern problem
- This motivates a **different GPU approach for large n**: restructuring the
  computation to exploit memory coalescing rather than just parallelizing the
  existing algorithm

### The algorithmic path forward (future work)

A coalescing-aware Metal shader would reorder the weight traversal to group
accesses by neighbor rather than by focal observation — essentially transposing
the computation. This is non-trivial algorithmically and requires restructuring
the sparse weight representation. It's a natural follow-on paper.

Alternatively, a block-sparse representation that groups spatially proximate
observations together in memory could improve cache coherence for both CPU and GPU
paths. This connects back to the 2013 paper's theme of data structure choices
for spatial computation.

### Connection to pastephens/libpysal sparse-kernel-weights branch

This is NOT the December 2025 pysal optimization work — it is a separate,
recent contribution (April 10-11, 2026) to the actual upstream libpysal
codebase, currently sitting in a fork as a PR-ready branch.

**Repository:** https://github.com/pastephens/libpysal/tree/sparse-kernel-weights
**Status:** 3 commits ahead of pysal/libpysal:main, ready to open as PR
**Files changed:** `libpysal/graph/_kernel.py` (+62/-29), `pyproject.toml` (+8)

**What the branch does:**
For kernels with compact support (bisquare, boxcar, triangular, cosine,
parabolic/discrete) with a fixed numeric bandwidth and euclidean metric,
uses `KDTree.sparse_distance_matrix` instead of building the full N×N
dense distance matrix. Reduces memory from O(N²) to O(N * avg_neighbors).

Commit message benchmark claim: **20-735x speedup** depending on dataset
size and bandwidth.

The fast path is skipped (existing behaviour preserved) when:
- kernel has infinite support (gaussian, exponential, identity)
- bandwidth is None or 'auto' (full matrix needed to compute it)
- metric is not euclidean (KDTree only supports euclidean)
- k is provided (already uses a tree-based path)

**The structural parallel to Metal crossover is exact:**

| Context | Problem | Fix | Status |
|---|---|---|---|
| `_kernel.py` compact kernels | Full N×N dense matrix, O(N²) memory | KDTree sparse (O(N·k)) | ✓ Fixed, 20-735x |
| Metal permutations at large n | Irregular scatter-reads, cache thrashing | Coalescing-aware shader | Open |

Both are the same class of problem: an algorithm written for a different
execution model (dense CPU computation) running on hardware that requires
a different memory access pattern to be efficient.

**Citation approach:**
At submission time this branch will either be:
1. Merged into libpysal main → cite as libpysal release notes / changelog
2. Still a PR → cite as "submitted" with the GitHub PR URL
3. Unpublished → cite as GitHub repository at specific commit SHA

Option 1 is cleanest and most likely given the PR is nearly ready.
Opening the PR before paper submission would be ideal.

**Suggested framing in paper (Discussion section):**

> A parallel optimization of libpysal's kernel weight construction
> (pastephens/libpysal, branch sparse-kernel-weights) demonstrates that
> replacing a dense O(N²) distance matrix with a sparse KDTree query
> achieves 20-735x speedup for compact support kernels. The Metal crossover
> observed here is the same structural problem on GPU hardware: the
> permutation loop performs irregular scatter-reads across the z vector
> that are poorly suited to the GPU's cache architecture. A coalescing-aware
> Metal shader, analogous to the KDTree sparse approach, is a natural
> direction for future work.

---

### Revised contribution statement

"We benchmark four implementations of Moran's I permutation inference on Apple
silicon and identify a crossover at n≈1,000 where the optimal backend transitions
from Metal GPU (77.8× over NumPy at small n) to CPU parallel (Numba, 3× faster
than Metal at n=10,000). We show this crossover is structurally determined by
irregular memory access patterns in sparse spatial weight traversal, not by
dispatch overhead, and characterize the conditions under which each backend should
be preferred."

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
   classification in PySAL*. IJGIS. Cited by 32. BibTeX key: `Rey01052013`
   "Refactoring a spatial analysis library to support parallelization... parallel
   implementations of Fisher-Jenks using a multi-core, single desktop environment."
   → The first paper: parallelizing spatial computation on CPU, desktop hardware.

2. **Rey, Stephens & Laura (2017)** — *Sampling and full enumeration strategies
   for Fisher-Jenks in big data settings*. Transactions in GIS. Cited by 32. BibTeX key: `Rey2017tgis`
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
