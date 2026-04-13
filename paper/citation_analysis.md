# Citation Analysis — Rey, Stephens & Laura (2017)
# 32 citing papers as of April 2026

## Complete citation list by domain

### Computational spatial methods (most directly related to new work)
- **Rey et al. (2013)** — Parallel optimal choropleth map classification in PySAL. IJGIS. Cited by 32. ← DIRECT PREDECESSOR (Philip is co-author)
- Prasad et al. (2017) — Parallel Processing over Spatial-Temporal Datasets. IEEE Big Data Congress. Cited by 50.
- Rey et al. (2022) — The PySAL Ecosystem: Philosophy and Implementation. Geographical Analysis. Cited by 62.
- Anselin (2024) — An Introduction to Spatial Data Science with GeoDa (textbook). Cited by 101.
- Zhou et al. (2019) — A visualization approach for discovering colocation patterns. IJGIS. Cited by 18.
- Fan et al. (2024) — Understanding reader takeaways in thematic maps. ACM CHI. Cited by 16.
- Fan (2025) — Addressing Design Errors and Deception in Data Visualizations. Dissertation.

### Remote sensing / ecology / environmental science
- Wu et al. (2022) — Ecological environment quality, Sahel region, remote sensing index. Journal of Arid Land. Cited by 79.
- Pan et al. (2024) — Aboveground biomass, Central Asia grassland, UAV. Environmental and Experimental Botany. Cited by 20.
- Dynowski et al. (2019) — Recreational activities impact on alpine lakes. Water (MDPI). Cited by 39.
- Zhao et al. (2025) — Biomass inversion, UAV remote sensing, Hulunbuir grassland. (Chinese language journal)
- Alhemiary (2016) — Remote sensing, vegetation degradation, Socotra island. Dissertation.

### Public health / epidemiology / one health
- Charette-Castonguay et al. (2025) — Zoonotic influenza risk, Nepal. One Health. Cited by 3.
- Larkins et al. (2023) — Risk mapping Taenia solium, Lao PDR. Tropical Medicine & International Health. Cited by 7.
- Larkins (2023) — Techniques for mapping Taenia solium, Lao PDR. Dissertation. Cited by 2.

### Urban / geography / tourism
- Atwal et al. (2022) — Predicting building types using OpenStreetMap. Scientific Reports. Cited by 74.
- Rashid et al. (2025) — Urban surfaces spatial distribution, Bangladesh. Remote Sensing (MDPI). Cited by 1.
- Qin et al. (2026) — Hotel chain diffusion dynamics, China. International Journal of Hospitality Management.
- Qin et al. (2023) — Location and regionalization of hotel chains, China. Tourism Geographies. Cited by 6.
- Shaban & Vermeylen (2022) — Creative industries, India. Book chapter.
- Gu et al. (2019) — Tourism source market, Nanjing. Scientia Geographica Sinica. (Chinese) Cited by 19.
- Rong et al. (2020) — Rural tourism source market, Sunan. (Chinese journal) Cited by 7.

### Disaster risk / hydrology
- Yin et al. (2026) — Flood risk assessment, Kelantan, Malaysia. Journal of Hydrology Regional Studies.
- Gouett-Hanna et al. (2022) — Flood risk access and equity, Metro Vancouver. Canadian Water Resources Journal. Cited by 7.
- Raczyński & Dyer (2022) — Low flow identification, breakpoint analysis. Water (MDPI). Cited by 19.

### Social science / extremism
- Brace et al. (2024) — 'Mixed, unclear, unstable' ideologies, incelosphere. Journal of Policing, Intelligence and Counter Terrorism. Cited by 51.
- Baele et al. (2023) — Super- and hyper-posters on extremist forums. Journal of Policing. Cited by 20.

### Miscellaneous applied
- Harper et al. (2020) — Mobile location data for hurricane evacuation. IEEE SIEDS. Cited by 5.
- Krzywnicka et al. (2020) — Municipal waste evaluation, Poland. IJERPH. Cited by 2.
- Winschel — Daily Fantasy Basketball Lineup Optimizer. (unpublished/ResearchGate)
- Zacherl — Predicting enrollment at Mercyhurst University. (unpublished/ResearchGate)
- Baeza-Loya et al. (2026) — Hair cell spatial patterning, zebrafish. Development. (biology)

---

## Analysis for related work section

### What this tells us about the citation profile

The 32 citations split roughly into three tiers:

**Tier 1 — Direct computational spatial methods peers (6 papers)**
These are the papers that matter most for positioning the new work. The PySAL
ecosystem paper (Rey et al. 2022) and Anselin's GeoDa textbook confirm the work
is canonical within the spatial data science community. The Prasad et al. IEEE
Big Data paper is the most direct computational peer — it's about parallel
spatial computation at scale and cites the Fisher-Jenks paper as context.
Zhou et al. (2019, IJGIS) cites the 25% random sampling recommendation directly.

**Tier 2 — Applied GIS users (15+ papers)**
Remote sensing, public health, urban geography, hydrology papers that use
Fisher-Jenks for classification and cite the sampling guidance. These confirm
the broad applied reach of the work but are not peers for this new paper.

**Tier 3 — Peripheral / methodological borrowers (10 papers)**
Tourism, social science, biology papers that borrow Fisher-Jenks as a tool.
Interesting that extremism/radicalization papers (Journal of Policing) appear
twice — Fisher-Jenks is being used for clustering social media posting behavior.
Not relevant to the new paper.

### Key observation: NO GPU/acceleration citations

None of the 32 citing papers extend the computational work toward GPU acceleration,
Metal, or modern hardware-aware implementations. The Prasad et al. IEEE paper
discusses parallel spatial processing roadmaps but doesn't follow through to
GPU-specific implementations. This confirms the gap the new work fills — the
computational acceleration lineage has not been continued in the 8 years since
the 2017 paper.

### Strongest related work sentence

"While prior work has examined computational tradeoffs in spatial classification
[Rey et al. 2017] and laid out roadmaps for parallel spatial-temporal processing
[Prasad et al. 2017], no subsequent work has extended these approaches to
GPU-accelerated inference on consumer hardware — a gap this paper addresses
through the first benchmark of permutation-based spatial statistics on Apple
silicon."

### Target venues (inferred from citation profile)

Most citing papers appear in:
- Transactions in GIS (home journal of the 2017 paper) ← primary target
- International Journal of Geographical Information Science
- Geographical Analysis (PySAL ecosystem paper)
- Computers, Environment and Urban Systems
- Water, Remote Sensing (MDPI) — less prestigious, not targets

**Recommended target: Transactions in GIS or Geographical Analysis**

Transactions in GIS published the 2017 paper and has the right audience.
Geographical Analysis is higher prestige and published the PySAL ecosystem
paper (Rey et al. 2022) — if the scope is framed broadly as a contribution
to spatial data science infrastructure, it could fit there.
