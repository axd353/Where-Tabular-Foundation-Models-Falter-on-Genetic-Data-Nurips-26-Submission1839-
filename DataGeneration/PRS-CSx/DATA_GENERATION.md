# PRS-CSx–Inspired Synthetic Task Family: Data Generation Specification

## Motivation

The hierarchical GP (HGP) pipeline (`test_train_dist_shift.py` → `make_data_ver3.py`) generates tasks by drawing ancestry-varying SNP effects from Gaussian processes. Extended instruction tuning exclusively on GP-based tasks risks over-adaptation to GP-specific inductive biases (smooth, dense, kernel-correlated effects).

This pipeline generates data using the **PRS-CSx continuous-shrinkage Bayesian modelling framework** (Ruan et al., *Nature Genetics* 2022), adapted to continuous ancestry. The effect-size prior is **sparse and heavy-tailed** (global–local shrinkage hierarchy), producing a complementary inductive bias: few SNPs with large effects, most near zero, with explicit control of cross-population genetic correlation.

The output structure (train/test/latents pickles, manifests, DataFrame columns) is **identical** to the HGP pipeline, so both task families can be used interchangeably for instruction tuning.

---

## 1. Notation and Fixed Constants

### Dimension Symbols

Consistent with `test_train_dist_shift.py` / `make_data_ver3.py`:

| Symbol | Meaning | Type |
|--------|---------|------|
| $M$ | Number of individuals | scalar int |
| $A$ | Dimensionality of continuous ancestry space | scalar int |
| $N$ | Number of SNPs (post-GWAS-selection, LD-pruned) | scalar int |
| $K_c$ | Number of ancestry clusters (`num_clusters`) | scalar int |

Index conventions: $i \in \{0, \ldots, M-1\}$ over individuals, $j \in \{0, \ldots, N-1\}$ over SNPs, $k \in \{0, \ldots, K_c - 1\}$ over clusters.

### Constants Fixed for ALL Tasks

| Symbol | Value | Shape | Description | Shared with HGP? |
|--------|-------|-------|-------------|-------------------|
| $M$ | $\{10000, 15000, 20000, 25000\}$ | scalar | Total individuals per task (sampled per task) | ⚠️ HGP fixed at 5000; PRS-CSx uses biobank-scale cohorts |
| $A$ | 16 | scalar | Ancestry coordinate dimensions | ✅ Same (`A=16` in HGP `sample_setting`) |
| $a$ | 1 | scalar | Shape param of local shrinkage (Strawderman–Berger) | ❌ New (PRS-CSx specific) |
| $b$ | $\tfrac{1}{2}$ | scalar | Shape param of intermediate gamma (Strawderman–Berger) | ❌ New (PRS-CSx specific) |
| $L_\text{scale1}$ | 1.0 | scalar | Target mean inter-cluster distance | ✅ Same (`L_scale1=1.0` in HGP) |
| `noise_percent` | 10.0 | scalar | % of individuals as uniform noise in cluster sizing | ✅ Same (HGP `noise_percent=10.0`) |
| `maf_range` | $(0.05, 0.5)$ | $(2,)$ | Bounds for ancestral MAF sampling | ⚠️ Similar (HGP uses $(0.1, 0.5)$; we widen to $0.05$ to allow rarer variants under Balding–Nichols) |
| `intercept` | 0.0 | scalar | Logit intercept for label generation | ✅ Same (`intercept=0.0` in HGP) |

### Constants Fixed Per Task

Sampled once per task from hyperparameter ranges at task creation time, then held fixed during simulation of that task's $M$ individuals:

| Symbol | Shape | Description |
|--------|-------|-------------|
| $N$ | scalar | Number of SNPs |
| $K_c$ | scalar | Number of clusters |
| $\phi$ | scalar | Global shrinkage parameter |
| $r_g$ | scalar | Cross-population genetic correlation upper bound (per-SNP $r_{g,j} \sim \text{Uniform}(0.04, r_g)$) |
| $\pi_\text{causal}$ | scalar | Fraction of SNPs that are causal |
| $\alpha$ | scalar | MAF-dependent architecture exponent |
| $h$ | scalar | Ancestry interpolation bandwidth upper bound (per-SNP $h_j \sim \text{Uniform}(0.01, h)$) |
| `cluster_sd_mult` | scalar | Base per-cluster SD in ancestry space ($\times L_\text{scale1}$) |
| `indiv_sd_choices` | $(K_c,)$ | Per-cluster width multipliers |
| `ancestry_group_cutpoints` | variable | Cluster group boundary indices |
| `ancestry_group_size_divisors` | variable | Cluster group size divisors |
| $\sigma_{\beta,k}$ | $(K_c,)$ | Per-population effect-size scale |
| $h^2_k$ | $(K_c,)$ | Per-population target SNP heritability |
| $F_{ST,k}$ | $(K_c,)$ | Per-cluster $F_{ST}$ (allele freq divergence) |

---

## 2. Data Generation Stages

For a single task, the generation proceeds through **7 stages** sequentially.

---

### Stage 0: Cluster Centres & Individual Ancestry

**Identical to the HGP pipeline** (`make_data_ver3.py` stages 1–3). Reuses: `_scale_centers_to_target_mean_distance`, `_make_imbalanced_cluster_ids`, `_assign_cluster_sds`.

#### (0a) Cluster centres

Sample $K_c$ cluster centres in $\mathbb{R}^A$, then rescale so mean pairwise distance equals $L_\text{scale1}$:

$$\mathbf{c}_k^\text{raw} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_A), \quad k = 0, \ldots, K_c - 1$$

$$\{\mathbf{c}_k\}_{k=0}^{K_c-1} = \texttt{\_scale\_centers\_to\_target\_mean\_distance}\!\left(\{\mathbf{c}_k^\text{raw}\},\; L_\text{scale1}\right)$$

**Shape:** `centers` — $(K_c, A)$

#### (0b) Imbalanced cluster assignments

$$\texttt{cluster\_ids} = \texttt{\_make\_imbalanced\_cluster\_ids}(M, K_c, \texttt{cutpoints}, \texttt{divisors}, \texttt{noise\_percent})$$

**Shape:** `cluster_ids` — $(M,)$ int

Cluster 0 is always the **majority** cluster. Minority clusters have sizes $\propto 1/\text{divisor}$ relative to the majority.

##### How cutpoints and divisors work

The $K_c$ clusters are partitioned into **ancestry groups** by `ancestry_group_cutpoints`. Cutpoints are *exclusive end-indices* into the cluster index array $[0, K_c)$. The first group always starts at cluster index 0.

With `cutpoints = $(c_1, c_2)$` the groups are:

| Group | Cluster indices | Relative weight per cluster | Role |
|-------|----------------|----------------------------|------|
| 0 (major) | $[0,\; c_1)$ | $w = 1.0$ | Largest ancestry group |
| 1 | $[c_1,\; c_2)$ | $w = 1/d_1$ | Intermediate minority |
| 2+ | $[c_2,\; K_c)$ | $w = 1/d_2$ | Smallest minority |

where $(d_1, d_2)$ = `ancestry_group_size_divisors`.

The **base count** for cluster $k$ with weight $w_k$ is:

$$n_k^{\text{base}} = \left\lfloor \lambda \cdot w_k \right\rfloor, \qquad \lambda = \left\lfloor \frac{(1 - f_\text{noise}) \cdot M}{\sum_{k'} w_{k'}} \right\rfloor$$

where $f_\text{noise} = \texttt{noise\_percent} / 100$. After base assignment, the remaining $M - \sum_k n_k^\text{base}$ individuals are assigned to clusters chosen **uniformly at random** (this is the noise component, $\approx 10\%$ of $M$).

##### Worked example: $K_c = 5$, cutpoints = $(1, 3)$, divisors = $(4, 6)$

| Group | Clusters | Weight $w_k$ | Meaning |
|-------|----------|-----------|---------|
| 0 | $\{0\}$ | 1.0 | 1 major cluster |
| 1 | $\{1, 2\}$ | $1/4 = 0.25$ | 2 intermediate minorities (each $\frac{1}{4}$ of major) |
| 2 | $\{3, 4\}$ | $1/6 \approx 0.167$ | 2 small minorities (each $\frac{1}{6}$ of major) |

Total weight: $1.0 + 2 \times 0.25 + 2 \times 0.167 = 1.833$. With $M = 5000$, $f_\text{noise} = 0.10$:

$$\lambda = \left\lfloor \frac{0.9 \times 5000}{1.833} \right\rfloor = \left\lfloor 2455 \right\rfloor = 2455$$

| Cluster | $n_k^\text{base}$ | After noise (approx) | Fraction |
|---------|-------------------|---------------------|----------|
| 0 | $\lfloor 2455 \times 1.0 \rfloor = 2455$ | $\approx 2555$ | $\approx 51\%$ |
| 1 | $\lfloor 2455 \times 0.25 \rfloor = 613$ | $\approx 713$ | $\approx 14\%$ |
| 2 | $\lfloor 2455 \times 0.25 \rfloor = 613$ | $\approx 713$ | $\approx 14\%$ |
| 3 | $\lfloor 2455 \times 0.167 \rfloor = 409$ | $\approx 509$ | $\approx 10\%$ |
| 4 | $\lfloor 2455 \times 0.167 \rfloor = 409$ | $\approx 509$ | $\approx 10\%$ |

##### Worked example: $K_c = 3$, cutpoints = $(1, 2)$, divisors = $(4,)$

With only one divisor, only two groups exist:

| Group | Clusters | Weight $w_k$ |
|-------|----------|-----------|
| 0 | $\{0\}$ | 1.0 |
| 1 | $\{1, 2\}$ | $1/4 = 0.25$ |

Total weight: $1.5$. With $M = 5000$, $f_\text{noise} = 0.10$:

$$\lambda = \left\lfloor \frac{4500}{1.5} \right\rfloor = 3000$$

| Cluster | $n_k^\text{base}$ | After noise (approx) | Fraction |
|---------|-------------------|---------------------|----------|
| 0 | 3000 | $\approx 3167$ | $\approx 63\%$ |
| 1 | 750 | $\approx 917$ | $\approx 18\%$ |
| 2 | 750 | $\approx 917$ | $\approx 18\%$ |

##### How this pipeline samples cutpoints/divisors

From `_sample_ancestry_group_structure` (shared with HGP):

- **Group 0** is always cluster 0 alone → cutpoints always start with 1.
- **Group 1** has 1 or 2 clusters (random, where $K_c$ allows). Second cutpoint $c_2 = 1 + |\text{Group 1}|$.
- **Group 2** (if it exists) contains all remaining clusters $[c_2, K_c)$.
- $d_1 = 4$ always; $d_2 \in \{5, 6, 7\}$ uniformly (only drawn when Group 2 is non-empty).

The net effect: cluster 0 is always $\sim 2\text{–}4\times$ larger than any minority cluster. The noise component ($\sim 10\%$) prevents any cluster from being *completely* absent, which is important for ensuring every cluster has enough individuals for balanced test splits.

#### (0c) Per-cluster ancestry spread

$$L_\text{scale2} = \texttt{cluster\_sd\_mult} \times L_\text{scale1}$$

Per-cluster width multipliers are assigned via `_assign_cluster_sds`, with the constraint that the majority cluster's SD is forced to $2\times$ the maximum minority SD:

$$\mathbf{m} = \texttt{\_assign\_cluster\_sds}(K_c, \texttt{indiv\_sd\_choices}) \qquad \text{shape: } (K_c,)$$

$$\sigma_k^\text{cluster} = m_k \cdot L_\text{scale2} \qquad \text{shape: } (K_c,)$$

#### (0d) Per-individual ancestry coordinates

For individual $i$ in cluster $k_i$:

$$\mathbf{a}_i \sim \mathcal{N}\!\left(\mathbf{c}_{k_i},\; (\sigma_{k_i}^\text{cluster})^2 \, \mathbf{I}_A\right)$$

**Shape:** $\mathbf{A}_\text{mat}$ — $(M, A)$

> **Admixture and cluster overlap.** The degree of overlap between clusters — and hence the fraction of effectively admixed individuals — is controlled jointly by **`cluster_sd_mult`** and **$K_c$**. See [Section 4: Admixture and Cluster Overlap](#admixture-and-cluster-overlap) for detailed analysis.

---

### Stage 1: Global–Local Continuous Shrinkage Prior (PRS-CSx core)

This is the core of the PRS-CSx model. The shrinkage hierarchy produces per-SNP local variance parameters $\psi_j$. These are **shared across populations** — neither $\phi$ nor $\psi_j$ depends on $k$.

#### (1a) Gamma–Gamma hierarchy

For each SNP $j = 0, \ldots, N-1$:

$$\delta_j \sim \text{Gamma}\!\left(b,\; \phi\right) \qquad \Rightarrow \qquad \mathbb{E}[\delta_j] = \frac{b}{\phi}$$

$$\psi_j \sim \text{Gamma}\!\left(a,\; \delta_j\right) \qquad \Rightarrow \qquad \mathbb{E}[\psi_j] = \frac{a}{\delta_j}$$

where $a = 1$, $b = \tfrac{1}{2}$, and $\text{Gamma}(\text{shape}, \text{rate})$ parametrisation is used throughout (i.e., if $X \sim \text{Gamma}(\zeta, \eta)$ then $f(x) \propto x^{\zeta-1} e^{-\eta x}$).

The marginal prior this induces on $\psi_j$ has:
- **Heavy mass near zero** → strong shrinkage on noise SNPs
- **Heavy (Cauchy-like) tails** → avoids over-shrinkage of true signals

**Shapes:** $\boldsymbol{\delta}$ — $(N,)$, $\boldsymbol{\psi}$ — $(N,)$

#### (1b) Causal/null SNP assignment

$$z_j \sim \text{Bernoulli}(\pi_\text{causal}), \quad j = 0, \ldots, N-1$$

**Shape:** $\mathbf{z}$ — $(N,)$ bool

If $z_j = 0$, then $\beta_{jk} = 0$ for all $k$ (this SNP is a null/noise feature).
If $z_j = 1$, the SNP is causal and its effects are drawn in Stage 2.

---

### Stage 2: Per-Population Effect Sizes

For each **causal** SNP $j$ (where $z_j = 1$), draw effect sizes jointly across all $K_c$ populations using a multivariate normal with a **per-SNP** genetic correlation $r_{g,j}$.

#### (2a) Per-SNP genetic correlation

Each SNP gets its own cross-population correlation drawn from the task-level upper bound $r_g$:

$$r_{g,j} \sim \text{Uniform}(0.04,\; r_g)$$

The floor of $0.04$ ensures that even when $r_g$ is moderate, no SNP has near-zero cross-population correlation (which would be biologically extreme for GWAS-selected variants). Different SNPs having different $r_{g,j}$ values reflects the biological reality that some loci have highly conserved effects across ancestries while others diverge substantially.

#### (2b) Cross-population covariance

For SNP $j$:

$$\boldsymbol{\Sigma}_\text{pop}^{(j)} = \psi_j \cdot \mathbf{C}_j$$

where $\mathbf{C}_j \in \mathbb{R}^{K_c \times K_c}$ is the per-SNP cross-population correlation/scale matrix:

$$C_{j,kk} = \sigma_{\beta,k}^2, \qquad C_{j,k\ell} = r_{g,j} \cdot \sigma_{\beta,k} \cdot \sigma_{\beta,\ell} \quad (k \neq \ell)$$

With the default $\sigma_{\beta,k} = 1$ for all $k$, this simplifies to:

$$\boldsymbol{\Sigma}_\text{pop}^{(j)} = \psi_j \begin{pmatrix} 1 & r_{g,j} & \cdots & r_{g,j} \\ r_{g,j} & 1 & \cdots & r_{g,j} \\ \vdots & \vdots & \ddots & \vdots \\ r_{g,j} & r_{g,j} & \cdots & 1 \end{pmatrix}$$

This matrix is positive definite for $r_{g,j} \in (-\tfrac{1}{K_c - 1}, 1]$, which is satisfied since $r_{g,j} \geq 0.04$.

#### (2c) Joint sampling

For each causal SNP $j$:

$$\begin{pmatrix} \beta_{j,0} \\ \beta_{j,1} \\ \vdots \\ \beta_{j,K_c-1} \end{pmatrix} \sim \mathcal{N}\!\left(\mathbf{0},\; \boldsymbol{\Sigma}_\text{pop}^{(j)}\right)$$

For non-causal SNPs ($z_j = 0$):

$$\beta_{j,k} = 0 \quad \forall\, k$$

**Shape:** $\boldsymbol{\beta}_\text{pop}$ — $(K_c, N)$, $\mathbf{r}_{g}$ — $(N,)$

**Factored sampling** (per-SNP Cholesky of $\mathbf{C}_j$, then scale by $\sqrt{\psi_j}$):

$$\mathbf{L}_j = \text{chol}(\mathbf{C}_j), \qquad \mathbf{u}_j \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{K_c})$$

$$\boldsymbol{\beta}_{\cdot,j} = \sqrt{\psi_j} \cdot \mathbf{L}_j \, \mathbf{u}_j$$

---

### Stage 3: Allele Frequencies via Balding–Nichols Model

Replaces the HGP pipeline's uniform-offset MAF model with the genetically principled **Balding–Nichols** model parameterised by $F_{ST}$ per cluster.

#### (3a) Ancestral MAFs

For each SNP $j$:

$$p_j^\text{anc} \sim \text{Uniform}(p_{\text{min}}, p_{\text{max}})$$

**Shape:** $\mathbf{p}^\text{anc}$ — $(N,)$

#### (3b) Cluster-specific MAFs

For each cluster $k$ and SNP $j$, let $F = F_{ST,k}$:

$$\alpha_{kj} = p_j^\text{anc} \cdot \frac{1 - F}{F}, \qquad \beta_{kj} = (1 - p_j^\text{anc}) \cdot \frac{1 - F}{F}$$

$$p_{k,j} \sim \text{Beta}(\alpha_{kj},\; \beta_{kj})$$

**Properties:**

$$\mathbb{E}[p_{k,j}] = p_j^\text{anc}$$

$$\text{Var}(p_{k,j}) = F_{ST,k} \cdot p_j^\text{anc} \cdot (1 - p_j^\text{anc})$$

Key: rare SNPs ($p_j^\text{anc}$ near 0) cannot drift as far as common SNPs — the Beta distribution naturally constrains this. Each SNP gets an **independent** per-cluster deviation, unlike the HGP's shared $\delta_k$ across all SNPs.

**Shape:** `maf_by_cluster` — $(K_c, N)$

#### (3c) Dosage genotypes

For individual $i$ in cluster $k_i$, SNP $j$:

$$G_{i,j} \sim \text{Binomial}(2,\; p_{k_i, j})$$

**Shape:** $\mathbf{G}$ — $(M, N)$ int in $\{0, 1, 2\}$

---

### Stage 4: MAF-Dependent Architecture Adjustment

Optional post-hoc rescaling of effect sizes by allele frequency, following PRS-CSx paper secondary simulation (7). Models **negative selection**: rare variants carry larger per-allele effects.

For each cluster $k$ and causal SNP $j$:

$$\beta_{\text{pop},k,j} \;\leftarrow\; \beta_{\text{pop},k,j} \cdot \left[2\, p_{k,j} \,(1 - p_{k,j})\right]^{\alpha/2}$$

**Scaling factor values:**

| $p_{k,j}$ | $\alpha = 0$ | $\alpha = -0.25$ |
|------------|-------------|-------------------|
| 0.05 | 1.00 | $[2 \cdot 0.05 \cdot 0.95]^{-0.125} \approx 1.29$ |
| 0.10 | 1.00 | $[2 \cdot 0.10 \cdot 0.90]^{-0.125} \approx 1.16$ |
| 0.30 | 1.00 | $[2 \cdot 0.30 \cdot 0.70]^{-0.125} \approx 0.99$ |
| 0.40 | 1.00 | $[2 \cdot 0.40 \cdot 0.60]^{-0.125} \approx 0.94$ |
| 0.50 | 1.00 | $[2 \cdot 0.50 \cdot 0.50]^{-0.125} \approx 0.92$ |

Net amplification (rare/common) at $\alpha = -0.25$: $\sim 1.37\times$ for $p=0.05$ vs $p=0.40$.

Since $p_{k,j}$ differs across clusters (via $F_{ST}$), this creates **additional non-stationarity**: the effective per-allele effect magnitude depends on the cluster *through its allele frequency*, even if the raw $\beta_{\text{pop},k,j}$ were identical across clusters.

**Shape:** $\boldsymbol{\beta}_\text{pop}$ — $(K_c, N)$ (modified in place)

---

### Stage 5: Ancestry-Continuous Interpolation

The PRS-CSx model operates on **discrete populations**. We extend to continuous ancestry by interpolating per-population effects using kernel weights based on distance from cluster centres. Crucially, **each SNP gets its own interpolation bandwidth** $h_j$, so different loci can have different spatial smoothness across the ancestral continuum.

#### (5a) Per-variant bandwidth sampling

The task-level parameter $h$ is the **upper bound**. Each SNP draws its own bandwidth independently:

$$h_j \sim \text{Uniform}(h_{\text{floor}},\; h), \quad j = 0, \ldots, N-1$$

where $h_{\text{floor}} = 0.01$ is a small positive floor to avoid division-by-zero in the kernel.

**Shape:** $\mathbf{h}$ — $(N,)$

**Biological motivation.** Different loci have different evolutionary histories, selection pressures, and gene-environment interaction profiles. A SNP involved in a locally adaptive trait (e.g., skin pigmentation) may have effects that change sharply across ancestry ($h_j$ small), while a SNP affecting a conserved metabolic pathway may have nearly identical effects everywhere ($h_j$ large). This is the PRS-CSx analogue of the HGP pipeline's per-SNP lengthscale multipliers (`ls_multipliers`).

#### (5b) Per-variant interpolation weights

For each individual $i$, cluster $k$, and SNP $j$:

$$\tilde{w}_{k,j}(\mathbf{a}_i) = \exp\!\left(-\frac{\|\mathbf{a}_i - \mathbf{c}_k\|^2}{2\,h_j^2}\right)$$

Normalise to sum to 1 over clusters (separately for each SNP):

$$w_{k,j}(\mathbf{a}_i) = \frac{\tilde{w}_{k,j}(\mathbf{a}_i)}{\sum_{k'=0}^{K_c-1} \tilde{w}_{k',j}(\mathbf{a}_i)}$$

**Shape:** `weights` — $(M, K_c, N)$

For a given SNP $j$, the $(M, K_c)$ slice $w_{\cdot,j}(\cdot)$ plays the same role as the old shared weight matrix but with bandwidth $h_j$. Each SNP has its own soft-membership profile over clusters.

#### (5c) Per-individual effects

$$\beta_{i,j} = \sum_{k=0}^{K_c-1} w_{k,j}(\mathbf{a}_i) \cdot \beta_{\text{pop},k,j}$$

This is **no longer a single matrix multiply** because each column $j$ of $\boldsymbol{\beta}_\text{pop}$ uses a different set of weights. In pseudocode:

```
for j in range(N):
    beta[:, j] = W[:, :, j] @ beta_pop[:, j]    # (M,Kc) @ (Kc,) -> (M,)
```

Or equivalently using broadcasting:

$$\beta_{i,j} = \sum_k W_{i,k,j} \cdot \beta_{\text{pop},k,j}$$

**Shape:** $\boldsymbol{\beta}$ — $(M, N)$

#### Effect of per-variant bandwidth on the learning task

| Individual location | SNP with small $h_j$ | SNP with large $h_j$ |
|---|---|---|
| Near cluster centre $\mathbf{c}_k$ | Effect $\approx \beta_{\text{pop},k,j}$ (hard assignment) | Effect $\approx$ blend of nearby clusters |
| Between two centres | Effect dominated by nearest centre (sharp jump) | Effect is smooth average (gradual transition) |
| Overall pattern | Discrete-population-like; sharp non-stationarity | Smooth, GP-like spatial variation |

Within a single individual, **some SNPs** may be effectively hard-assigned to a population while **other SNPs** are blended — creating a rich, heterogeneous non-stationarity pattern that would be impossible with a single shared bandwidth.

---

### Stage 6: Phenotype Generation (Heritability-Calibrated Liability Model)

#### (6a) Genetic value

$$g_i = \sum_{j=0}^{N-1} G_{i,j} \cdot \beta_{i,j}$$

In matrix form: $\mathbf{g} = \text{rowsum}(\mathbf{G} \odot \boldsymbol{\beta})$ where $\odot$ is elementwise multiply.

**Shape:** $\mathbf{g}$ — $(M,)$

#### (6b) Heritability-calibrated noise

$h^2_k$ is specified **before** generation and controls the signal-to-noise ratio for cluster $k$'s individuals.

The noise is anchored to the **population-wide** genetic variance, computed over all $M$ individuals:

$$\widehat{\text{Var}}(g) = \text{Var}\!\left(\{g_i\}_{i=0}^{M-1}\right) \qquad \text{(empirical, from Stage 6a output)}$$

$$\sigma_{\epsilon,k}^2 = \widehat{\text{Var}}(g) \cdot \frac{1 - h^2_k}{h^2_k}$$

**Shape:** $\boldsymbol{\sigma}_\epsilon$ — $(K_c,)$

This is the **only** place $h^2_k$ enters the generative process. It does not affect effect-size magnitudes; it controls how much noise obscures the genetic signal.

> **Design rationale — why pooled variance, not per-cluster variance.**
> The earlier formulation used $\widehat{\text{Var}}_k(g)$ (per-cluster) as the baseline. Because majority is the largest cluster with the most genetically diverse individuals, $\widehat{\text{Var}}_0(g) \gg \widehat{\text{Var}}_k(g)$ for $k > 0$ (empirically $\sim 2.8\times$ on average). Even with higher $h^2_0$, majority received larger **absolute** noise $\sigma_{\epsilon,0}$, inadvertently making case/control labels noisier for majority in the raw G feature space seen by classifiers (lower Fisher discriminant ratio). Using the global $\widehat{\text{Var}}(g)$ as a shared baseline ensures $h^2_k$ is the **sole** differentiator: since $h^2_\text{major} \geq h^2_\text{minority}$ by construction (Section 4), majority now always gets strictly less noise, giving it cleaner labels and a higher Fisher ratio — consistent with the intended design where majority's higher heritability should translate to easier classification.

#### (6c) Latent liability

For individual $i$ in cluster $k_i$:

$$\epsilon_i \sim \mathcal{N}(0,\; \sigma_{\epsilon,k_i}^2)$$

$$\eta_i = \texttt{intercept} + g_i + \epsilon_i$$

**Shape:** $\boldsymbol{\eta}$ — $(M,)$

#### (6d) Binarisation

$$p_i = \sigma(\eta_i) = \frac{1}{1 + e^{-\eta_i}}$$

A **per-cluster** balanced threshold is applied: for each cluster $k$ independently,

$$t_k = \texttt{\_choose\_balanced\_threshold}\!\left(\{p_i : \texttt{cluster\_ids}[i] = k\}\right)$$

$$y_i = \begin{cases} 1 & \text{if } p_i \geq t_{k_i} \\ 0 & \text{otherwise} \end{cases}$$

**Shape:** $\mathbf{y}$ — $(M,)$ int

> **Design rationale — why per-cluster threshold, not a single global threshold.**
> A single global threshold is calibrated mainly by the majority cluster (which comprises 60–77% of $M$), leaving each minority cluster's threshold to fall wherever the global threshold happens to cut its $p$-distribution. This produced per-cluster prevalences ranging from 22% to 87% (observed empirically), creating a systematic "free accuracy" shortcut: a classifier that learns to identify ancestry via $A_*$ features can predict $y=1$ for minority individuals and be correct ~78% of the time without using any genetic information. Per-cluster thresholding guarantees each cluster independently has $\approx 50\%$ prevalence, eliminating this confound and ensuring the learning task is the same across all ancestries: predict disease status from genotypes under distribution shift.

---

## 3. Final Data and Pickle Structure

After generation, the train/test split is performed **identically to the HGP pipeline**: `_make_train_test_split_indices` ensures `test_per_cluster=50` balanced samples per cluster, with the remaining as training (possibly imbalanced).

### Artifacts per accepted setting

1. **Train pickle:** `<stem>_train.pkl`
2. **Test pickle:** `<stem>_test.pkl`
3. **Latents pickle:** `<stem>_latents.pkl`
4. **One manifest row** in `train_manifest.csv` or `eval_manifest.csv`

### Train/Test pickle contents

Each is a dict with identical structure to the HGP pipeline:

| Key | Type | Description |
|-----|------|-------------|
| `"table"` | `pandas.DataFrame` | Feature matrix + labels. Columns: `A_0, ..., A_{A-1}` (ancestry), `G_0, ..., G_{N-1}` (genotypes), `y` (binary label), `cluster_id` (int), `sample_index` (int, into full population) |
| `"columns"` | `list[str]` | Ordered column names of the DataFrame |

Downstream code (TabPFN evaluation, etc.) does not need to know which generative model produced the data.

### Latents pickle contents

| Key | Shape / Type | Description |
|-----|-------------|-------------|
| `"A"` | $(M, A)$ | Ancestry coordinates (full population) |
| `"G"` | $(M, N)$ | Dosage genotypes (full population) |
| `"y"` | $(M,)$ int | Binary labels (full population) |
| `"beta"` | $(M, N)$ | Per-individual interpolated effects $\beta_{i,j}$ |
| `"beta_pop"` | $(K_c, N)$ | Per-population effects (after MAF adjustment, before interpolation) |
| `"cluster_ids"` | $(M,)$ int | Cluster assignment per individual |
| `"train_indices"` | $(n_\text{train},)$ int | Row indices into full population arrays |
| `"test_indices"` | $(n_\text{test},)$ int | Row indices into full population arrays |
| `"psi"` | $(N,)$ | Per-SNP local shrinkage $\psi_j$ |
| `"delta"` | $(N,)$ | Per-SNP intermediate gamma $\delta_j$ |
| `"causal_mask"` | $(N,)$ bool | True for causal SNPs ($z_j = 1$) |
| `"maf_by_cluster"` | $(K_c, N)$ | Cluster-specific MAFs used for genotyping |
| `"maf_ancestral"` | $(N,)$ | Ancestral MAFs $p_j^\text{anc}$ |
| `"centers"` | $(K_c, A)$ | Cluster centres in ancestry space |
| `"h_per_snp"` | $(N,)$ | Per-SNP interpolation bandwidth $h_j$ |
| `"interp_weights"` | $(M, K_c, N)$ | Per-SNP interpolation weights $w_{k,j}(\mathbf{a}_i)$ |
| `"h_sq_k"` | $(K_c,)$ | Target heritability per cluster |
| `"sigma_eps"` | $(K_c,)$ | Calibrated residual noise SD per cluster |
| `"F_ST_k"` | $(K_c,)$ | $F_{ST}$ per cluster used for MAF generation |
| `"sim_config"` | dict | Full simulation config (all parameters) |
| `"setting"` | dict | Task setting (hyperparameters sampled) |
| `"seed"` | int | RNG seed |

**Row-alignment:** train table row $r$ corresponds to `train_indices[r]` in the full population arrays. Same for test.

**Key differences from HGP latents:** No `K_aa` (ancestry kernel matrix), no `drift_b`, no `kernel_info` dict. Added: `beta_pop`, `psi`, `delta`, `causal_mask`, `maf_ancestral`, `h_per_snp`, `interp_weights` (now 3D), `h_sq_k`, `sigma_eps`, `F_ST_k`.

---

## 4. Hyperparameters: What Varies Across Tasks

Each task samples the following hyperparameters. Together they define the "family" of tasks for instruction tuning.

| Hyperparameter | Range | Shape | Stage |
|----------------|-------|-------|-------|
| $M$ (individuals) | $\{10000, 15000, 20000, 25000\}$ | scalar | 0 |
| $\phi$ (global shrinkage) | $\{0.05,\, 0.1,\, 0.5,\, 1.0\}$ | scalar | 1a |
| $r_g$ (genetic correlation UB) | $\text{Uniform}(0.3, 0.9)$ | scalar | 2a |
| $\pi_\text{causal}$ (causal fraction) | $\{0.8,\, 0.9,\, 1.0\}$ | scalar | 1b |
| $\alpha$ (MAF-dep exponent) | $\{0\, (p=2/3),\, -0.25\, (p=1/3)\}$ | scalar | 4 |
| $h$ (interp bandwidth upper bound) | $\{0.3,\, 0.5,\, 1.0,\, 2.0,\, 4.0\} \times L_\text{scale1}$ | scalar | 5a |
| $h^2_k$ (heritability) | see below | $(K_c,)$ | 6b |
| $F_{ST,k}$ (genetic drift) | $\text{Uniform}(0.01, 0.25)$ | $(K_c,)$ | 3b |
| $K_c$ (num clusters) | $\{2,\, 3,\, 4,\, 5,\, 6\}$, $p = [0.10,\, 0.35,\, 0.35,\, 0.10,\, 0.10]$ | scalar | 0b |
| $N$ (num SNPs) | $\{40,\, 60,\, 80,\, 100\}$ | scalar | all |
| `cluster_sd_mult` | $\{0.004,\, 0.04,\, 0.2\}$ | scalar | 0c |
| `indiv_sd_choices` | per-cluster (see HGP logic) | $(K_c,)$ | 0c |
| `ancestry_group_cutpoints` | sampled (see HGP logic) | variable | 0b |
| `ancestry_group_size_divisors` | sampled (see HGP logic) | variable | 0b |
| $\sigma_{\beta,k}$ | $1.0$ for all $k$ (fixed) | $(K_c,)$ | 2a |

**$h^2_k$ sampling:**

$$h^2_\text{major} \sim \text{Uniform}(0.3, 0.6)$$

$$h^2_k = h^2_\text{major} \cdot U_k, \quad U_k \sim \text{Uniform}(0.5, 1.0) \quad \text{for each minority cluster } k$$

---

### Detailed Hyperparameter Effects

#### $\phi$ — Global Shrinkage

**Stage 1a:** $\delta_j \sim \text{Gamma}(b, \phi)$, then $\psi_j \sim \text{Gamma}(a, \delta_j)$.

Since $\mathbb{E}[\delta_j] = b / \phi$ and $\mathbb{E}[\psi_j \mid \delta_j] = a / \delta_j$:

| | Small $\phi$ (e.g. 0.01) | Large $\phi$ (e.g. 1.0) |
|---|---|---|
| $\mathbb{E}[\delta_j]$ | $= 0.5/0.01 = 50$ (large) | $= 0.5/1.0 = 0.5$ (small) |
| $\mathbb{E}[\psi_j]$ | $\approx 1/50 = 0.02$ (shrunk) | $\approx 1/0.5 = 2$ (large) |
| Architecture | **Sparse**: few SNPs dominate, most $\approx 0$ | **Polygenic**: many SNPs with moderate effects |
| Task | Identify the right subset | Aggregate many weak signals |

Values below 0.05 are excluded: post-GWAS-selected SNPs are enriched for true associations, so very strong sparsity priors ($\phi < 0.05$) are unrealistic. Similarly, $\pi_\text{causal} \in \{0.8, 0.9, 1.0\}$ (rather than the full $[0,1]$ range) reflects that GWAS-selection already filters toward causal variants.

---

#### $r_g$ — Cross-Population Genetic Correlation Upper Bound

**Stage 2a:** Per-SNP $r_{g,j} \sim \text{Uniform}(0.04, r_g)$, then **Stage 2b:** $C_{j,k\ell} = r_{g,j} \cdot \sigma_{\beta,k} \cdot \sigma_{\beta,\ell}$ for $k \neq \ell$.

$r_g$ is the **upper bound** for per-SNP genetic correlations. It controls the *range* of cross-population effect concordance across SNPs. Some SNPs will have nearly identical effects across populations (high $r_{g,j}$) while others will diverge substantially (low $r_{g,j}$ near 0.04), reflecting the biological reality that evolutionary constraint varies across loci.

| | Small $r_g$ (e.g. $0.3$) | Large $r_g$ (e.g. $0.9$) |
|---|---|---|
| Per-SNP $r_{g,j}$ range | $[0.04, 0.3]$ — all SNPs have weak cross-pop correlation | $[0.04, 0.9]$ — mix of divergent and concordant SNPs |
| Effect concordance | Weakly correlated across pops for all SNPs | **Different SNPs have very different cross-pop concordance** |
| Non-stationarity | Strongly non-stationary across the board | Rich heterogeneity: some SNPs stationary, others not |
| Analogy to $h$ | Similar to small $h$ (all SNPs behave similarly) | Similar to large $h$ (within-task diversity of behaviours) |

**Key insight:** A large $r_g$ doesn't mean all SNPs have high concordance — it means the *range* $[0.04, r_g]$ is wide, so some SNPs are concordant and others diverge. This per-SNP heterogeneity is richer than a single shared correlation and parallels the per-SNP bandwidth $h_j$ design in Stage 5.

---

#### $\pi_\text{causal}$ — Causal Fraction

**Stage 1b:** $z_j \sim \text{Bernoulli}(\pi_\text{causal})$.

| | $\pi_\text{causal} = 1.0$ | $\pi_\text{causal} = 0.7$ |
|---|---|---|
| Null SNPs | None | $\sim 30\%$ of features are exact zeros |
| Realism | All top GWAS hits are true | Some are false positives |
| Model task | Focus on effect heterogeneity | Must also learn feature selection |

---

#### $\alpha$ — MAF-Dependent Architecture

**Stage 4:** $\beta_{\text{pop},k,j} \leftarrow \beta_{\text{pop},k,j} \cdot [2\,p_{k,j}(1 - p_{k,j})]^{\alpha/2}$

| | $\alpha = 0$ | $\alpha = -0.25$ |
|---|---|---|
| Scaling | No-op | Rare variants amplified $\sim 1.37\times$ vs common |
| Non-stationarity | None from this mechanism | Additional NS via cluster-specific $p_{k,j}$ differences |
| Interaction | — | Amplified by larger $F_{ST,k}$ (more MAF divergence) |

---

#### $h$ — Interpolation Bandwidth Upper Bound

**Stage 5a:** $h_j \sim \text{Uniform}(h_{\text{floor}}, h)$ per SNP, then **Stage 5b:** $\tilde{w}_{k,j}(\mathbf{a}_i) = \exp(-\|\mathbf{a}_i - \mathbf{c}_k\|^2 / 2h_j^2)$

$h$ is the **upper bound** for per-SNP bandwidths. It controls the *range* of spatial smoothness across SNPs. Does **not** create non-stationarity on its own — if $r_g = 1$ and $\alpha = 0$, all $\beta_{\text{pop},k,j}$ are identical across clusters and interpolation is irrelevant regardless of $h_j$.

| | Small $h$ (e.g. $0.3 \times L_\text{scale1}$) | Large $h$ (e.g. $4.0 \times L_\text{scale1}$) |
|---|---|---|
| Per-SNP $h_j$ range | $[0.01, 0.3]$ — all SNPs have relatively sharp profiles | $[0.01, 4.0]$ — wide mix of sharp and very smooth |
| Effect heterogeneity | All SNPs have similar (sharp) spatial profiles | **Different SNPs have very different spatial profiles** |
| Limit (all $h_j$ small) | Recovers PRS-CSx discrete-pop model | — |
| Limit (all $h_j$ large) | — | Effects nearly constant across ancestry |

**Key insight:** A large $h$ doesn't just mean smooth blending — it means the *range* $[0.01, h]$ is wide, so some SNPs are sharp and others are smooth. This within-task heterogeneity is richer than a single shared bandwidth and parallels the HGP's per-SNP lengthscale multipliers.

**Per-variant effect:** When $h_j$ is sampled large, the interpolation kernel for SNP $j$ is flat across clusters, so $\beta_{i,j} \approx \bar{\beta}_{\cdot,j}$ (the mean population effect) for all individuals regardless of ancestry — the effect-size distribution for that variant is homogeneous across the sample. When $h_j$ is sampled small, the kernel is sharp and each individual's weight concentrates on the nearest cluster centre, giving $\beta_{i,j} \approx \beta_{\text{pop},k_{\text{nearest}},j}$ — the effect-size distribution for that variant splits into distinct per-cluster modes, reflecting cross-population differences.

---

#### $h^2_k$ — Per-Population Heritability

**Stage 6b:** $\sigma_{\epsilon,k}^2 = \widehat{\text{Var}}(g) \cdot (1 - h^2_k) / h^2_k$ (global $\widehat{\text{Var}}(g)$ — see Stage 6b rationale)

| | High $h^2_k$ (e.g. 0.5) | Low $h^2_k$ (e.g. 0.25) |
|---|---|---|
| Noise level | $\sigma_\epsilon^2 = \widehat{\text{Var}}(g)$ (equal to global signal) | $\sigma_\epsilon^2 = 3 \times \widehat{\text{Var}}(g)$ (noise dominates) |
| Classification | Easier | Harder |
| Non-stationarity | More detectable | Masked by noise |

**Different $h^2_k$ across clusters:** having $h^2_\text{major} = 0.5$ and $h^2_\text{minority} = 0.25$ means minorities get more noise (lower SNR) relative to the same global genetic variance baseline — the model must learn to borrow strength. Crucially, since $h^2_\text{major} \geq h^2_\text{minority}$ is enforced by the sampling procedure (Section 4), majority always has cleaner labels after Stage 6b, which is the intended design.

---

#### $F_{ST,k}$ — Per-Cluster Genetic Drift

**Stage 3b:** $p_{k,j} \sim \text{Beta}\!\left(p_j^\text{anc} \cdot \tfrac{1-F}{F},\; (1-p_j^\text{anc}) \cdot \tfrac{1-F}{F}\right)$

The variance of allele frequency around ancestral:

$$\text{Var}(p_{k,j}) = F_{ST,k} \cdot p_j^\text{anc} (1 - p_j^\text{anc})$$

| $F_{ST,k}$ | Analogy | Effect |
|------------|---------|--------|
| 0.01 | Closely related (N. vs S. Europeans) | MAFs nearly identical → minimal genotype domain shift |
| 0.05–0.10 | Continental (EUR vs EAS) | Moderate divergence → realistic cross-ancestry scenario |
| 0.15–0.25 | Highly diverged (EUR vs AFR) | Large MAF shifts → strong genotype-level domain shift |
| Combined with $\alpha \neq 0$ | — | Also creates non-stationarity in effective effect magnitudes |

---

#### $K_c$ — Number of Clusters

| | $K_c = 2$ | $K_c = 6$ |
|---|---|---|
| Structure | One majority, one minority | Complex multi-group hierarchy |
| Non-stationarity | Single jump | Rich multi-population patterns |
| Interpolation in Stage 5 | Two-point blend | Six-point blend; more admixed individuals possible |

---

#### `cluster_sd_mult` — Within-Cluster Ancestry Spread

**Stage 0c:** $L_\text{scale2} = \texttt{cluster\_sd\_mult} \times L_\text{scale1}$

| | Small (0.004) | Large (0.2) |
|---|---|---|
| Clusters | Tight, minimal overlap | Wide, substantial overlap |
| Ancestry features | Clearly indicate cluster $\to$ easy ID | Ambiguous $\to$ hard cluster ID |
| Interpolation | Weights hard-assign to nearest | Weights split across clusters |
| Admixture | Very few admixed individuals | Many effectively admixed |

---

### Admixture and Cluster Overlap

In real human genetics, individuals are often **admixed** — they have ancestry contributions from multiple population groups. In our simulation, admixture manifests as individuals whose ancestry coordinates $\mathbf{a}_i$ fall in **overlap regions** between clusters, causing their interpolation weights $w_k(\mathbf{a}_i)$ to be split across multiple clusters rather than dominated by one.

#### When does overlap happen?

Overlap between clusters $k$ and $\ell$ depends on the ratio of their combined spread to their separation:

$$\text{overlap}(k, \ell) \;\propto\; \frac{\sigma_k^\text{cluster} + \sigma_\ell^\text{cluster}}{\|\mathbf{c}_k - \mathbf{c}_\ell\|}$$

Since $\sigma_k^\text{cluster} = m_k \cdot \texttt{cluster\_sd\_mult} \cdot L_\text{scale1}$ and inter-centre distances are $\sim L_\text{scale1}$, the ratio simplifies to:

$$\text{overlap}(k, \ell) \;\propto\; (m_k + m_\ell) \cdot \texttt{cluster\_sd\_mult}$$

So the **two knobs** controlling admixture are:

1. **`cluster_sd_mult`** (dominant): scales all cluster widths uniformly.
2. **`indiv_sd_choices`** / the assigned multipliers $m_k$: create asymmetric spreads.

#### Settings that produce admixed individuals

| Setting | `cluster_sd_mult` | $K_c$ | Effect |
|---------|-------------------|--------|--------|
| **Tight, well-separated** | 0.004 | any | $\sigma_k \approx 0.004$. Overlap $\approx 0.008/1.0 = 0.8\%$ of inter-centre distance. Virtually no admixed individuals. Interpolation weights $\to$ hard assignment. |
| **Moderate overlap** | 0.04 | 3–4 | $\sigma_k \approx 0.04\text{–}0.08$. Overlap $\approx 8\text{–}16\%$. Some individuals in boundary regions; interpolation weights meaningfully split. |
| **Heavy overlap / truly admixed** | 0.2 | 4–6 | $\sigma_k \approx 0.2\text{–}0.4$. Overlap $>40\%$. Many individuals have ancestry coordinates between multiple centres. Interpolation weights are diffuse; these individuals are effectively **multi-ancestry admixed**. Cluster identity from ancestry features alone becomes ambiguous. |

#### What admixture means for the learning task

When cluster overlap is high:

- **Ancestry features** ($A_0, \ldots, A_{A-1}$) no longer cleanly separate populations → the model cannot simply learn "if ancestry is near centre $k$, use cluster $k$'s effects."
- **Effect sizes** for admixed individuals are **blends** of multiple populations' effects (via Stage 5 interpolation), so the model must learn to combine information from multiple population-specific effect patterns.
- **Genotype features** ($G_0, \ldots, G_{N-1}$) are still generated from the individual's **hard** cluster assignment (Stage 3c uses $p_{k_i, j}$), creating a mismatch: the individual's allele frequencies reflect their assigned cluster, but their effect sizes reflect a blend. This mimics real admixture where an individual's genotype distribution follows local ancestry blocks, while the phenotypic effect of each allele depends on the overall genetic background.

#### Interaction with $h$ / per-SNP $h_j$ (interpolation bandwidth)

Even with tight clusters (`cluster_sd_mult` = 0.004), SNPs with large $h_j$ will still produce blended effects for individuals near cluster boundaries — but there will be very few such individuals. Conversely, even with high overlap (`cluster_sd_mult` = 0.2), SNPs with small $h_j$ will produce sharp effect transitions, creating a scenario where physically "admixed" individuals experience abrupt, discrete-population-like effects for those particular loci.

With per-variant bandwidths, an admixed individual may simultaneously experience:
- **Sharp** effects at loci with small $h_j$ (effectively assigned to the nearest population)
- **Blended** effects at loci with large $h_j$ (smoothly interpolated across populations)

This heterogeneity within individuals is a realistic feature of admixture, where local ancestry varies across the genome.

The richest admixture scenarios combine:
- `cluster_sd_mult` $\geq 0.04$ (physical overlap in ancestry space)
- $h \geq 0.5 \times L_\text{scale1}$ (wide range of per-SNP bandwidths, so some loci are sharp and others smooth)
- $r_g < 1$ (these blends actually matter because population effects differ)

---

## Non-Stationarity Summary

Three mechanisms create effect-size non-stationarity (effects differ across the ancestral continuum), listed in order of importance:

| # | Mechanism | Controlled by | Stage | Creates NS alone? |
|---|-----------|---------------|-------|-------------------|
| 1 | Cross-population effect divergence | $r_g$ | 2 | **Yes** (primary) |
| 2 | MAF-dependent effect rescaling | $\alpha \neq 0$ with variable $F_{ST,k}$ | 4 | **Yes** (even if $r_g = 1$) |
| 3 | Per-SNP interpolation heterogeneity | $h$ (→ per-SNP $h_j$) | 5 | **No** (shapes #1 and #2, but adds within-individual heterogeneity) |

Richest non-stationarity: use all three together — $r_g < 1$ creates different raw effects, $\alpha \neq 0$ adds MAF-mediated differences on top, and per-variant $h_j$ (drawn from $[0.01, h]$) ensures different SNPs have different spatial smoothness, so within a single individual some effects are sharp/population-specific while others are smoothly blended.

---

## Comparison with HGP Pipeline

| Aspect | HGP Pipeline | This Pipeline (PRS-CSx) |
|--------|-------------|------------------------|
| Effect-size prior | GP (smooth, dense, kernel-correlated) | Continuous shrinkage (sparse, heavy-tailed) |
| Which SNPs matter | All SNPs vary (lengthscale-dependent) | Explicit causal fraction $\pi_\text{causal}$; nulls are exact 0 |
| Non-stationarity mechanism | Per-SNP GP lengthscale in ancestry kernel | Cross-pop $r_g < 1$ + MAF-dep $\alpha$ + per-SNP interpolation $h_j$ |
| Allele frequency model | Uniform cluster offset | Balding–Nichols ($F_{ST}$) |
| Cross-pop coupling | GP kernel correlations | Shared $\psi_j$ + explicit $r_g$ |
| Sparsity | Implicit/absent | Explicit via $\phi$ hierarchy |
| MAF-dep architecture | Not modelled | Optional $\alpha$ parameter |
| Heritability control | `logit_noise_sd` (indirect) | Per-cluster $h^2_k$ (direct) |
| Output structure | train/test/latents pickles | **IDENTICAL** train/test/latents pickles |
