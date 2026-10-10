# CIB-IRL: Causal Information Bottleneck Inverse Reinforcement Learning

**Unsupervised image clustering that ignores spurious correlations, by jointly learning a causal representation (Information Bottleneck) and a latent reward (multi-intention Inverse RL).**

> **Status:** research prototype, work in progress. The conceptual framework is stable; the implementation and benchmarks are being built (see [Roadmap](#9-status--roadmap)).

---

## Table of Contents

1. [TL;DR](#1-tldr)
2. [Motivation](#2-motivation)
3. [Background & Reading List](#3-background--reading-list)
4. [Method](#4-method)
5. [Open Design Questions](#5-open-design-questions)
6. [Evaluation Protocol](#6-evaluation-protocol)
7. [Ablations](#7-ablations)
8. [Repository Structure](#8-repository-structure-planned)
9. [Status & Roadmap](#9-status--roadmap)
10. [Citation & License](#10-citation--license)

---

## 1. TL;DR

Clustering images by raw distance (K-Means, GMM) or by contrastive features (SimCLR, MoCo) tends to group images by **what is easy to see** (background, texture, style) rather than **what matters** (the object). CIB-IRL fixes this by recasting clustering as a **single-step decision problem**:

- The *state* is a compressed latent code `z = f_φ(x)` of the image.
- The *action* is a cluster assignment `a ∈ {1, …, K}`.
- A *global latent reward* `R_θ(z, a, m)`, conditioned on a latent intent/concept `m`, explains why a given assignment is "good".
- An **Information Bottleneck** on the encoder removes any information that the reward does not need. A feature is *useless* (spurious) if removing it leaves the categorization reward unchanged.

Encoder, intent predictor and reward are trained **jointly, end to end**.

---

## 2. Motivation

### 2.1 What already exists

| Family | Examples | Strength | Limitation for our problem |
|---|---|---|---|
| Geometric / statistical clustering | K-Means, GMM | Simple, fast | Depends entirely on the embedding geometry |
| Self-supervised learning | SimCLR, MoCo v3, CLIP | Strong, structured latent spaces | Augmentations alone do not guarantee that backgrounds are discarded |
| Multi-intention / meta IRL | Babeş-Vroman et al. 2011, Meta-IRL | Separates unlabeled demonstrations by inferring several reward functions | Designed for **temporal trajectories**, not static images |

### 2.2 The gap

1. **Modality mismatch.** Multi-intention IRL methods operate on sequences of states and actions. A static image is categorized in a *single step*, so these methods cannot be applied directly.
2. **Spurious correlations.** When applied to images, such methods inherit the shortcut-learning behavior of unsupervised learners. An agent grouping animal images will often cluster by background (e.g. ducks and boats together because both contain water) instead of by the causal features of the object.

### 2.3 Our idea

We reformulate static image clustering as a **single-step MDP driven by an implicit global reward**, and we pair the IRL pipeline with a **learned causal encoder constrained by an Information Bottleneck**, trained jointly with the reward. Compression removes what the reward does not use; the reward, in turn, is shaped by what the encoder keeps.

---

## 3. Background & Reading List

Five papers cover the theoretical foundations of the framework:

1. **Meta-Inverse Reinforcement Learning with Probabilistic Context Variables**, Yu, Yu, Finn, Ermon (NeurIPS 2019).
   *Why:* extracts an implicit reward tied to a latent context variable `m` in a multi-task setting.
2. **Deep Variational Information Bottleneck**, Alemi, Fischer, Dillon, Murphy (ICLR 2017).
   *Why:* compress the input while retaining task-relevant information, via variational bounds on mutual information.
3. **Invariant Risk Minimization**, Arjovsky, Bottou, Gulrajani, Lopez-Paz (2019).
   *Why:* foundational theory of causal invariance across environments (here: styles/backgrounds).
4. **Invariant Information Clustering for Unsupervised Image Classification and Segmentation**, Ji, Henriques, Vedaldi (ICCV 2019).
   *Why:* maximizing mutual information between augmented views to drive unsupervised clustering.
5. **Apprenticeship Learning About Multiple Intentions**, Babeş-Vroman, Marivate, Subramanian, Littman (ICML 2011).
   *Why:* early demonstration that clustering and IRL can be optimized *jointly* rather than sequentially.

Datasets and baselines referenced below: Sagawa et al. 2020 (Waterbirds / Group DRO), Chen et al. 2020 (SimCLR), Chen, Xie, He 2021 (MoCo v3).

---

## 4. Method

### 4.1 Pipeline

```mermaid
flowchart TD
    x["Input image x<br/>(content + spurious factors)"] --> enc["Causal encoder f_φ<br/>stochastic, IB-regularized"]
    enc --> z["Causal latent z"]
    z --> qpsi["Intent predictor q_ψ(m | z)"]
    qpsi --> m["Latent context m"]
    z --> rew["Global reward R_θ(z, a, m)"]
    m --> rew
    rew --> pi["Policy π(a | z, m) ∝ exp R_θ(z, a, m)"]
    pi --> a["Cluster assignment a ∈ {1..K}"]
    enc -. "compression: I(z; x)" .-> ib["Information Bottleneck"]
    ib -. "β" .-> enc
```

### 4.2 Single-step MDP

| Component | Definition |
|---|---|
| State `s` | Latent vector `z = f_φ(x)` extracted from the raw image `x` |
| Action `a` | Cluster index, `a ∈ {1, …, K}` |
| Latent context `m` | Continuous semantic variable for the underlying concept (e.g. species morphology), inferred by `q_ψ(m \| z)` |
| Reward | `R_θ(z, a, m)`, a single global reward shared across the dataset |
| Policy | Maximum-entropy optimal policy. With a single step there is no horizon, so `Q = R` and `π(a \| z, m) = exp R_θ(z, a, m) / Σ_a' exp R_θ(z, a', m)` |

### 4.3 Joint objective

Encoder `f_φ`, intent predictor `q_ψ` and reward `R_θ` are optimized together:

$$
\max_{\phi,\psi,\theta}\;\; \mathcal{I}\big(f_\phi(x);\, a, m\big) \;-\; \beta\,\mathcal{I}\big(f_\phi(x);\, x\big)
$$

- **Relevance term** `I(f_φ(x); a, m)`: mutual information between the filtered representation and the categorization output under the inferred reward. It keeps in `z` every causal marker needed for clear, confident clustering decisions.
- **Compression term** `I(f_φ(x); x)`: forces the encoder to be as restrictive as possible, deleting bits that vary a lot or do not help clustering (changing backgrounds, styles).
- **`β`** controls the compression / utility trade-off.

The agent's policy `π(a | z, m)` maximizes the extracted reward `R_θ(z, a, m)`.

### 4.4 Tractable training objective

As in the Deep VIB, both terms are replaced by variational bounds. With a stochastic encoder `p_φ(z|x)` and a fixed prior `r(z)`:

$$
I(z;x) \;\le\; \mathbb{E}_{x}\,\mathrm{KL}\big(p_\phi(z\mid x)\,\|\,r(z)\big),
\qquad
I(z;a,m) \;\ge\; H(a,m) + \mathbb{E}\big[\log q_\psi(m\mid z) + \log \pi_\theta(a\mid z,m)\big]
$$

yielding the loss to minimize:

$$
\mathcal{L}(\phi,\psi,\theta) \;=\; -\,\mathbb{E}\big[\log q_\psi(m\mid z) + \log \pi_\theta(a\mid z,m)\big] \;+\; \beta\,\mathbb{E}_x\,\mathrm{KL}\big(p_\phi(z\mid x)\,\|\,r(z)\big) \;+\; \lambda\,\mathcal{L}_{\text{inv}}
$$

### 4.5 Reward-consistency (invariance) term: *proposed*

A feature is spurious if removing it leaves the reward unchanged. We operationalize this with a consistency penalty under nuisance interventions `t ~ 𝒯` (background swaps, color jitter, style transfer):

$$
\mathcal{L}_{\text{inv}} \;=\; \mathbb{E}_{x,\,t\sim\mathcal{T}}\,\Big\| R_\theta\big(f_\phi(x),\cdot,m\big) - R_\theta\big(f_\phi(t(x)),\cdot,m\big)\Big\|^2
$$

This is the bridge to IRM (invariance across environments) and IIC (agreement across views). Its exact form is an open design choice and is covered by the ablations in [Section 7](#7-ablations).

---

## 5. Open Design Questions

These are the points the implementation must settle, and the ones a reviewer is most likely to ask about:

1. **What plays the role of "demonstrations"?** Classical IRL needs expert behavior. In an unsupervised setting, the supervision signal has to come from somewhere: our current candidate is cross-view consistency of assignments (IIC-style), so that the "expert" is the implicit labeling that stays stable under nuisance transformations.
2. **Self-referential relevance term.** `a` and `m` are produced by the model itself, so maximizing `I(z; a, m)` alone admits trivial or degenerate solutions. The consistency term and a cluster-balance regularizer (e.g. a marginal-entropy bonus on `a`) are the proposed safeguards.
3. **Identifiability of `m`.** What guarantees that `m` captures a semantic concept rather than another nuisance factor? Diagnostics: NMI between `m` and the known spurious attribute.
4. **Choice of `K`.** Fixed and known in the benchmarks; sensitivity to a wrong `K` is part of the ablations.

---

## 6. Evaluation Protocol

### 6.1 Datasets (engineered spurious correlations)

| Dataset | Setup | What it tests |
|---|---|---|
| **Colored MNIST** | Digits are strongly correlated with background/foreground color at training time; correlation is randomized or inverted at test time. 10-class variant for clustering; the binary IRM variant as a sanity check. | Whether the model clusters by digit identity instead of color |
| **Waterbirds** | Landbirds on land and waterbirds on water (strong train-time correlation); minority groups (landbird on water, waterbird on land) at test time. | Whether the model clusters by bird category instead of land/water background |

### 6.2 Baselines

1. **Geometric / statistical clustering:** K-Means and GMM on standard ResNet / ViT embeddings.
2. **Pure contrastive learning:** SimCLR, MoCo v3 (and IIC) followed by clustering, to see whether contrastive augmentations alone discard backgrounds.
3. **Standard multi-intention IRL:** a temporal IRL model adapted to single-step images **without** the causal Information Bottleneck encoder.

### 6.3 Metrics

- **Clustering Accuracy (ACC)** with optimal (Hungarian) cluster-to-label matching, and **Normalized Mutual Information (NMI)**, both measured strictly against the true semantic labels (bird species, digit ID), ignoring background.
- **Out-of-distribution (OOD) robustness:** drop in performance (`ΔACC`) between the correlated test set and the test set where the background correlation is inverted.
- **Worst-group ACC** (Waterbirds): accuracy on the least-represented (class, background) group.
- **Spurious leakage:** `NMI(cluster, background)`. Lower is better; a model that clusters by background scores high.

Reporting: mean ± std over several seeds, with fixed `K` equal to the number of true classes.

---

## 7. Ablations

| Ablation | Question answered |
|---|---|
| `β = 0` (no Information Bottleneck) | Does compression alone remove the spurious features? |
| `λ = 0` (no reward-consistency term) | Is the invariance term necessary, or is the IB sufficient? |
| Sequential training (encoder → IRL) instead of joint | Does joint optimization matter, as Babeş-Vroman et al. suggest? |
| `β` sweep | Where is the compression / utility trade-off, and is there a sweet spot? |
| Wrong `K` | How sensitive is the method to the number of clusters? |
| Without latent context `m` | What does the intent variable contribute? |

---

## 8. Repository Structure (planned)

```
cib-irl/
├── README.md
├── configs/          # experiment configs (datasets, β, λ, K)
├── cibirl/
│   ├── encoder.py    # causal encoder f_φ (stochastic, VIB head)
│   ├── intent.py     # intent predictor q_ψ(m | z)
│   ├── reward.py     # global reward R_θ(z, a, m) and policy π
│   ├── losses.py     # variational IB bound, consistency term
│   └── train.py      # joint optimization loop
├── data/             # Colored MNIST, Waterbirds loaders
├── baselines/        # K-Means/GMM, SimCLR, MoCo v3, IIC, temporal IRL
├── eval/             # ACC, NMI, OOD drop, worst-group, leakage
└── scripts/          # launchers (local and SLURM)
```

---

## 9. Status & Roadmap

- [x] Problem statement and literature review
- [x] Architecture and joint objective
- [ ] Variational formulation validated (Section 4.4) and consistency term fixed (Section 4.5)
- [ ] Colored MNIST pipeline and baselines
- [ ] Waterbirds pipeline and baselines
- [ ] Full ablation suite
- [ ] Write-up

---

## 10. Citation & License

```bibtex
@misc{cibirl,
  title  = {CIB-IRL: Causal Information Bottleneck Inverse Reinforcement Learning},
  author = {TODO},
  year   = {2026},
  note   = {Work in progress}
}
```

License: *TODO*.
