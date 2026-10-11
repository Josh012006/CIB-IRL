# CIB-IRL: Causal Information Bottleneck Inverse Reinforcement Learning

**Unsupervised image clustering that groups images by *what they are*, not by *where they appear*, by learning a causal representation (Information Bottleneck) together with a latent reward (multi-intention Inverse RL) from experts built out of image transformations, and a policy that builds the clusters by reinforcement learning.**

> **Status:** research prototype, work in progress. The conceptual framework is stable; the implementation and benchmarks are being built (see [Roadmap](#11-status--roadmap)).

---

## Table of Contents

1. [The Idea](#1-the-idea)
2. [Motivation](#2-motivation)
3. [Background & Reading List](#3-background--reading-list)
4. [Method](#4-method)
5. [Where Do the Experts Come From?](#5-where-do-the-experts-come-from)
6. [Components & Training](#6-components--training)
7. [Decisions & Open Questions](#7-decisions--open-questions)
8. [Evaluation Protocol](#8-evaluation-protocol)
9. [Ablations](#9-ablations)
10. [Repository Structure](#10-repository-structure-planned)
11. [Status & Roadmap](#11-status--roadmap)
12. [License](#12-license)

---

## 1. The Idea

**In one sentence:** cluster images the way a child learns categories, by finding the few features that make things *distinguishable* and ignoring everything else.

### 1.1 Intuition

We recognize a cat from a handful of features (ears, whiskers, the shape of the face), not from the sofa it sits on. A baby does this without being taught any label: it simply learns to tell things apart. We want an algorithm that does the same, i.e. groups images by what they *are*, even when the background or style would be an easier (but meaningless) signal.

### 1.2 Clustering as decision-making (the IRL view)

We turn clustering into a sequential decision problem:

- **The policy** is the act of clustering. Images arrive one after the other; for each, the policy decides which group it goes into: one of the clusters **built so far**, or a fresh one. Its decision is conditioned on what already exists, as in any unsupervised method that builds its groups along the way. It is a policy in the RL sense, a separate network trained to maximize a return, **not** a softmax of the reward.
- **The reward** is the intention behind the decision, the *intention of distinction*. It scores a placement by how well the image fits the group it joins **and** how well it is told apart from the other groups. As in any RL problem, the reward says what is worth doing and the policy learns how to do it over a whole sequence of decisions.
- **The expert.** We can see the creator of the data (the world that produced the cats and the dogs) as an expert who, in building these images, optimized some reward. Multi-intention IRL methods such as PEMIRL recover from expert behavior the reward attached to each category.
- **The intent.** As in Meta-IRL, a group of images has an *intent*: what its members have in common (the context variable `m`). The memory of the groups built so far is made of these intents, so the reward and the policy compare the intent of the arriving image with the intent of each existing group.
- **Our step beyond.** Besides the reward that makes each category what it is, there also exists a reward that **distinguishes categories from one another**, the reason we do not confuse them. Finding it *is* learning to distinguish. Optimizing it by RL, over decisions that depend on the clusters already formed, is clustering.

### 1.3 The obstacle, and our answer

IRL needs expert demonstrations, but an unsupervised problem has none (a supervised version would not be interesting). Our answer is to **build the experts from the images themselves**:

- all the transformed versions of an image (crops, color changes, new backgrounds, transformations of transformations) obviously show the same thing: when one of them arrives and another is already in a group, the expert demonstrates "join that group";
- two random images usually show different things: the expert demonstrates "do not put these together".

The expert never says *which* cluster in absolute terms; it only answers "same group or not?". That is enough to learn a reward that tells clusters apart. These decisions are produced on the fly, while the policy builds an episode one image at a time: the data pipeline generated the views, so it knows which ones share an orbit and can label every placement the policy makes. Details in [Section 5](#5-where-do-the-experts-come-from).

### 1.4 Two safeguards against shortcuts

Image datasets are full of spurious cues (background, texture, style) that are easier to use than the object. Two ingredients keep the reward on the object:

- **Information Bottleneck.** An encoder is trained to keep as much information as the reward needs and to compress away the rest, so a feature that does not help the decision is deleted.
- **Invariance check.** If the encoder really removed the useless features, the decision must not change when the image is perturbed in ways that keep the object (e.g. new background). We add this as an explicit term.

### 1.5 At a glance

| Human intuition | In CIB-IRL |
|---|---|
| The few features that make a cat a cat | Causal latent `z` (encoder + Information Bottleneck) |
| "This looks like a cat-type thing" | Latent intent `m` (intent predictor `q_ψ`: one per orbit during training, one per image at test time) |
| The groups I have formed so far | Cluster memory `C`: for each cluster, the average intent of its members (empty slots included) |
| Why this image belongs here and not there | Reward `R_θ(z, m, C, a)` |
| Putting it in a group (or starting a new one) | Policy `π_ω(a \| z, m, C)`, `a ∈ {1..K}`, trained by RL |
| Someone who already knows what goes together | Expert built from transformations ([Section 5](#5-where-do-the-experts-come-from)) |
| Not being fooled by the background | Compression + invariance of the decision |

---

## 2. Motivation

### 2.1 What already exists

| Family | Examples | Strength | Limitation for our problem |
|---|---|---|---|
| Geometric / statistical clustering | K-Means, GMM | Simple, fast | Depends entirely on the embedding geometry |
| Self-supervised learning | SimCLR, MoCo v3, CLIP | Strong, structured latent spaces | Augmentations alone do not guarantee that backgrounds are discarded |
| Multi-intention / meta IRL | Babeş-Vroman et al. 2011, PEMIRL | Separates unlabeled demonstrations by inferring several reward functions | Designed for **temporal trajectories** of real experts, not static images |

### 2.2 The gap

1. **Modality mismatch.** Multi-intention IRL methods operate on sequences of states and actions produced by experts. A static image has no trajectory and nobody demonstrates the choice. We create the sequence ourselves: the *clustering process* is the trajectory (images are assigned one after the other, and each assignment changes the state), and the expert demonstrations are fabricated from transformations.
2. **Spurious correlations.** Applied to images, such methods inherit the shortcut-learning behavior of unsupervised learners. An agent grouping animal images will often cluster by background (ducks and boats together because both contain water) instead of by the causal features of the object.
3. **A single reward.** Most of the IRL based methods don't actually use the IRL itself for the unsupervised clustering part. But rather they end up with a bunch of rewards, each representing a different cluster's features. Our approach tries to encode the concept of distinguishing in a single reward.

---

## 3. Background & Reading List

Six papers cover the theoretical foundations of the framework:

- [ ] Yu, L., Yu, T., Finn, C., & Ermon, S. (2019).
  *Meta-Inverse Reinforcement Learning with Probabilistic Context Variables*. NeurIPS 2019.
  [PDF](https://arxiv.org/pdf/1909.09314.pdf)
  Core multi-intention IRL method — learns rewards from unlabeled multi-task demonstrations
  through a latent context variable `m` inferred from the demonstrations, and infers the reward
  of a new task from a single demonstration. Source of our intent `m` (Section 4.2).

- [ ] Alemi, A. A., Fischer, I., Dillon, J. V., & Murphy, K. (2017).
  *Deep Variational Information Bottleneck*. ICLR 2017.
  [PDF](https://arxiv.org/pdf/1612.00410.pdf)
  Core compression method — turns the Information Bottleneck into a trainable objective with
  a variational bound on `I(z; x)` (a KL to a prior) and a bound on relevance. Gives the
  `β·KL` term of our encoder (Section 4.4).

- [ ] Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2019).
  *Invariant Risk Minimization*. arXiv.
  [PDF](https://arxiv.org/pdf/1907.02893.pdf)
  Foundational theory of causal invariance — features whose predictive relation is stable across
  environments are the causal ones. Frames our nuisance family `𝒯` as "environments", and
  introduces Colored MNIST.

- [ ] Ji, X., Henriques, J. F., & Vedaldi, A. (2019).
  *Invariant Information Clustering for Unsupervised Image Classification and Segmentation*. ICCV 2019.
  [PDF](https://arxiv.org/pdf/1807.06653.pdf)
  Closest image-clustering approach in spirit — maximizes the mutual information between the
  cluster assignments of two augmented views of the same image. Our forced moves and
  invariance term play a similar role in a different formalism.

- [ ] Babeş-Vroman, M., Marivate, V., Subramanian, K., & Littman, M. (2011).
  *Apprenticeship Learning About Multiple Intentions*. ICML 2011.
  [Paper page](https://mlanthology.org/icml/2011/babes2011icml-apprenticeship)
  Early "clustering and IRL together" — clusters unlabeled trajectories by inferring one
  objective per cluster with an EM procedure over IRL. Closest ancestor of the idea that a
  cluster is defined by its reward.

- [ ] Le-Khac, P. H., Healy, G., & Smeaton, A. F. (2020).
  *Contrastive Representation Learning: A Framework and Review*. IEEE Access, 8.
  [PDF](https://arxiv.org/pdf/2010.05113.pdf)
  Core contrastive-learning overview — a review that unifies the methods (positive and negative
  pairs, loss functions, families of methods) and gives the vocabulary used by the other
  papers. Enough to understand what our `L_pos` and `L_neg` borrow from contrastive learning
  (Section 5.3).

Additional references used by the expert design ([Section 5](#5-where-do-the-experts-come-from)): **Debiased Contrastive Learning** (Chuang et al., NeurIPS 2020), **SCAN: Learning to Classify Images without Labels** (Van Gansbeke et al., ECCV 2020), **SimCLR** (Chen et al., 2020), **MoCo v3** (Chen, Xie, He, 2021), and **Waterbirds / Group DRO** (Sagawa et al., 2020).

---

## 4. Method

### 4.1 Pipeline

```mermaid
flowchart TD
    x["Input image x<br/>(content + spurious factors)"] --> enc["Causal encoder f_φ<br/>stochastic, IB-regularized"]
    enc --> z["Causal latent z"]
    z --> qpsi["Intent predictor q_ψ(m | z)"]
    qpsi --> m["Latent context m"]
    z --> pol["Policy π_ω(a | z, m, C)<br/>trained by RL"]
    C["Cluster memory C<br/>(average intent of the members of each cluster<br/>built so far, empty slots included)"] --> pol
    m --> pol
    pol --> a["Assignment a ∈ {1..K}<br/>existing cluster, or empty slot = new cluster"]
    a -- "transition: update the summary of cluster a" --> C
    z --> rew["Reward R_θ(z, m, C, a)<br/>fit to the experts (IRL)"]
    C --> rew
    m --> rew
    m -- "averaged into the summary of the chosen cluster" --> C
    exp["Experts<br/>(orbits, random pairs)"] -. "IRL: fit the reward" .-> rew
    rew -. "RL: maximize cumulative return" .-> pol
    enc -. "compression: I(z; x)" .-> ib["Information Bottleneck"]
    ib -. "β" .-> enc
```

### 4.2 Episodic MDP over a growing set of clusters

An *episode* is one sweep over a minibatch of views, presented one after the other in random order. The cluster memory `C` persists across episodes and across training: it is the set of clusters built so far.

| Component | Definition |
|---|---|
| State `s_t` | `(z_t, m_t, C_t)`. `z_t = f_φ(x_t)` is the code of the image arriving at step `t` and `m_t = μ_ψ(z_t)` is its inferred intent. `C_t = {(n_k, c_k)}_{k=1..K}` is the **cluster memory**: for each slot `k`, the number of members `n_k` and the **average intent** `c_k` of its members. An empty slot (`n_k = 0`) carries a learned "empty" embedding. `C` is a memory, not a differentiable function: it is updated by the transition and is stop-gradient everywhere |
| Action `a_t` | A slot `a ∈ {1, …, K}`. Choosing a non-empty slot means *joining* that cluster. Choosing an empty slot means *opening a new cluster*. `K` is an **upper bound**; the number of clusters actually constructed starts at 0 and grows during training |
| Transition | The chosen slot is updated: `n_a ← n_a + 1`, `c_a ← c_a + (m_t − c_a)/n_a` (running mean of the member intents, or a slow EMA once a slot is mature). The next image then sees a different state |
| Reward | `R_θ(s_t, a_t)`: compatibility of `z_t` with the chosen group **and** distinctness from the other groups. One network, permutation-equivariant over slots. Learned by IRL from the experts ([Section 5](#5-where-do-the-experts-come-from)) |
| Policy | `π_ω(a \| s_t)`, a **separate network** trained by RL to maximize the entropy-regularized cumulative return `E[Σ_t γ^t R_θ(s_t, a_t)] + α·H(π_ω)`. It is *not* `softmax(R_θ)` |
| Expert | Not observed: built from image transformations ([Section 5](#5-where-do-the-experts-come-from)) |

**What `m` is, and why the cluster memory is made of intents.** In Meta-IRL (PEMIRL), the context `m` is inferred from a *set* of demonstrations that share a task, and it indexes the reward of that task. Our counterpart of a set of demonstrations is a **group of images**: an orbit at first, then a cluster that gathers orbits. The `m` of a group is its **intent**, what its members have in common. This gives two levels:

- **Per image, `m_u = μ_ψ(z_u)`.** The intent that a single image expresses. During training `q_ψ` is pushed to give the same intent to all views of an orbit (Section 6.2), so `m` is the orbit-consistent, nuisance-free part of the code. At test time it is computed from one image, as PEMIRL infers the context of a new task from a single demonstration.
- **Per cluster, `c_k`.** The average of the intents of its members. The intent of a cluster *is* the summary stored in the memory.

The reward and the policy therefore receive no extra label: they compare the intent of the arriving image with the intents of the existing clusters. We do not take `m` to be the cluster index (a distribution over slots), because that would duplicate the output of the policy. One honest remark: `m` is a function of `z`, so it adds no information. What it adds is an extra invariance signal (orbit consistency, `L_ψ`) and a space in which clusters are compared. The ablations "without `m`" and "summaries in `z`-space" measure it ([Section 9](#9-ablations)).

**What RL brings that a classifier does not.** The reward is a *one-step* score: it says whether placing this image here is good *given the clusters as they currently are*. A placement also changes the cluster summary, hence the rewards of all later images. The policy optimizes this whole chain: it can leave an uncertain image in a fresh slot instead of polluting an existing summary, or open a cluster early because several later images will need it. If `γ = 0` the return is the immediate reward and the policy is essentially greedy on `R_θ`; the ablation `γ = 0` / `softmax(R_θ)` ([Section 9](#9-ablations)) measures what the sequential structure really adds.

### 4.3 Joint objective

Encoder `f_φ`, intent predictor `q_ψ` and reward `R_θ` are optimized together (the policy `π_ω` is trained against the reward, Section 6.5):

$$
\max_{\phi,\psi,\theta} \mathcal{I}\big(f_\phi(x); a, m\big) - \beta\mathcal{I}\big(f_\phi(x); x\big)
$$

- **Relevance term** `I(f_φ(x); a, m)`: mutual information between the filtered representation and the categorization output (the assignment `a`, plus the intent). It keeps in `z` every causal marker needed for clear, confident clustering decisions.
- **Compression term** `I(f_φ(x); x)`: forces the encoder to be as restrictive as possible, deleting bits that vary a lot or do not help clustering (changing backgrounds, styles).
- **`β`** controls the compression / utility trade-off.

### 4.4 Tractable training objective

As in the Deep VIB, both terms are replaced by variational bounds. With a stochastic encoder `p_φ(z|x)` and a fixed prior `r(z)`:

$$
I(z;x) \le \mathbb{E}_{x}\mathrm{KL}\big(p_\phi(z\mid x)\|r(z)\big),
\qquad
I(z;a,m) \ge H(a,m) + \mathbb{E}\big[\log q_\psi(m\mid z) + \log q_\theta(a\mid z,C,m)\big]
$$

where `q_θ(a | s) = softmax_a R_θ(s, a)` is the **choice model induced by the reward** (Section 5.2). It is used to score the experts' decisions; it is not the policy that acts. Two remarks on the second bound:

- **The targets are relational, not categorical.** Our experts say whether an image goes with a group that already exists, or whether two images go together, not which cluster is "the right" one ([Section 5](#5-where-do-the-experts-come-from)). We therefore instantiate the relevance term with the *expert likelihood*: the intent loss `L_ψ` and the reward loss `L_θ` of [Section 6](#6-components--training) play the roles of `−E log q_ψ(m|z)` and `−E log q_θ(a|z,C,m)`. This is a surrogate for the bound, not a strict bound.
- **The entropy `H(a,m)`** is what prevents collapse in the bound. In our setting it is supplied by the repulsion between random pairs ([Section 5.3](#53-two-forces-attraction-and-repulsion)) and by the finite number of slots.

The loss minimized on the encoder, intent predictor and reward is

$$
\mathcal{L} = \mathcal{L}_\psi + \mathcal{L}_\theta + \beta\mathbb{E}_x\mathrm{KL}\big(p_\phi(z\mid x)\|r(z)\big) + \lambda\mathcal{L}_{\text{inv}}
$$

and the policy is trained separately, with its own RL loss `L_ω` (Section 6.4), in alternation with it (Section 6.5).

### 4.5 Decision-consistency (invariance) term

A feature is spurious if removing it leaves the decision unchanged. We operationalize this with a consistency penalty under nuisance perturbations `t ~ 𝒯` (background swaps, color jitter, style transfer). Because a softmax policy is only defined up to an additive constant per image in the reward, and because the policy is what actually decides, we compare the **policies** on the clean and the perturbed view, in the same cluster state. With `z = f_φ(x)` and `z_t = f_φ(t(x))`:

$$
\mathcal{L}_{\text{inv}}(\phi) = \mathbb{E}_{x,t\sim\mathcal{T}}\mathrm{KL}\Big(\pi_{\bar\omega}(\cdot\mid z,\bar C,\bar m)\Big\|\pi_{\bar\omega}(\cdot\mid z_t,\bar C,\bar m)\Big)
$$

A bar means **stop-gradient**: the value is used but not updated by this term. The policy `ω`, the intent `ψ` (hence `m̄`, the intent of the clean view, given to both views) and the cluster memory `C` are *frozen* here, while the encoder `φ` is updated through both views. The term therefore tests the dependence of the decision on `z`; the invariance of `m` itself is enforced by `L_ψ`. Section 6.5 explains why. This is the bridge to IRM (invariance across environments) and IIC (agreement across views). It acts as a *check* on the encoder: if the useless features were really compressed away, the distribution over clusters must not move. Alternatives (the reward-induced choice model `q_θ̄` instead of `π_ω̄`, raw reward differences, `ω` unfrozen) are covered by the ablations in [Section 9](#9-ablations).

---

## 5. Where Do the Experts Come From?

### 5.1 The problem

IRL learns a reward by asking: *which reward would make the expert's behavior optimal?* Here the "behavior" is putting an image in the right cluster, and nobody has done that: no labels, no demonstrations. At the very start we have **no experts, no clusters, no intents**, and the cluster memory is empty.

### 5.2 The key idea: an image's transformations are an expert demonstration

Take one image and transform it: crop it, change its colors, replace its background, transform the result again. Whatever we get shows the *same thing*. Without placing a single label, we hold a group of images that the expert would put in the same cluster, and we can produce as many as we like because transformed images can be transformed again.

**Definitions.**

- **Nuisance family `𝒯`.** Transformations that change appearance (crop, color, blur, background, style) but are meant to preserve the object.
- **Orbit of an image `x_i`.** `𝒪_i = { t_D ∘ … ∘ t_1 (x_i) : t_k ∈ 𝒯, D ≤ D_max }`: everything reachable by chaining at most `D_max` transformations, with bounded cumulative strength (long chains drift away from the object).
- **Demonstration (episode).** An episode presents views drawn from several orbits, one after the other, in random order. It plays the role of the trajectory in PEMIRL; the state of each decision is `(z_t, m_t, C_t)`, with the cluster memory `C_t` as the process built it. The clusters the expert refers to are the ones in `C_t`: the expert needs no memory of its own.

**The expert's decisions.** The expert is *relational*: it never names a cluster in absolute terms. In the state `s_t = (z_t, m_t, C_t)` where view `u` arrives:

- **Forced move** (certain, by construction). If another view of the same orbit has already been placed, in cluster `k`, the expert action is `a = k`: *join the group of your twin*.
- **Free move.** If `u` is the first view of its orbit, nothing is demonstrated about *where* it goes. It is not supervised by attraction; it is only constrained by repulsion (below), and the policy chooses.
- **Repulsion** (probably right). A view `v` from a different orbit that already sits in cluster `k(v)` is, in all likelihood, a different thing: the expert says "do not join `k(v)` because of `v`". This is wrong with probability `τ⁺`, the chance that two random images share a concept (≈ `1/K` for balanced classes): two different cat photos are "different orbits" but the same concept.

This is exactly the structure of *must-link / cannot-link* constraints in constrained clustering, generated automatically. The reward is therefore identified up to a relabeling of the clusters, which is all clustering needs.

**How an episode becomes expert data, step by step.** In every episode:

1. Draw `B` images from the images dedicated to building the experts, and `n` views of each with `𝒯` (so `B·n` views, `n` views per orbit).
2. Shuffle the `B·n` views into one sequence. The episode is this sequence: the views arrive one at a time, and each placement updates the memory `C` (the transition).
3. At each step the policy places the arriving view, seeing only `(z_t, m_t, C_t)`: it has no access to orbit identity.
4. In parallel, the data pipeline, which generated the views and therefore knows which ones share an orbit, labels the step: forced move, free move, or repulsion pairs (previous list). These labels feed only the reward loss `L_θ`.

A small example with `K` slots and six views, in this order: `A1, B1, A2, C1, B2, C2` (`A`, `B`, `C` are three images, each with two views; `A` and `C` are two different cats, `B` is a dog):

| Step | Arriving view | Case | What the expert says |
|---|---|---|---|
| 1 | `A1` | Free move | Nothing yet to repel |
| 2 | `B1` | Free move | Repulsion: do not join the cluster of `A1` |
| 3 | `A2` | Forced move | Join the cluster where the policy put `A1` |
| 4 | `C1` | Free move | Repulsion against the clusters of `A1`, `B1`, `A2`. The repulsion from `A` is a false negative (two cats), absorbed by `τ⁺` |
| 5 | `B2` | Forced move | Join the cluster of `B1` |
| 6 | `C2` | Forced move | Join the cluster of `C1` |

Nothing in these signals says that `A` and `C` go together: that merge comes from the limited slots and from the similarity of the codes ([Section 5.4](#54-why-two-different-cat-photos-end-up-together)).

**Difference with classical IRL.** In classical IRL the demonstrations are fixed data `(s, a)`, produced by the expert in its own states, and the learner's policy rolls out in the environment on its own states without ever reading the expert's actions: only the reward (or the discriminator) compares the two. We keep the second half: the policy never sees orbit identity nor the labels, and only the reward loss uses them. We differ in the first half: the expert is an oracle and not a stored dataset. Since we generated the views, we know for any state the policy reaches what the expert would do (a forced move) or refuse (a repulsion). The demonstrations are therefore labeled on the states the policy visits, as in DAgger, and they depend on its placements: the twin's cluster is wherever the policy put it. The signal is "be consistent with your own placements of the same orbit, and tell different images apart", which is weaker than copying an external expert but needs no labeled cluster at all.

**The reward-induced choice model.** To score the expert's decisions we model the expert as choosing slot `a` with probability proportional to `exp R_θ(s, a)`:

$$
q_\theta(a\mid s) = \frac{\exp R_\theta(s,a)}{\sum_{a'=1}^{K}\exp R_\theta(s,a')}
$$

Learning the reward that makes the expert's decisions likely under `q_θ` is maximum-likelihood IRL with relational demonstrations. Two approximations are explicit: the expert is modelled as Boltzmann-rational on the *immediate* reward (no soft value iteration through the transitions), and the demonstrations are labeled on the policy's own states (previous paragraphs). Acting is left to the separate policy `π_ω` (Section 6.4).

```mermaid
flowchart LR
    X["Unlabeled images"] --> O["Orbit of each image<br/>(views via 𝒯)"]
    O --> F["Forced moves<br/>twin already in cluster k → action k"]
    X --> N["Random pairs<br/>different orbits → 'not together'<br/>(wrong ≈ 1/K of the time)"]
    F --> E["Expert decisions<br/>in the states visited by the policy"]
    N --> E
    E --> L["Expert likelihood under q_θ(a | s)<br/>= softmax of R_θ"]
```

### 5.3 Two forces: attraction and repulsion

Positives alone are not enough: putting everything in one cluster satisfies "an orbit stays together" perfectly. We need a force that says *different things should be told apart*, which is the "reward that distinguishes".

- **Attraction (forced moves).** When view `u` arrives and a twin is already in cluster `k⁺(u)` (the cluster of the first-placed view of its orbit), the reward must make joining that cluster likely:

$$
\mathcal{L}_{\text{pos}} = -\mathbb{E}_{u:\text{twin placed}}\big[\log q_\theta\big(k^+(u)\mid s_u\big)\big]
$$

- **Repulsion (random negatives).** For a view `u` and a random previously placed view `v` of another orbit, sitting in cluster `k(v)`, the reward must make joining `k(v)` unlikely. With `ℓ(u,v) = −log(1 − q_θ(k(v) | s_u))`, the naive negative term is `E_v ℓ(u,v)`. A fraction `τ⁺` of random `v` are actually of the same concept, so we subtract their share, approximated by the anchor's own twins (the standard debiasing of Chuang et al. 2020). We use `τ⁺ = 1/K` on Colored MNIST; on Waterbirds, where classes are imbalanced, `τ⁺` is swept as a hyperparameter (see assumption A2):

$$
\mathcal{L}_{\text{neg}} = \mathbb{E}_{u}\max\Big\{\tfrac{1}{1-\tau^+}\Big\}(\mathbb{E}_{v\sim\text{placed}}\ell(u,v)-\tau^+\mathbb{E}_{v^+\in\text{twins}(u)}\ell(u,v^+)\Big),0\Big\}
$$

The expert loss is `L_expert = L_pos + L_neg`, and it is what trains the reward (Section 6.3). The same repulsion is what makes an **empty slot attractive**: when an image fits no existing group, the mass leaves the occupied slots and goes to an empty one, i.e. the reward values *opening a new cluster*.

**Why random negatives are not an intervention.** We never choose pairs using knowledge of concepts. The only assumption is that random pairs are rarely of the same concept, with a *known* error rate `τ⁺`. Choosing negatives from images we know to be different would inject labels at the start and defeat the unsupervised goal; we do not do it.

### 5.4 Why two different cat photos end up together

At the start every image has its own orbit, so nothing says "these two cats belong together". Three things make them merge anyway:

1. **Limited slots.** There are at most `K` clusters. Once the slots are in use, a new orbit must join an existing group, and orbits that look alike must share one. While free slots remain, opening a new cluster is possible but is rewarded only when the image is genuinely incompatible with the groups that exist.
2. **Similarity in `z`.** Orbits with similar codes get similar rewards for the same group. Two cats share "cat features", so they fall in the same group.
3. **The Information Bottleneck decides what "similar" means.** If `z` still contains the background, merging happens along the background axis (cats on grass with dogs on grass). The compression is what makes the merging happen along the animal.

So the encoder is not a side detail: it determines *along which axis* the expert's weak signal is turned into categories.

### 5.5 Phase 2: merging progressively (deferred)

> **Deferred.** Phase 2 is set aside for now: the first experiments use Phase 1 only (orbits + random negatives). The design below is provisional and will be revisited.

Phase 1 (orbits + random negatives) is the foundation. Phase 2 speeds up merging of the same concept across different orbits by adding **extra positives** during training: pairs of views from *different* orbits that the model itself considers the same thing. Concretely, a linked pair of orbits `(i, j)` acts as one orbit in later episodes: a view of `j` arriving after a view of `i` is a forced move toward the cluster of `i`, and the pair is removed from the repulsion. The expert therefore moves, slowly and under the brakes below, and two cat orbits end up grouped explicitly instead of only through the limited slots ([Section 5.4](#54-why-two-different-cat-photos-end-up-together)).

Candidate pairs `(u, v)` are accepted if all of these hold:

- `u` and `v` are **mutual `k`-nearest neighbors** in the code space of a slowly-moving copy of the encoder (EMA);
- both are assigned to the same cluster with **high confidence** (`max_a π_ω ≥ γ_c`);
- the number of accepted pairs per epoch is **capped** (`≤ M`), with a weight that ramps up over training.

Accepted pairs are added to the positives and **removed from the negatives**, so the model is no longer told that they are different. This is the progressive merging: it is useful but can be unstable (the model confirms its own mistakes, and the expert moves while the reward is still learning). The brakes above, plus monitoring of the *churn* (fraction of images changing cluster between epochs), are how we keep it under control.

### 5.6 Declared assumptions

We prefer to state what the expert silently relies on:

| Assumption | Meaning | Consequence |
|---|---|---|
| **A1. `𝒯` preserves the object** | The family of transformations changes appearance, not identity. It encodes *prior knowledge about nuisances*, not about classes. | The method is unsupervised with respect to classes, but weakly informed about nuisances, like the environments of IRM. If `𝒯` never changes the background, the expert cannot teach invariance to it, and the burden falls on the Information Bottleneck. |
| **A2. Random pairs are rarely the same concept** | `τ⁺ ≈ 1/K` for balanced classes. | The true value is `τ⁺ = Σ_c p_c²`, the probability that two random images share a class (0.625 for two classes at 75% / 25%, instead of 0.5). We use `τ⁺ = 1/K` on Colored MNIST and sweep `τ⁺` on Waterbirds. |
| **A3. `K` is a known upper bound** | `K` is given and at least the true number of concepts. The model is free to use fewer slots. | The number of clusters actually constructed is monitored; sensitivity to a `K` larger or smaller than the truth is part of the ablations. |

### 5.7 Failure modes and what we monitor

| Failure | Symptom | Guard / monitor |
|---|---|---|
| Background shortcut | High `NMI(cluster, background)` | Nuisance-targeted `𝒯` (A1), compression `β`; **track leakage from step 0**, not only at the end |
| Low-level shortcut (color histogram, texture) | Clusters follow simple statistics | Diverse and strong `𝒯`, compression `β` |
| Collapse | Few clusters used (everything joins one slot) | Repulsion ([5.3](#53-two-forces-attraction-and-repulsion)); number of non-empty slots; entropy of the cluster-size histogram |
| Fragmentation | One concept split across clusters, too many slots opened | Upper bound `K`, Phase 2 merging, over-clustering analysis |
| Cold start of empty slots | Early on, all slots are empty and every placement looks equally good | Learned empty-slot embedding; warm-up of the reward on forced moves before the policy acts on its own; monitor the number of non-empty slots from step 0 |
| Order dependence | Different clusters for different image orders within an episode | Random order every episode; summaries updated slowly once mature; report variance across orders at test time |
| Non-stationary state and reward | The policy chases a reward and a memory that both move | Alternating updates with a slow reward relative to the policy; frozen reward during the policy step (Section 6.5) |
| RL instability | Return oscillates, policy entropy collapses | Entropy bonus `α`, small policy learning rate, critic target network, comparison with the closed-form `softmax(R_θ)` control |
| Semantic drift along chains | Views in an orbit no longer show the same object | Bound `D_max` and cumulative strength |
| Over-strong transformations | Crops that remove the object, noisy "twin" decisions | Bound the strength of each `t ∈ 𝒯` |
| Confirmation bias (Phase 2, deferred) | Churn does not decrease, errors reinforced | EMA copy, mutual neighbors, confidence `γ_c`, rate cap `M` |
| Moving expert | Reward loss oscillates | Change the expert slowly relative to reward learning |
| False negatives | About `1/K` of random pairs are the same concept | Debiasing; check for class imbalance |

### 5.8 Relation to existing work

Taken separately, the pieces are known: orbit positives and random negatives are contrastive learning (SimCLR), the debiasing is Chuang et al. 2020, the group-joining decision is close to online prototype / sequential clustering, the neighbor mining of Phase 2 is close to SCAN, and relational supervision is constrained clustering. **What we test** is whether casting them as expert demonstrations for a latent reward, with an RL policy that builds the clusters, an Information Bottleneck on the encoder and a decision-level invariance check, removes spurious features better than the same pipeline with a plain classification head (see [Ablations](#9-ablations)).

---

## 6. Components & Training

The architecture has four trained components: the **encoder**, the **intent predictor**, the **reward model** and the **policy** (a separate network with its critic, trained by RL), plus a non-differentiable **cluster memory** that the policy writes to. This section fixes the notation, then gives for each component its role, its loss, and the intuition behind each term. It closes with what joint and sequential training mean in practice, and exactly what is frozen where.

### 6.0 Notation

- For a view `u`: `z_u = f_φ(u)` and `m_u = μ_ψ(z_u)` (mean of the inferred intent).
- `C = {(n_k, c_k)}` is the cluster memory (Section 4.2): `c_k` is the average of the intents `m_v` of the members of cluster `k`. `s_u = (z_u, m_u, C)` is the state in which `u` arrives. `C` is always treated as a constant. States and summaries always use the **single-view** intent `m_u`, as at test time; the orbit mean `m*` below is only a training target for `q_ψ`.
- `R_θ(s, a)` is the reward of placing the arriving image in slot `a`. `q_θ(a | s) = softmax_a R_θ(s, a)` is the choice model it induces, used to score the experts (Section 5.2).
- `π_ω(a | s)` is the **policy**, a network of its own, trained by RL against `R_θ`. It is not `q_θ`.
- A bar (`θ̄`, `ω̄`, `m̄`, `C̄`) means **stop-gradient**: the value is used in the computation but not updated by that loss.
- Expert signals (Section 5): forced moves (twin already placed), random previously placed views (repulsion), with `τ⁺ = 1/K`.
- Intent target of an image `i`: `m*_i = sg[ mean over the orbit views of μ_ψ(z_v) ]`, where `sg[·]` is stop-gradient. The intent is a property of the orbit, not of a single view.

### 6.1 Encoder `f_φ`

**Role.** The only component that sees pixels. It maps an image to a stochastic code `z = μ_φ(x) + σ_φ(x) ⊙ ε`, `ε ~ 𝒩(0, I)`, and everything downstream sees only `z`. Its job is a trade-off: keep what the intent and the reward need, discard everything else, in particular backgrounds and styles.

**Loss.**

$$
\mathcal{L}_{\text{enc}}(\phi) = \underbrace{\mathcal{L}_\psi + \mathcal{L}_\theta}_{\text{utility, through } z} + \beta\underbrace{\mathbb{E}_x\mathrm{KL}\big(p_\phi(z\mid x)\|\mathcal{N}(0,I)\big)}_{\text{compression}} + \lambda\underbrace{\mathbb{E}_{x,t}\mathrm{KL}\big(\pi_{\bar\omega}(\cdot\mid z,\bar C,\bar m)\|\pi_{\bar\omega}(\cdot\mid z_t,\bar C,\bar m)\big)}_{\mathcal{L}_{\text{inv}}(\omega,\psi,C\text{ frozen})}
$$

For a diagonal Gaussian the KL has a closed form: `½ Σ_j (μ_j² + σ_j² − 1 − log σ_j²)`.

**Reading the terms.**
- **Utility `L_ψ + L_θ`.** The gradients of the intent and expert losses flow back into `z`. This is the "retain as much information as possible" pressure: anything needed to predict the concept and to tell images apart must survive.
- **Compression `β·KL`.** Pulls every code toward the prior, i.e. makes `z` carry as few bits about `x` as possible. `β` sets how hard it pushes.
- **Invariance `λ·L_inv`.** The check on the encoder. `L_pos` already asks that the views of an image join the same cluster; `L_inv` asks that the whole distribution over clusters chosen by the policy stay the same under perturbation. `ω`, `ψ` and `C` are frozen in this term (see Section 6.5); otherwise the policy could simply learn to ignore the image and satisfy it for free.

### 6.2 Intent predictor `q_ψ`

**Role.** Infers from `z` a continuous intent `m` (what the image is "about", e.g. species morphology), the counterpart of the context variable of Meta-IRL. It plays two roles. For an image, it is the nuisance-free summary of what the image expresses. For a cluster, the average of the intents of its members is the cluster's summary in the memory `C`, so the intent of a cluster is built from the intents of its images, as PEMIRL infers the context of a task from its demonstrations. The reward and the policy receive `m` and compare it with the intents of the existing clusters.

**Loss.**

$$
\mathcal{L}_\psi = -\mathbb{E}_{i,u\in\text{orbit}_i}\big[\log q_\psi(m^\*_i \mid z_u)\big]
$$

**Reading the terms.**
- **Log-likelihood of the orbit's intent `m*`.** Every view of an image must give the same concept. A concept that changes when only the background changes is penalized, so `m` is pushed toward content.
- **Gaussian case.** With `q_ψ(m | z) = 𝒩(μ_ψ(z), σ_ψ²(z))`, this is a squared error weighted by the predicted precision, plus `log σ_ψ`: the model can express uncertainty but pays for it.
- **Caveat.** `m` is a function of `z`: it adds an invariance signal, not information. A constant `m` would leave the memory with only counts and reduce the model to single-intention IRL. The "without `m`" ablation measures what `m` really contributes.

### 6.3 Reward model `R_θ`

**Role.** Scores every placement `(s, a)`: how well the arriving image, coded by `z`, fits the group in slot `a` **and** how well it is told apart from the other groups in the memory `C`. It is the object that IRL recovers, the *intention of distinction*, shared by the whole dataset. It is a permutation-equivariant network over the slots, `R_θ(z, m, C, a) = g_θ(z, m, c_a, n_a, pool_{k≠a}(c_k, n_k))`: it reads the chosen slot together with a pooled view of the others, so the identity of a slot never matters, only its content (the first part measures compatibility, the second distinctness). It is trained by IRL on the experts; it is **not** the policy.

**Loss.** Maximum-likelihood IRL of the expert's decisions (Section 5), under the choice model `q_θ`:

$$
\mathcal{L}_\theta = \mathcal{L}_{\text{pos}} + \mathcal{L}_{\text{neg}} - \rho\mathcal{H}\big(\mathbb{E}_sq_\theta(\cdot\mid s)\big), \qquad \rho = 0 \text{ by default}
$$

**Reading the terms.**
- **`L_pos` (attraction, forced moves).** Raises the reward of the twin's cluster for the arriving view.
- **`L_neg` (repulsion).** Lowers the reward of joining a cluster that holds a different orbit, with the false-negative correction `τ⁺`. This is the "reward that distinguishes": it exists because the softmax inside `q_θ` is *relative*, so lowering a cluster's reward for one image raises the others, empty slots included.
- **`ρ·H(marginal)` (off by default, `ρ = 0`).** Entropy of the average choice distribution over the batch, a safeguard against collapse. With repulsion in place it should be unnecessary. The activation rule is fixed in advance: we monitor the effective number of clusters `K_eff = exp(H(p̄))`, where `p̄` is the average assignment distribution; if `K_eff` falls below `0.7·K` (initial value, to be validated) during the first epochs, a small `ρ` is switched on. We avoid a large `ρ` by default because uniform cluster usage is wrong for imbalanced classes (e.g. Waterbirds), and because `K` is only an upper bound.

### 6.4 Policy `π_ω` (separate network, trained by RL)

**Role.** The agent that builds the clustering. At each step it sees the arriving code and the clusters built so far, and picks a slot: join a group, or open a new one. It is a separate network `π_ω(a | z, m, C)` (set-conditioned over the slots, like the reward), with a critic `V_ξ(s)` for the return estimates. At inference the assignment is `a = argmax_a π_ω(a | s)` with the memory `C` frozen at the end of training.

**Loss.** Entropy-regularized RL against the *frozen* reward (actor-critic, e.g. PPO with advantages from `V_ξ`):

$$
\mathcal{L}_\omega = -\mathbb{E}_{\pi_\omega}\Big[\sum_{t}\gamma^{t}R_{\bar\theta}(s_t,a_t)\Big] - \alpha\mathbb{E}\big[\mathcal{H}\big(\pi_\omega(\cdot\mid s_t)\big)\big]
$$

**Reading the terms.**
- **Return `Σ γ^t R_θ̄`.** The policy is judged on the *whole sequence*: each placement changes the memory `C`, hence the rewards that follow. This is what makes it a policy and not a softmax of the reward (Section 4.2).
- **Entropy bonus `α·H`.** Keeps exploration alive, in particular the exploration of empty slots in the early phase.
- **`θ̄` frozen.** During the policy step the reward does not move; it is the ruler the policy is optimized against. The encoder is frozen too: `z` and `C` are given, so RL never reaches the pixels.
- **`γ`.** `γ = 0` makes the policy essentially greedy on `R_θ`; the control in Section 9 compares it with `γ > 0`.

### 6.5 Joint training (alternating reward and policy steps)

The encoder, the intent predictor and the reward are trained jointly, in alternation with the policy. One iteration:

1. **Rollout.** Draw an episode: `B` images, `n` views each via `𝒯`, shuffled into one sequence ([Section 5.2](#52-the-key-idea-an-images-transformations-are-an-expert-demonstration)). The current policy `π_ω` places the views one by one, seeing only `(z_t, m_t, C_t)`; the memory `C` is updated by each transition. In parallel, the pipeline labels the steps from orbit membership (forced moves, repulsion pairs). These labels are used only in the reward step.
2. **Reward step (IRL).** Minimize the total loss on `φ, ψ, θ`:

$$
\mathcal{L} = \mathcal{L}_\psi + \mathcal{L}_\theta + \beta\mathrm{KL} + \lambda\mathcal{L}_{\text{inv}}
$$

3. **Policy step (RL).** With `φ, ψ, θ` frozen, minimize `L_ω` (and the critic loss) on `ω, ξ` on the same or a fresh rollout.

This is the objective of Section 4.4, plus the RL loss of Section 6.4, which is why the training is alternating and not a single loss. Gradient routing (what each term updates, and what it treats as constant):

| Term | Updates | Frozen (stop-gradient) |
|---|---|---|
| `L_ψ` (intent) | `ψ`, and `φ` through `z` | The orbit target `m*` |
| `L_θ` (expert likelihood) | `θ`, and `φ` through `z` | The memory `C` and the intent `m`: `ψ` gets no gradient from `L_θ`. The intent is trained only by `L_ψ`, so the reward cannot bend it to make its own job easier, and the cluster summaries stay stable |
| KL (compression) | `φ` | None |
| `L_inv` (invariance) | `φ` only, through both `z` and `z_t` | `ω`, `ψ` (hence `m`) and `C` |
| `L_ω` (RL return + entropy) | `ω` (and the critic `ξ`) | `φ`, `ψ`, `θ`, `C`: the policy step treats the encoder and the reward as given |

**What is frozen in `L_inv`, and why.** With `ω̄`, `m̄`, `C̄` frozen, the invariance term has gradient `∇_ω L_inv = 0` and `∇_ψ L_inv = 0`, while `∇_φ L_inv ≠ 0`. This means the policy is used as a *fixed measuring stick*: if the decision it takes changes under a perturbation, the *encoder* is what gets corrected. If `ω` were free, the term would have a trivial minimizer: any policy that does not depend on the image,

$$
\pi_\omega(\cdot\mid z,C,m) = \pi_\omega(\cdot\mid C,m) \Longrightarrow \pi_\omega(\cdot\mid z,C,m)=\pi_\omega(\cdot\mid z_t,C,m) \Longrightarrow \mathcal{L}_{\text{inv}} = 0 ,
$$

so the check would push the policy toward ignoring the image instead of pushing the encoder toward dropping the nuisance. The same holds for `m`: a free `ψ` could make `m` insensitive and cheat the term. Freezing is local to `L_inv`: `ω` keeps being trained by RL through `L_ω`, and `θ` by the experts through `L_θ`, so the ruler still evolves, but never through the check. The ablations "`ω` unfrozen" and "`q_θ̄` instead of `π_ω̄`" ([Section 9](#9-ablations)) measure how much these choices matter.

**What it means.** "Useless" is defined *relative to the reward and policy currently being learned*: the encoder compresses what they do not use, while they adapt to whatever the encoder keeps. This co-adaptation is the point of the method, and also its main risk: the early reward is random, so it can push the bottleneck to discard useful bits, and the early policy is random, so the memory `C` is noisy. Mitigations: a warm-up of `β` from 0, a reward warm-up on forced moves before the policy acts alone, and a reward updated more slowly than the policy.

### 6.6 Sequential training

Two stages, nothing flowing back from the second to the first.

1. **Encoder alone.** No reward exists yet. The encoder is trained on the *same expert signals* (Section 5) but with a reward-free similarity head (e.g. a cosine / contrastive head on `z`), together with `β·KL` and an invariance penalty on the code itself, `‖z − z_t‖²`.
2. **IRL and RL modules on frozen codes.** `φ` is frozen. `L_ψ + L_θ` are minimized on `z` and `L_ω` trains the policy, still in alternation (reward step, then policy step). There is no KL, no `L_inv`, and no gradient into the encoder.

| | Joint | Sequential |
|---|---|---|
| Optimization | One loop, all networks, alternating reward / policy steps | Encoder first, then intent / reward / policy |
| Does the encoder see the reward? | Yes (through `L_ψ`, `L_θ`, `L_inv`) | No, only a similarity head |
| Meaning of "useless" | Irrelevant to the current reward and policy | Irrelevant to the similarity objective |
| Strengths | Matches the definition of the method; the encoder is shaped by what clustering needs | Simple, stable, modular; the IRL / RL stage can be re-run without retraining the encoder |
| Risks | Moving targets, instability, need for warm-up | Frozen stage-1 errors; the encoder may drop cluster-relevant bits or keep spurious ones |

Sequential training is the natural control for the claim that *joint* optimization matters (see [Ablations](#9-ablations)); with `β = 0` it reduces to the "standard multi-intention IRL" baseline.

### 6.7 All the objectives in one place

**Representation and state.** For an arriving view `u`:

$$
z_u = \mu_\phi(u)+\sigma_\phi(u)\odot\varepsilon,\qquad m_u = \mu_\psi(z_u),\qquad s_u=(z_u,m_u,C),\qquad C=\{(n_k,c_k)\}_{k=1}^{K}
$$

**Transition (cluster memory).** After the assignment `a`, the chosen slot absorbs the intent of the view (an empty slot, `n_a = 0`, thereby becomes a new cluster):

$$
n_a \leftarrow n_a+1,\qquad c_a \leftarrow c_a+\frac{m_u-c_a}{n_a}
$$

**Intent (Section 6.2).** Every view of an orbit must give the intent of the orbit:

$$
\mathcal{L}_\psi=-\mathbb{E}_{i,u\in\text{orbit}_i}\big[\log q_\psi(m^*_i\mid z_u)\big],\qquad m^*_i=\mathrm{sg}\Big[\tfrac{1}{|\text{orbit}_i|}\textstyle\sum_{v\in\text{orbit}_i}\mu_\psi(z_v)\Big]
$$

**Reward (IRL on the experts, Sections 5.2, 5.3 and 6.3).** The choice model induced by the reward and the expert loss:

$$
q_\theta(a\mid s)=\frac{\exp R_\theta(s,a)}{\sum_{a'}\exp R_\theta(s,a')},\qquad
\mathcal{L}_{\text{pos}}=-\mathbb{E}\big[\log q_\theta(k^+(u)\mid s_u)\big]
$$

$$
\mathcal{L}_{\text{neg}}=\mathbb{E}_{u}\max\Big\{\tfrac{1}{1-\tau^+}\Big(\mathbb{E}_{v}\ell(u,v)-\tau^+\mathbb{E}_{v^+}\ell(u,v^+)\Big),0\Big\},\quad \ell(u,v)=-\log\big(1-q_\theta(k(v)\mid s_u)\big)
$$

$$
\mathcal{L}_\theta=\mathcal{L}_{\text{pos}}+\mathcal{L}_{\text{neg}}-\rho\mathcal{H}\big(\mathbb{E}_sq_\theta(\cdot\mid s)\big),\qquad \rho=0\text{ by default}
$$

**Invariance (Section 4.5).** The policy is the fixed ruler; only the encoder is updated:

$$
\mathcal{L}_{\text{inv}}=\mathbb{E}_{x,t\sim\mathcal{T}}\mathrm{KL}\Big(\pi_{\bar\omega}(\cdot\mid z,\bar m,\bar C)\Big\|\pi_{\bar\omega}(\cdot\mid z_t,\bar m,\bar C)\Big)
$$

**Reward step** (updates `φ, ψ, θ`):

$$
\mathcal{L}=\mathcal{L}_\psi+\mathcal{L}_\theta+\beta\mathbb{E}_x\mathrm{KL}\big(p_\phi(z\mid x)\|\mathcal{N}(0,I)\big)+\lambda\mathcal{L}_{\text{inv}}
$$

**Policy step** (updates `ω` and the critic `ξ`; `φ, ψ, θ, C` frozen). The policy maximizes the return of the learned reward, and the critic regresses it:

$$
\mathcal{L}_\omega=-\mathbb{E}_{\pi_\omega}\Big[\sum_t\gamma^t R_{\bar\theta}(s_t,a_t)\Big]-\alpha\mathbb{E}\big[\mathcal{H}(\pi_\omega(\cdot\mid s_t))\big],\qquad
\mathcal{L}_\xi=\mathbb{E}\big[(V_\xi(s_t)-\hat G_t)^2\big]
$$

with `Ĝ_t = Σ_{j≥0} γ^j R_θ̄(s_{t+j}, a_{t+j})`.

**Inference.** `C` is frozen at the end of training and each image is assigned by `a = argmax_a π_ω(a | z, m, C)` with `m = μ_ψ(z)`.

---

## 7. Decisions & Open Questions

### 7.1 Decisions taken

Each decision is given with its reason and with what we rejected.

| # | Decision | Why | Rejected alternative |
|---|---|---|---|
| 1 | **Expert = transformations.** The views of an orbit show the same thing, so they give certain "join your twin" decisions (forced moves). | IRL needs demonstrations and an unsupervised problem has none; the images themselves can provide the "same" signal for free. | Experts built from labels (not unsupervised). |
| 2 | **Relational expert.** The expert never names a cluster in absolute terms: forced moves (attraction), free moves (first view of an orbit), repulsion between random views. | Must-link / cannot-link constraints are enough to identify the reward up to a relabeling of clusters, which is all clustering needs. | Absolute cluster labels as expert actions. |
| 3 | **Random negatives with a known false-negative rate `τ⁺`, debiased.** | Random pairs are rarely the same concept; the error rate is known and corrected (Chuang et al. 2020). It is not an intervention. | Negatives chosen from concepts known to be different: they inject labels at the start and bias the unsupervised goal. |
| 4 | **`𝒯` is prior knowledge about nuisances** (assumption A1). | It encodes what should *not* matter, not what the classes are, like the environments of IRM. | Pretending the method needs no prior at all. |
| 5 | **A separate policy `π_ω`, trained by RL, conditioned on the clusters built so far.** Actions are the existing clusters or an empty slot; each placement updates the memory `C`. | IRL must bring something new. The reward is the *intention of distinction* (learned from the experts) and the policy optimizes the *return* over a sequence of placements that change the state. A closed-form `softmax(R)` would only be a per-image classifier. | The policy as the softmax of the reward (kept as a control, `γ = 0`); no cluster memory (kept as an ablation). |
| 6 | **`K` is an upper bound; clusters are opened progressively.** Slots start empty, choosing an empty slot opens a cluster, and the number of constructed clusters grows during training. | Consistent with the cluster memory (a cluster exists once it has members) and with the "unsupervised, built along the way" intuition. `K` known as an upper bound is a weaker assumption than `K` exact. | A fixed `K` all active from step 0. |
| 7 | **Alternating training.** Reward step (IRL: encoder, intent, reward) then policy step (RL, with the others frozen). | The policy is optimized against a reward, so the reward must be a ruler that does not move during the policy step; both co-adapt across iterations. | A single end-to-end loss through the RL return. |
| 8 | **`m` is the intent, and the cluster memory is made of intents.** `m` is inferred per image by `q_ψ` (made consistent over an orbit by `L_ψ`); the summary of a cluster is the average intent of its members; the reward and the policy receive `z`, `m` and `C`. `ψ` is trained only by `L_ψ`. | It is the direct counterpart of Meta-IRL: the context of a task is inferred from the set of its demonstrations. It avoids a redundant per-image latent next to the memory. | `m` as the cluster index (duplicates the policy output); `m` absent (kept as an ablation); `ψ` trained by `L_θ` (the reward could distort the intent). |
| 9 | **Invariance compares policies, with `ω`, `ψ` and `C` frozen.** `L_inv` is a KL between the policies on the clean and perturbed views, in the same cluster state. | A reward is defined only up to a per-image constant under a softmax, so policies are the comparable objects. A free policy could ignore the image and give `L_inv = 0` for free. | Raw reward differences; `ω` unfrozen (kept as ablations). |
| 10 | **No collapse safeguard by default.** `ρ = 0`; switched on only if `K_eff < 0.7·K` early in training. | Repulsion and the finite number of slots already prevent collapse; a uniform-usage bias is wrong for imbalanced classes. | A large `ρ` from the start. |
| 11 | **`τ⁺` and `K`.** `τ⁺ = 1/K` on Colored MNIST, swept on Waterbirds; `K` equals the true number of classes in the main results, with sensitivity in the ablations. | `1/K` is exact for balanced classes; Waterbirds is imbalanced (`τ⁺ = Σ_c p_c²`). | A single universal `τ⁺`. |

### 7.2 Still open

No design question is open at this stage. Points to settle by experiment:

1. Whether the reward and the policy should see `z` and `m`, or only `m` (ablation, [Section 9](#9-ablations)).
2. The update rate of the memory (running mean vs. slow EMA) and the RL hyperparameters (`γ`, `α`, ratio between reward and policy steps).
3. Whether the reward needs a warm-up on forced moves before the policy acts alone (cold start).
4. Whether episodes are drawn from the whole training set or from a dedicated portion of it, and the number of orbits `B` and views `n` per episode.

### 7.3 Deferred

- **Phase 2 (progressive merging).** When to start merging and how fast (`k`, `γ_c`, `M`, EMA momentum), given the risk of confirmation bias. Set aside; first experiments use Phase 1 only.

---

## 8. Evaluation Protocol

### 8.1 Datasets (engineered spurious correlations)

| Dataset | Setup | What it tests |
|---|---|---|
| **Colored MNIST** | Digits are strongly correlated with background/foreground color at training time; correlation is randomized or inverted at test time. 10-class variant for clustering; the binary IRM variant as a sanity check. | Whether the model clusters by digit identity instead of color |
| **Waterbirds** | Landbirds on land and waterbirds on water (strong train-time correlation); minority groups (landbird on water, waterbird on land) at test time. | Whether the model clusters by bird category instead of land/water background |

### 8.2 Baselines

1. **Geometric / statistical clustering:** K-Means and GMM on standard ResNet / ViT embeddings.
2. **Pure contrastive learning:** SimCLR, MoCo v3 (and IIC) followed by clustering, to see whether contrastive augmentations alone discard backgrounds.
3. **Standard multi-intention IRL:** a temporal IRL model adapted to single-step images **without** the causal Information Bottleneck encoder.

### 8.3 Metrics

- **Clustering Accuracy (ACC)** with optimal (Hungarian) cluster-to-label matching, and **Normalized Mutual Information (NMI)**, both measured strictly against the true semantic labels (bird species, digit ID), ignoring background.
- **Out-of-distribution (OOD) robustness:** drop in performance (`ΔACC`) between the correlated test set and the test set where the background correlation is inverted.
- **Worst-group ACC** (Waterbirds): accuracy on the least-represented (class, background) group.
- **Spurious leakage:** `NMI(cluster, background)`. Lower is better; a model that clusters by background scores high.

Training-time diagnostics (Section 5.7): leakage from step 0, number of non-empty slots, cluster-size entropy, churn between epochs, return and policy entropy, sensitivity to the order of images, and `NMI(m, background)`.

Reporting: mean ± std over several seeds, with `K` (upper bound) set to the number of true classes and the number of constructed clusters reported alongside.

---

## 9. Ablations

| Ablation | Question answered |
|---|---|
| **Plain classification head** instead of reward + policy + `m` (same experts, same encoder) | Does the IRL / RL formalism add anything over contrastive / IIC-style clustering? |
| **Closed-form policy `softmax(R_θ)`** (or `γ = 0`) instead of the RL-trained `π_ω` | Does the separate policy and the sequential return bring anything beyond a per-image reward? |
| **Reward per image, no cluster memory** (`R_θ(z, a)` only) | Does conditioning on the clusters built so far matter? |
| Persistent memory vs memory reset every episode; random vs fixed image order | How much does the result depend on the way the clusters are built along the sequence? |
| `β = 0` (no Information Bottleneck) | Does compression alone remove the spurious features? |
| `λ = 0` (no decision-consistency term) | Is the invariance term necessary, or is the IB sufficient? |
| `q_θ̄` instead of `π_ω̄` in `L_inv`; raw reward differences; `ω` and `ψ` unfrozen in `L_inv` | Does the form of the check or the freezing matter (does the policy drift toward ignoring the image)? |
| `ρ > 0` from the start | Does an explicit anti-collapse term help, or hurt on imbalanced classes? |
| No negatives (positives only) | Does collapse appear, as predicted? |
| No debiasing (`τ⁺ = 0`) | How much do false negatives hurt? |
| `𝒯` families (with / without background-targeted transformations) and `D_max` sweep | How much of the result comes from prior knowledge about nuisances (A1)? |
| Phase 2 off / on (deferred), and brakes removed one by one (EMA, mutual neighbors, `γ_c`, cap `M`) | Is progressive merging useful, and which brake matters? |
| Sequential training (encoder → IRL / RL) instead of joint; reward-to-policy step ratio | Does joint optimization matter, and how sensitive is the alternation? |
| `β` sweep | Where is the compression / utility trade-off, and is there a sweet spot? |
| `K` larger than the true number, smaller than the true number, imbalanced classes, and `τ⁺` sweep | How sensitive is the method to the upper bound `K` and to the value of `τ⁺`, and does it open the right number of clusters? |
| Without latent context `m` | What does the intent variable contribute? |
| Cluster summaries in `z`-space instead of `m`-space; reward and policy see only `m` (no `z`) | Where should clusters be compared, and does `z` bring more than `m` to the decisions? |

---

## 10. Repository Structure (planned)

```
cib-irl/
├── README.md
├── configs/          # experiment configs (datasets, β, λ, K, γ, α, 𝒯, Phase 2 brakes)
├── cibirl/
│   ├── encoder.py    # causal encoder f_φ (stochastic, VIB head)
│   ├── intent.py     # intent predictor q_ψ(m | z)
│   ├── memory.py     # cluster memory C: slot summaries, empty slots, transitions
│   ├── reward.py     # reward R_θ(z, m, C, a) and induced choice model q_θ
│   ├── policy.py     # RL policy π_ω(a | z, m, C) and critic V_ξ
│   ├── experts.py    # orbits, forced moves, random pairs, debiasing (Phase 2 mining later)
│   ├── losses.py     # variational IB bound, expert likelihood, RL loss, consistency term
│   └── train.py      # rollouts, alternating reward / policy loop, Phase 1 / Phase 2 schedule
├── data/             # Colored MNIST, Waterbirds loaders
├── baselines/        # K-Means/GMM, SimCLR, MoCo v3, IIC, temporal IRL
├── eval/             # ACC, NMI, OOD drop, worst-group, leakage, churn, order sensitivity
└── scripts/          # launchers (local and SLURM)
```

---

## 11. Status & Roadmap

- [x] Problem statement and literature review
- [x] Architecture and joint objective
- [x] Expert design (orbits, relational decisions, forced moves, debiased negatives, Phase 2)
- [x] Policy design: separate RL policy conditioned on the clusters built so far, `K` as an upper bound, alternating training
- [x] Design questions settled: experts, policy, `m` ([Section 7.1](#71-decisions-taken))
- [ ] Variational formulation validated (Section 4.4) and consistency term fixed (Section 4.5)
- [ ] Expert module, cluster memory and monitoring (leakage, churn, non-empty slots, cluster-size entropy)
- [ ] Stable reward / policy alternation on Colored MNIST (RL stability, order dependence)
- [ ] Colored MNIST pipeline and baselines
- [ ] Waterbirds pipeline and baselines
- [ ] Full ablation suite (Phase 2 deferred)
- [ ] Write-up

---

## 12. License

MIT License
