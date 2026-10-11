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
- **The expert.** We can see the creator of the data (the world that produced the cats and the dogs) as an expert who, in building these images, optimized some reward. Multi-intention IRL methods such as PEMIRL recover from expert behavior the reward attached to each category. Our expert is imperfect on purpose and **never constrains the learner**: it only guides the reward toward what it means to tell things apart.
- **The intent.** As in Meta-IRL, a group of images has an *intent*: what its members have in common (the context variable `m`). The memory of the groups built so far is made of these intents, so the reward and the policy compare the intent of the arriving image with the intent of each existing group.
- **Our step beyond.** Besides the reward that makes each category what it is, there also exists a reward that **distinguishes categories from one another**, the reason we do not confuse them. Finding it *is* learning to distinguish. Optimizing it by RL, over decisions that depend on the clusters already formed, is clustering.

### 1.3 The obstacle, and our answer

IRL needs expert demonstrations, but an unsupervised problem has none (a supervised version would not be interesting). Our answer is to **build the experts from the images themselves**:

- all the transformed versions of an image (crops, color changes, new backgrounds, transformations of transformations) obviously show the same thing: when one of them arrives and another is already in a group, the expert demonstrates "join that group";
- two random images usually show different things: the expert demonstrates "do not put these together".

The expert never says *which* cluster in absolute terms; it only answers "same group or not?". That is enough to learn a reward that tells clusters apart. These decisions form expert demonstrations: episodes in which the views of a few images arrive one after the other and a rule-based expert places them in its own groups. The data pipeline generated the views, so it knows which ones share an orbit. The expert does not constrain the learner's policy, which builds its own clusters; only the reward learns from the demonstrations. Details in [Section 5](#5-where-do-the-experts-come-from).

### 1.4 Two safeguards against shortcuts

Image datasets are full of spurious cues (background, texture, style) that are easier to use than the object. Two ingredients keep the reward on the object:

- **Information Bottleneck.** An encoder is trained to keep as much information as the reward needs and to compress away the rest, so a feature that does not help the decision is deleted.
- **Invariance check.** If the encoder really removed the useless features, the decision must not change when the image is perturbed in ways that keep the object (e.g. new background). We add this as an explicit term.

### 1.5 At a glance

| Human intuition | In CIB-IRL |
|---|---|
| The few features that make a cat a cat | Causal latent `z` (encoder + Information Bottleneck) |
| "This looks like a cat-type thing" | Latent intent `m` (intent predictor `q_ψ`: one per orbit during training, one per image at test time) |
| The groups I have formed so far | The learner's cluster memory `C`: for each cluster, the list of its members and their average intent (empty slots included); it persists across episodes |
| Why this image belongs here and not there | Reward `R_θ(z, m, C, a)` |
| Putting it in a group (or starting a new one) | Policy `π_ω(a \| z, m, C)`, `a ∈ {1..K}`, trained by RL |
| Someone who already knows what goes together | A rule-based expert built from transformations, with its own per-episode memory; it guides the reward and never constrains the learner ([Section 5](#5-where-do-the-experts-come-from)) |
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
3. **A single reward.** Most of the IRL based methods don't actually use the IRL itself for the unsupervised clustering part. But rather they end up with a bunch of rewards, each representing a different cluster's features. Our approach tries to encode the concept of distinguishing in a single reward, and to use it for the clustering itself: the learned reward is what the policy optimizes to build the clusters. To our knowledge, this use of IRL for the clustering step is the main novelty of the project; whether it pays off is what the experiments test.

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
    C["Learner's cluster memory C<br/>(average intent of the members of each cluster<br/>built so far, empty slots included)"] --> pol
    m --> pol
    pol --> a["Assignment a ∈ {1..K}<br/>existing cluster, or empty slot = new cluster"]
    a -- "transition: update the summary of cluster a" --> C
    z --> rew["Reward R_θ(z, m, C, a)<br/>fit to the experts (IRL)"]
    C --> rew
    m --> rew
    m -- "averaged into the summary of the chosen cluster" --> C
    exp["Expert episodes<br/>(orbits, random pairs,<br/>expert's own per-episode memory)"] -. "IRL: fit the reward" .-> rew
    rew -. "RL: maximize cumulative return" .-> pol
    enc -. "compression: I(z; x)" .-> ib["Information Bottleneck"]
    ib -. "β" .-> enc
```

### 4.2 Episodic MDP over a growing set of clusters

Two kinds of episodes share the same state, transition and reward:

- **Learner episodes.** The learner's policy `π_ω` sweeps a minibatch of training images, one view per image, presented one after the other in random order. The learner's cluster memory `C` **persists across episodes and across training**: it is the set of clusters built so far. These episodes carry no label and no expert.
- **Expert episodes.** A dataset of demonstrations built before training from a portion of the images ([Section 5.2](#52-the-key-idea-an-images-transformations-are-an-expert-demonstration)): the views of a few images, shuffled, placed by a rule-based expert in the expert's own memory `C^E`. This memory starts empty at the beginning of each episode and is discarded at its end. Only the reward learns from these episodes.

The expert never parametrizes the learner: `π_ω` rolls out on its own states and is free to build memories and clusters that differ from the expert's.

| Component | Definition |
|---|---|
| State `s_t` | `(z_t, m_t, C_t)`. `z_t = f_φ(x_t)` is the code of the image arriving at step `t` and `m_t = μ_ψ(z_t)` is its inferred intent. `C_t = {(n_k, c_k)}_{k=1..K}` is the **cluster memory**: for each slot `k`, the number of members `n_k` and the **average intent** `c_k` of its members. An empty slot (`n_k = 0`) carries a learned "empty" embedding. `C` is a memory, not a differentiable function: it is updated by the transition and is stop-gradient everywhere. In an expert episode the state is the same, with the expert's memory `C^E_t` in place of `C_t` |
| Action `a_t` | A slot `a ∈ {1, …, K}`. Choosing a non-empty slot means *joining* that cluster. Choosing an empty slot means *opening a new cluster*. `K` is an **upper bound**; the number of clusters actually constructed starts at 0 and grows during training |
| Transition | The chosen slot is updated: `n_a ← n_a + 1`, `c_a ← c_a + (m_t − c_a)/n_a` (running mean of the member intents). The next image then sees a different state |
| Reward | `R_θ(s_t, a_t)`: compatibility of `z_t` with the chosen group **and** distinctness from the other groups. One network, permutation-equivariant over slots, reading the summaries `c_k` but not the counts `n_k`. Learned by IRL from the experts ([Section 5](#5-where-do-the-experts-come-from)) |
| Policy | `π_ω(a \| s_t)`, a **separate network** trained by RL to maximize the entropy-regularized cumulative return `E[Σ_t γ^t R_θ(s_t, a_t)] + α·H(π_ω)`. It is *not* `softmax(R_θ)` |
| Expert | Not observed: a rule-based expert built from image transformations, acting in its own per-episode memory ([Section 5](#5-where-do-the-experts-come-from)) |

**Keeping the learner's memory consistent.** The memory keeps, for each slot, the list of its members (a map image → slot, so an image belongs to at most one slot at a time). When an image is visited again, its previous placement is withdrawn (`n` and `c` are updated) before the new placement. Because the encoder keeps changing, a summary computed with an old encoder would no longer match the intents of new images: at the end of each epoch the summaries `c_k` are **recomputed from the members with the current encoder**. The reward does not receive the counts `n_k`: in an expert episode a slot holds at most `n` views, in a learner episode it can hold thousands, so a count would be out of distribution. The counts are still used by the transition and to know whether a slot is empty.

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

A feature is spurious if removing it leaves the decision unchanged. We operationalize this with a consistency penalty under nuisance perturbations `t ~ 𝒯` (background swaps, color jitter, style transfer). Because a softmax policy is only defined up to an additive constant per image in the reward, and because the policy is what actually decides, we compare the **policies** on the clean and the perturbed view, in the same (learner's) cluster state. With `z = f_φ(x)` and `z_t = f_φ(t(x))`:

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
- **Expert portion `𝒟_E`.** The portion of the training images dedicated to building the expert demonstrations; its size is a hyperparameter. The learner's environment is the whole training set.
- **Demonstration (episode).** An episode presents views drawn from several orbits, one after the other, in random order. It plays the role of the trajectory in PEMIRL. The state of each decision is `(z_t, m_t, C^E_t)`, where `C^E_t` is the **expert's own memory**: the groups the expert has formed so far in this episode, built by the same transition as the learner's memory ([Section 4.2](#42-episodic-mdp-over-a-growing-set-of-clusters)). It starts empty and is discarded at the end of the episode.

**The expert's rule.** The expert is *relational*: it never names a cluster in absolute terms. When view `u` arrives in the state `s^E_t = (z_t, m_t, C^E_t)`:

- **Forced move** (certain, by construction). If another view of the same orbit has already been placed, in slot `k` of the expert's memory, the expert action is `a = k`: *join the group of your twin*.
- **Free move.** If `u` is the first view of its orbit, the expert opens a new slot for it, so that the next views of the orbit have a group to join. In Phase 1, two different orbits are two different groups, *even when they show the same concept* (two cat photos). This placement builds the expert's state, but nothing is demonstrated about where the view "should" go: it is not supervised by attraction, it is only constrained by repulsion (below).
- **Repulsion** (probably right). A view `v` from a different orbit that already sits in slot `k(v)` of the expert's memory is, in all likelihood, a different thing: the expert says "do not join `k(v)` because of `v`". This is wrong with probability `τ⁺`, the chance that two random images share a concept (≈ `1/K` for balanced classes): two different cat photos are "different orbits" but the same concept.

This is exactly the structure of *must-link / cannot-link* constraints in constrained clustering, generated automatically. The reward is therefore identified up to a relabeling of the clusters, which is all clustering needs.

**The expert does not parametrize the learner.** The expert is not optimal: in Phase 1 it separates two cats. It is never used to restrict the learner. Its only job is to train the reward toward what it means to tell things apart. The learner's policy `π_ω` never sees orbit identity, the expert's slots or the expert's actions; it can build memories and clusters different from the expert's (and, with a limited number of slots, it typically has to merge what the expert separates).

**How an episode becomes expert data, step by step.** For each expert episode:

1. Draw `B` images from `𝒟_E` (with `B ≤ K`, so that the expert's memory fits in the `K` slots of the learner), and `n` views of each with `𝒯` (so `B·n` views, `n` views per orbit).
2. Shuffle the `B·n` views into one sequence. The episode is this sequence: the views arrive one at a time, and each placement updates the expert's memory (the transition).
3. The expert places each view following its rule, in its own memory `C^E`, which starts empty. This gives, at each step, a state `(z_t, m_t, C^E_t)` and an expert action.
4. The pipeline, which generated the views and therefore knows which ones share an orbit, labels each step: forced move, free move, or repulsion pairs. These labels feed only the reward loss `L_θ`.

**What is stored and what is recomputed.** The set of expert episodes is built once, before training, and reused, as the demonstrations of classical IRL. It stores the **structure** only: the images, the transformation recipes, the order of the views and the expert's slots. The codes `z_t` and intents `m_t` are recomputed with the current encoder at every training step, and `C^E_t` is rebuilt by the transition from the expert's slots. The expert's memory never survives its episode.

A small example with six views, in this order: `A1, B1, A2, C1, B2, C2` (`A`, `B`, `C` are three images, each with two views; `A` and `C` are two different cats, `B` is a dog):

| Step | Arriving view | Case | What the expert does and says |
|---|---|---|---|
| 1 | `A1` | Free move | Opens slot 1 for `A`. Nothing yet to repel |
| 2 | `B1` | Free move | Opens slot 2 for `B`. Repulsion: do not join slot 1 (`A1`) |
| 3 | `A2` | Forced move | Joins slot 1, the slot of `A1` |
| 4 | `C1` | Free move | Opens slot 3 for `C`. Repulsion against slots 1 and 2. The repulsion from `A` is a false negative (two cats), absorbed by `τ⁺` |
| 5 | `B2` | Forced move | Joins slot 2, the slot of `B1` |
| 6 | `C2` | Forced move | Joins slot 3, the slot of `C1` |

In the expert's memory the two cats `A` and `C` sit in two different slots (the Phase 1 tolerance). Nothing in these signals says that they go together: if they end up in one cluster, that is the learner's doing ([Section 5.4](#54-why-two-different-cat-photos-end-up-together)).

**The same structure as classical IRL.** The demonstrations are data produced by the expert in the expert's own states and fixed before training. The learner's policy `π_ω` interacts with its environment (all the training images) on its own states and never reads the expert's actions; only the reward compares what the expert does with what the learner does. Two things are specific to us: the expert is a rule (orbit membership) and not an agent, and it is deliberately imperfect. Since the pipeline knows the orbits, it could also label the placements that the learner makes in its own episodes; we do not do this in the main method, because the signal would become "be consistent with your own placements" instead of "imitate an independent expert". It is kept as an ablation ([Section 9](#9-ablations)).

**The reward-induced choice model.** To score the expert's decisions we model the expert as choosing slot `a` with probability proportional to `exp R_θ(s, a)`:

$$
q_\theta(a\mid s) = \frac{\exp R_\theta(s,a)}{\sum_{a'=1}^{K}\exp R_\theta(s,a')}
$$

Learning the reward that makes the expert's decisions likely under `q_θ` is maximum-likelihood IRL with relational demonstrations. Two approximations are explicit: the expert is modelled as Boltzmann-rational on the *immediate* reward (no soft value iteration through the transitions), and the reward is trained on the expert's states only while it is used on the learner's states ([Section 6.3](#63-reward-model-r_θ)). Acting is left to the separate policy `π_ω` (Section 6.4).

```mermaid
flowchart LR
    X["Unlabeled images<br/>(expert portion)"] --> O["Orbit of each image<br/>(views via 𝒯)"]
    O --> F["Forced moves<br/>twin already in expert slot k → action k"]
    X --> N["Random pairs<br/>different orbits → 'not together'<br/>(wrong ≈ 1/K of the time)"]
    F --> E["Expert decisions<br/>in the expert's own states"]
    N --> E
    E --> L["Expert likelihood under q_θ(a | s)<br/>= softmax of R_θ"]
```

### 5.3 Two forces: attraction and repulsion

Positives alone are not enough: putting everything in one cluster satisfies "an orbit stays together" perfectly. We need a force that says *different things should be told apart*, which is the "reward that distinguishes".

- **Attraction (forced moves).** When view `u` arrives and a twin is already in slot `k⁺(u)` of the expert's memory (the slot opened by the first view of its orbit), the reward must make joining that slot likely (here and below `s_u` is the expert's state):

$$
\mathcal{L}_{\text{pos}} = -\mathbb{E}_{u:\text{twin placed}}\big[\log q_\theta\big(k^+(u)\mid s_u\big)\big]
$$

- **Repulsion (random negatives).** For a view `u` and a random previously placed view `v` of another orbit, sitting in the expert's slot `k(v)`, the reward must make joining `k(v)` unlikely. With `ℓ(u,v) = −log(1 − q_θ(k(v) | s_u))`, the naive negative term is `E_v ℓ(u,v)`. A fraction `τ⁺` of random `v` are actually of the same concept, so we subtract their share, approximated by a positive pair (the standard debiasing of Chuang et al. 2020): the anchor's own twin `v⁺` when one is already placed, and the batch average over the forced moves otherwise. We use `τ⁺ = 1/K` on Colored MNIST; on Waterbirds, where classes are imbalanced, `τ⁺` is swept as a hyperparameter (see assumption A2):

$$
\mathcal{L}_{\text{neg}} = \mathbb{E}_{u} \max\Big\{\tfrac{1}{1-\tau^+}\Big(\mathbb{E}_{v\sim\text{placed}} \ell(u,v) - \tau^+ \bar\ell^{+}(u) \Big), 0 \Big\}
$$

with `ℓ̄⁺(u) = −log(1 − q_θ(k⁺(u) | s_u))` if a twin of `u` is placed, and the average of this quantity over the forced moves of the batch otherwise (so that the correction is defined for first views too).

The expert loss is `L_expert = L_pos + L_neg`, and it is what trains the reward (Section 6.3). The same repulsion is what makes an **empty slot attractive**: when an image fits no existing group, the mass leaves the occupied slots and goes to an empty one, i.e. the reward values *opening a new cluster*. In an expert episode an empty slot is always available when an orbit starts, which is why `B ≤ K`.

**Why random negatives are not an intervention.** We never choose pairs using knowledge of concepts. The only assumption is that random pairs are rarely of the same concept, with a *known* error rate `τ⁺`. Choosing negatives from images we know to be different would inject labels at the start and defeat the unsupervised goal; we do not do it. In Phase 2, the candidate pairs mined by the model are excluded from the repulsion as well ([Section 5.5](#55-phase-1-and-phase-2-tolerating-then-correcting-the-expert)).

### 5.4 Why two different cat photos end up together

In Phase 1 the expert keeps two different orbits in two different slots, so nothing in the demonstrations says "these two cats belong together". The merge is the **learner's**: it is free to put them in one cluster, and with a limited number of slots it typically has to. Three things make it happen:

1. **Limited slots.** There are at most `K` clusters. Once the slots are in use, a new orbit must join an existing group, and orbits that look alike must share one. While free slots remain, opening a new cluster is possible but is rewarded only when the image is genuinely incompatible with the groups that exist.
2. **Similarity in `z`.** Orbits with similar codes get similar rewards for the same group. Two cats share "cat features", so they fall in the same group.
3. **The Information Bottleneck decides what "similar" means.** If `z` still contains the background, merging happens along the background axis (cats on grass with dogs on grass). The compression is what makes the merging happen along the animal.

So the encoder is not a side detail: it determines *along which axis* the expert's weak signal is turned into categories.

### 5.5 Phase 1 and Phase 2: tolerating, then correcting the expert

The project runs the expert in two phases and compares them. Both use the same learner, the same reward and the same losses; only the expert's demonstrations change.

**Phase 1: the expert is tolerated as it is.** Each image is its own orbit, so the expert separates two different cat photos. This is a known and accepted flaw: the expert is not optimal and it does not parametrize the learner ([Section 5.2](#52-the-key-idea-an-images-transformations-are-an-expert-demonstration)). The question of Phase 1 is how far this flaw biases the learner. The bias is not small on paper: with `K = 10` balanced classes, an episode of `B` random images contains at least one same-class pair with probability 10% for `B = 2`, 28% for `B = 3` and about 70% for `B = 5` (the episode is collision-free with probability 30%), and the expert places each such pair in two different groups. The repulsion `τ⁺` correction absorbs part of this, but nothing in the demonstrations ever *attracts* two different cats. Phase 1 measures what the learner does with this: does it still merge the cats (limited slots, similarity in `z`, [Section 5.4](#54-why-two-different-cat-photos-end-up-together)), or does it fragment into one cluster per orbit?

**Phase 2: the expert is corrected during training.** Phase 2 changes the expert's demonstrations, slowly, so that two orbits that the model itself considers the same concept are treated as one. It does not change the learner or the loss. Its components:

1. **Relinking of the expert's slots.** A link between two images `i` and `j` merges their orbits for the construction of the expert episodes: in an episode that contains views of both, the expert puts them in the *same* slot of its own memory. A view of `j` arriving after a view of `i` becomes a forced move toward the slot of `i`, and the pair is removed from the repulsion (also from the `τ⁺` correction). The expert's memory is still per episode: what changes is the rule that fills it.
2. **Source of the links.** Links come from **mutual `k`-nearest neighbors in the code space of a slowly-moving copy of the encoder (EMA)**, not from the learner's policy: using the policy's own placements would make the expert agree with the learner by construction and would give the reward nothing independent to learn from. A link is accepted only if both images are neighbors of each other and their cluster assignment by the learner (used as a confidence filter, not as a label) is confident, `max_a π_ω ≥ γ_c`.
3. **Episodes enriched with neighbors.** In Phase 1 the images of an episode are drawn at random, so two cats rarely meet. In Phase 2, an episode is built from half random images and half of their mutual neighbors. This is what puts candidate same-concept pairs in front of the expert, so that a link has an effect on the reward.
4. **Brakes.** The expert moves while the reward is still learning, and the model can confirm its own mistakes. We therefore:
   - **cap** the number of accepted links per epoch (`≤ M`) and **ramp up** their weight over training;
   - require **persistence**: a link must be proposed over several consecutive epochs before it is used;
   - apply **no automatic transitive closure**: if `i~j` and `j~k`, then `i~k` is not added by itself (chains drift from one concept to another, and a single wrong link would merge two classes);
   - **decrease the effective `τ⁺`** as links remove part of the false negatives from the repulsion;
   - monitor the **churn** (fraction of images changing cluster between epochs) and the number of links used.

**Comparison protocol.** The two phases are run with the same seeds, the same dataset, the same hyperparameters and the same budget. We report ACC, NMI, worst-group ACC, spurious leakage and the number of constructed clusters for Phase 1 and for Phase 2, and conclude whether Phase 2 **improves or degrades** the result. A degradation is a legitimate outcome and would be reported as such (confirmation bias, chain drift). The ablations of [Section 9](#9-ablations) remove the brakes one by one. Two Phase 1 diagnostics make the comparison interpretable: the **same-class rate among the nearest neighbors in `m`-space** (does the learner's intent space already group different cats?) and the number of non-empty slots against the number of expert orbits per episode.

**Considered, not adopted: initializing the expert with a pixel heuristic.** For Colored MNIST, a nuisance-invariant heuristic on raw pixels (e.g. grayscale binarized shape) could provide initial cross-orbit links. It would inject a prior about the classes and about the dataset, so it is not part of the method; it could be used as a sanity check to see how much a good initial linking would help.

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
| Order dependence | Different clusters for different image orders within an episode | Random order every episode; summaries recomputed with the current encoder at each epoch; report variance across orders at test time |
| Distribution shift, expert vs learner | The reward is trained on the expert's small single-orbit groups and applied to the learner's large mixed groups | No count `n_k` in the reward input; memory summaries are averages; diagnostic: reward statistics on expert vs learner states; ablation with expert labels on learner states ([Section 9](#9-ablations)) |
| Stale summaries | Summaries computed by an old encoder no longer match the intents of new images | Recompute the summaries from the member lists at each epoch; ablation against a slow EMA |
| Expert bias (Phase 1) | Two images of the same concept are separated by the expert, so the learner fragments by orbit | Limited slots; Phase 1 vs Phase 2 comparison; same-class rate among nearest neighbors in `m`-space |
| Non-stationary state and reward | The policy chases a reward and a memory that both move | Alternating updates with a slow reward relative to the policy; frozen reward during the policy step (Section 6.5) |
| RL instability | Return oscillates, policy entropy collapses | Entropy bonus `α`, small policy learning rate, critic target network, comparison with the closed-form `softmax(R_θ)` control |
| Semantic drift along chains | Views in an orbit no longer show the same object | Bound `D_max` and cumulative strength |
| Over-strong transformations | Crops that remove the object, noisy "twin" decisions | Bound the strength of each `t ∈ 𝒯` |
| Confirmation bias and chain drift (Phase 2) | Churn does not decrease, errors reinforced, distinct classes merged through a chain of links | EMA copy, mutual neighbors, confidence `γ_c`, rate cap `M`, persistence over several epochs, no automatic transitive closure |
| Moving expert (Phase 2) | Reward loss oscillates | Change the expert slowly relative to reward learning (cap and ramp-up) |
| False negatives | About `1/K` of random pairs are the same concept | Debiasing; check for class imbalance |

### 5.8 Relation to existing work

Taken separately, the pieces are known: orbit positives and random negatives are contrastive learning (SimCLR), the debiasing is Chuang et al. 2020, the group-joining decision is close to online prototype / sequential clustering, the neighbor mining of Phase 2 is close to SCAN (mutual nearest neighbors), and relational supervision is constrained clustering. **What we test** is whether casting them as expert demonstrations for a latent reward, with an RL policy that builds the clusters, an Information Bottleneck on the encoder and a decision-level invariance check, removes spurious features better than the same pipeline with a plain classification head (see [Ablations](#9-ablations)).

---

## 6. Components & Training

The architecture has four trained components: the **encoder**, the **intent predictor**, the **reward model** and the **policy** (a separate network with its critic, trained by RL), plus the learner's non-differentiable **cluster memory** that the policy writes to (the expert has a memory of its own, per episode). This section fixes the notation, then gives for each component its role, its loss, and the intuition behind each term. It closes with what joint and sequential training mean in practice, and exactly what is frozen where.

### 6.0 Notation

- For a view `u`: `z_u = f_φ(u)` and `m_u = μ_ψ(z_u)` (mean of the inferred intent).
- `C = {(n_k, c_k)}` is the learner's cluster memory (Section 4.2): `c_k` is the average of the intents `m_v` of the members of cluster `k`, recomputed with the current encoder at each epoch. `s_u = (z_u, m_u, C)` is the state in which `u` arrives. In an expert episode the memory is the expert's own, `C^E`, and the state is written `s^E_u = (z_u, m_u, C^E)`. Memories are always treated as constants. States and summaries always use the **single-view** intent `m_u`, as at test time; the orbit mean `m*` below is only a training target for `q_ψ`.
- `R_θ(s, a)` is the reward of placing the arriving image in slot `a`. `q_θ(a | s) = softmax_a R_θ(s, a)` is the choice model it induces, used to score the experts (Section 5.2).
- `π_ω(a | s)` is the **policy**, a network of its own, trained by RL against `R_θ`. It is not `q_θ`.
- A bar (`θ̄`, `ω̄`, `m̄`, `C̄`) means **stop-gradient**: the value is used in the computation but not updated by that loss.
- Expert signals (Section 5), always in the expert's states `s^E`: forced moves (twin already placed in the expert's memory), random previously placed views (repulsion), with `τ⁺ = 1/K`. The expert never acts on the learner: only `L_θ` reads them.
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

**Role.** Scores every placement `(s, a)`: how well the arriving image, coded by `z`, fits the group in slot `a` **and** how well it is told apart from the other groups in the memory `C`. It is the object that IRL recovers, the *intention of distinction*, shared by the whole dataset. It is a permutation-equivariant network over the slots, `R_θ(z, m, C, a) = g_θ(z, m, c_a, pool_{k≠a} c_k)`: it reads the chosen slot together with a pooled view of the others, so the identity of a slot never matters, only its content (the first part measures compatibility, the second distinctness). It does **not** read the counts `n_k`, which would differ between the expert's small groups and the learner's large ones. It is trained by IRL on the expert's states `s^E` and applied to the learner's states `s`, which is a known distribution shift ([Section 5.7](#57-failure-modes-and-what-we-monitor)); it is **not** the policy.

**Loss.** Maximum-likelihood IRL of the expert's decisions (Section 5), under the choice model `q_θ`:

$$
\mathcal{L}_\theta = \mathcal{L}_{\text{pos}} + \mathcal{L}_{\text{neg}} - \rho\mathcal{H}\big(\mathbb{E}_sq_\theta(\cdot\mid s)\big), \qquad \rho = 0 \text{ by default}
$$

**Reading the terms.**
- **`L_pos` (attraction, forced moves).** Raises the reward of the twin's slot, in the expert's memory, for the arriving view.
- **`L_neg` (repulsion).** Lowers the reward of joining a slot that holds a different orbit in the expert's memory, with the false-negative correction `τ⁺`. This is the "reward that distinguishes": it exists because the softmax inside `q_θ` is *relative*, so lowering a cluster's reward for one image raises the others, empty slots included.
- **`ρ·H(marginal)` (off by default, `ρ = 0`).** Entropy of the average choice distribution over the batch, a safeguard against collapse. With repulsion in place it should be unnecessary. The activation rule is fixed in advance: we monitor the effective number of clusters `K_eff = exp(H(p̄))`, where `p̄` is the average assignment distribution; if `K_eff` falls below `0.7·K` (initial value, to be validated) during the first epochs, a small `ρ` is switched on. We avoid a large `ρ` by default because uniform cluster usage is wrong for imbalanced classes (e.g. Waterbirds), and because `K` is only an upper bound.

### 6.4 Policy `π_ω` (separate network, trained by RL)

**Role.** The agent that builds the clustering. At each step it sees the arriving code and the clusters built so far, and picks a slot: join a group, or open a new one. It is a separate network `π_ω(a | z, m, C)` (set-conditioned over the slots, like the reward), with a critic `V_ξ(s)` for the return estimates. It acts on the learner's states only and never sees the expert, the orbits or the expert's actions. At inference the assignment is `a = argmax_a π_ω(a | s)` with the memory `C` frozen at the end of training.

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

1. **Two batches.**
   - *Expert batch.* A minibatch of expert episodes ([Section 5.2](#52-the-key-idea-an-images-transformations-are-an-expert-demonstration)), fixed in structure: `B` images, `n` views each, shuffled, placed by the expert in its own memory `C^E`. The codes `z_t`, intents `m_t` and the memory summaries are recomputed with the current encoder. The pipeline labels the steps (forced moves, repulsion pairs). The learner's policy plays no role in this batch.
   - *Learner rollout.* The current policy `π_ω` sweeps a minibatch of training images, one view each, placing them one by one, seeing only `(z_t, m_t, C_t)`; the persistent memory `C` is updated by each transition. No label is used.
2. **Reward step (IRL).** Minimize the total loss on `φ, ψ, θ`, with `L_ψ`, `L_θ` computed on the expert batch and `L_inv` on the learner's states:

$$
\mathcal{L} = \mathcal{L}_\psi + \mathcal{L}_\theta + \beta\mathrm{KL} + \lambda\mathcal{L}_{\text{inv}}
$$

3. **Policy step (RL).** With `φ, ψ, θ` frozen, minimize `L_ω` (and the critic loss) on `ω, ξ` on learner rollouts. The reward is evaluated on the learner's states; it has never seen the learner's decisions.
4. **End of epoch.** Recompute the summaries `c_k` of the learner's memory from the member lists with the current encoder.

This is the objective of Section 4.4, plus the RL loss of Section 6.4, which is why the training is alternating and not a single loss. Gradient routing (what each term updates, and what it treats as constant):

| Term | Updates | Frozen (stop-gradient) |
|---|---|---|
| `L_ψ` (intent) | `ψ`, and `φ` through `z` | The orbit target `m*` |
| `L_θ` (expert likelihood) | `θ`, and `φ` through `z` | The memory `C^E` and the intent `m`: `ψ` gets no gradient from `L_θ`. The intent is trained only by `L_ψ`, so the reward cannot bend it to make its own job easier, and the cluster summaries stay stable |
| KL (compression) | `φ` | None |
| `L_inv` (invariance) | `φ` only, through both `z` and `z_t` | `ω`, `ψ` (hence `m`) and `C` |
| `L_ω` (RL return + entropy) | `ω` (and the critic `ξ`) | `φ`, `ψ`, `θ`, `C`: the policy step treats the encoder and the reward as given |

**What is frozen in `L_inv`, and why.** With `ω̄`, `m̄`, `C̄` frozen, the invariance term has gradient `∇_ω L_inv = 0` and `∇_ψ L_inv = 0`, while `∇_φ L_inv ≠ 0`. This means the policy is used as a *fixed measuring stick*: if the decision it takes changes under a perturbation, the *encoder* is what gets corrected. If `ω` were free, the term would have a trivial minimizer: any policy that does not depend on the image,

$$
\pi_\omega(\cdot\mid z,C,m) = \pi_\omega(\cdot\mid C,m) \Longrightarrow \pi_\omega(\cdot\mid z,C,m)=\pi_\omega(\cdot\mid z_t,C,m) \Longrightarrow \mathcal{L}_{\text{inv}} = 0 ,
$$

so the check would push the policy toward ignoring the image instead of pushing the encoder toward dropping the nuisance. The same holds for `m`: a free `ψ` could make `m` insensitive and cheat the term. Freezing is local to `L_inv`: `ω` keeps being trained by RL through `L_ω`, and `θ` by the experts through `L_θ`, so the ruler still evolves, but never through the check. The ablations "`ω` unfrozen" and "`q_θ̄` instead of `π_ω̄`" ([Section 9](#9-ablations)) measure how much these choices matter.

**What it means.** "Useless" is defined *relative to the reward and policy currently being learned*: the encoder compresses what they do not use, while they adapt to whatever the encoder keeps. This co-adaptation is the point of the method, and also its main risk: the early reward is random, so it can push the bottleneck to discard useful bits, and the early policy is random, so the memory `C` is noisy. Mitigations: a warm-up of `β` from 0, a reward warm-up on forced moves before the policy acts alone, and a reward updated more slowly than the policy. Since the expert's trajectories are fixed, the reward does not chase the policy's own memory; the only moving parts it sees are the encoder and, in Phase 2, the expert's links.

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
z_u = \mu_\phi(u)+\sigma_\phi(u)\odot\varepsilon,\qquad m_u = \mu_\psi(z_u),\qquad s_u=(z_u,m_u,C),\qquad s^E_u=(z_u,m_u,C^E),\qquad C=\{(n_k,c_k)\}_{k=1}^{K}
$$

**Transition (cluster memory, for both the learner's `C` and the expert's `C^E`).** After the assignment `a`, the chosen slot absorbs the intent of the view (an empty slot, `n_a = 0`, thereby becomes a new cluster):

$$
n_a \leftarrow n_a+1,\qquad c_a \leftarrow c_a+\frac{m_u-c_a}{n_a}
$$

**Memory recompute (learner's memory, end of each epoch).** With `M_k` the list of members of slot `k` and the current encoder:

$$
c_k \leftarrow \frac{1}{|M_k|}\sum_{u\in M_k}\mu_\psi\big(\mu_\phi(u)\big)
$$

**Intent (Section 6.2).** Every view of an orbit must give the intent of the orbit:

$$
\mathcal{L}_\psi=-\mathbb{E}_{i,u\in\text{orbit}_i}\big[\log q_\psi(m^*_i\mid z_u)\big],\qquad m^*_i=\mathrm{sg}\Big[\tfrac{1}{|\text{orbit}_i|}\textstyle\sum_{v\in\text{orbit}_i}\mu_\psi(z_v)\Big]
$$

**Reward (IRL on the experts, Sections 5.2, 5.3 and 6.3).** The choice model induced by the reward and the expert loss, evaluated in the expert's states:

$$
q_\theta(a\mid s)=\frac{\exp R_\theta(s,a)}{\sum_{a'}\exp R_\theta(s,a')},\qquad
\mathcal{L}_{\text{pos}}=-\mathbb{E}\big[\log q_\theta(k^+(u)\mid s^E_u)\big]
$$

$$
\mathcal{L}_{\text{neg}}=\mathbb{E}_{u}\max\Big\{\tfrac{1}{1-\tau^+}\Big(\mathbb{E}_{v}\ell(u,v)-\tau^+\mathbb{E}_{v^+}\ell(u,v^+)\Big),0\Big\},\quad \ell(u,v)=-\log\big(1-q_\theta(k(v)\mid s^E_u)\big)
$$

with `k⁺(u)` and `k(v)` slots of the expert's memory `C^E`, and `ℓ(u, v⁺)` replaced by the batch average over the forced moves when `u` has no placed twin.

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
| 12 | **The expert does not parametrize the learner.** The learner's policy `π_ω` never sees orbits, the expert's slots or the expert's actions, and nothing restricts the memories and clusters it builds. | The expert is not optimal (in Phase 1 it separates two cats); it is only used to guide the reward toward what it means to tell things apart. | Constraining the learner to the expert's groups; policy imitation of the expert. |
| 13 | **Fixed expert trajectories with a per-episode expert memory.** Demonstrations are built before training from a portion of the images: each orbit is a slot of the expert's own memory, the first view opens it, the later views join it. Only the structure is stored; `z`, `m` and the memory summaries are recomputed with the current encoder. | Same status as expert demonstrations in classical IRL: data produced by the expert in its own states, independent of the learner. The learner rolls out on its own states. | Labelling the learner's own placements on the fly (a DAgger-like variant; kept as an ablation): the signal would become "be consistent with your own placements". |
| 14 | **The learner's memory persists, with member lists and summaries recomputed with the current encoder at each epoch.** | The clusters must survive an episode to be clusters; the encoder moves, so stored summaries go stale. | Memory reset every episode (ablation); summaries updated by a slow EMA only (ablation). |
| 15 | **The counts `n_k` are not given to the reward.** | A slot holds at most `n` views in an expert episode and thousands in a learner episode; a count would be out of distribution. | Reward reading `n_k`. |
| 16 | **Two phases.** Phase 1 tolerates that the expert separates two images of the same concept and measures how much this biases the learner. Phase 2 relinks the expert's slots progressively (mutual neighbors of an EMA encoder, neighbor-enriched episodes, brakes) and measures whether this improves or degrades the result. | Phase 1 gives the baseline against which the correction is evaluated; the correction is risky (confirmation bias, chain drift) and must be shown to help. | Starting with the correction; initializing the links with a pixel heuristic (set aside: it injects a prior on the classes). |

### 7.2 Still open

No design question is open at this stage. Points to settle by experiment:

1. Whether the reward and the policy should see `z` and `m`, or only `m` (ablation, [Section 9](#9-ablations)).
2. The update rate of the memory within an epoch (running mean vs. slow EMA) and the RL hyperparameters (`γ`, `α`, ratio between reward and policy steps).
3. Whether the reward needs a warm-up on forced moves before the policy acts alone (cold start).
4. The size of the expert portion `𝒟_E` against the whole training set used by the learner, and the number of orbits `B` and views `n` per expert episode.
5. Whether the views of each orbit are resampled at each epoch while the structure of the expert episodes (images, order, expert slots) stays fixed (proposed, to be confirmed).

### 7.3 Phase 2 parameters

- **Phase 2 (progressive merging).** When to start relinking and how fast (`k`, `γ_c`, cap `M`, ramp-up, persistence length, EMA momentum, share of neighbors in the episodes, schedule of the effective `τ⁺`), given the risk of confirmation bias. These are tuned after Phase 1 is stable; Phase 1 and Phase 2 are compared as described in [Section 5.5](#55-phase-1-and-phase-2-tolerating-then-correcting-the-expert).

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

Training-time diagnostics (Section 5.7): leakage from step 0, number of non-empty slots, cluster-size entropy, churn between epochs, return and policy entropy, sensitivity to the order of images, `NMI(m, background)`, and the **same-class rate among the nearest neighbors in `m`-space** (Phase 1: does the learner group different images of one concept although the expert does not?). Phase 1 and Phase 2 are always reported side by side ([Section 5.5](#55-phase-1-and-phase-2-tolerating-then-correcting-the-expert)).

Reporting: mean ± std over several seeds, with `K` (upper bound) set to the number of true classes and the number of constructed clusters reported alongside.

---

## 9. Ablations

| Ablation | Question answered |
|---|---|
| **Plain classification head** instead of reward + policy + `m` (same experts, same encoder) | Does the IRL / RL formalism add anything over contrastive / IIC-style clustering? |
| **Closed-form policy `softmax(R_θ)`** (or `γ = 0`) instead of the RL-trained `π_ω` | Does the separate policy and the sequential return bring anything beyond a per-image reward? |
| **Reward per image, no cluster memory** (`R_θ(z, a)` only) | Does conditioning on the clusters built so far matter? |
| Persistent memory vs memory reset every episode; random vs fixed image order | How much does the result depend on the way the clusters are built along the sequence? |
| Summaries recomputed each epoch vs slow EMA only | Do stale summaries matter? |
| Expert labels on the learner's own rollouts (DAgger-like) instead of fixed expert trajectories | Does labelling the learner's states help the reward, or does it make the signal circular? |
| Reward reading the counts `n_k` | Does the count create a shift between expert and learner states? |
| `β = 0` (no Information Bottleneck) | Does compression alone remove the spurious features? |
| `λ = 0` (no decision-consistency term) | Is the invariance term necessary, or is the IB sufficient? |
| `q_θ̄` instead of `π_ω̄` in `L_inv`; raw reward differences; `ω` and `ψ` unfrozen in `L_inv` | Does the form of the check or the freezing matter (does the policy drift toward ignoring the image)? |
| `ρ > 0` from the start | Does an explicit anti-collapse term help, or hurt on imbalanced classes? |
| No negatives (positives only) | Does collapse appear, as predicted? |
| No debiasing (`τ⁺ = 0`) | How much do false negatives hurt? |
| `𝒯` families (with / without background-targeted transformations) and `D_max` sweep | How much of the result comes from prior knowledge about nuisances (A1)? |
| Phase 1 vs Phase 2 | Does relinking the expert's slots improve or degrade the clustering? |
| Phase 2 with random episodes vs episodes enriched with neighbors | Are the neighbors in the episodes what makes the links effective? |
| Phase 2 with brakes removed one by one (EMA, mutual neighbors, `γ_c`, cap `M`, persistence, no transitive closure) | Which brake matters? |
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
│   ├── memory.py     # learner's persistent memory C: member lists, slot summaries, recompute each epoch; expert's per-episode memory C^E
│   ├── reward.py     # reward R_θ(z, m, C, a) and induced choice model q_θ
│   ├── policy.py     # RL policy π_ω(a | z, m, C) and critic V_ξ
│   ├── experts.py    # expert trajectory generator (orbits, forced moves, random pairs, debiasing) and Phase 2 relinker (mutual-kNN links, brakes)
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
- [x] Expert design: fixed expert trajectories with a per-episode expert memory, relational decisions, debiased negatives; Phase 1 / Phase 2 protocol
- [x] Policy design: separate RL policy conditioned on the clusters built so far, `K` as an upper bound, alternating training
- [x] Design questions settled: experts, policy, `m` ([Section 7.1](#71-decisions-taken))
- [ ] Variational formulation validated (Section 4.4) and consistency term fixed (Section 4.5)
- [ ] Expert trajectory generator, learner's persistent memory (member lists, recompute) and monitoring (leakage, churn, non-empty slots, cluster-size entropy)
- [ ] Stable reward / policy alternation on Colored MNIST (RL stability, order dependence)
- [ ] Phase 1 on Colored MNIST (expert tolerated as is), with the same-class rate among neighbors in `m`-space
- [ ] Colored MNIST pipeline and baselines
- [ ] Phase 2 (relinking, neighbor-enriched episodes, brakes) and Phase 1 vs Phase 2 comparison
- [ ] Waterbirds pipeline and baselines
- [ ] Full ablation suite
- [ ] Write-up

---

## 12. License

MIT License
