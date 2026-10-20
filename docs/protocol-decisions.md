# Protocol decisions (2026-10-13)

## 1. Family of transformations T
T is not a robustness setting: it defines the invariance that is imposed (what T changes = nuisance, what it preserves = identity). A bank is a written, versioned object (`configs/transforms/*.yml`, module `cibirl/transforms.py`).

- `generic`: moderate crop + brightness/contrast/saturation jitter + rare grayscale. Applied as is to all datasets (main table). Never touches the hue.
- `natural`: `generic` + horizontal flip. Natural images only (a mirrored digit or a mirrored toy shape is another identity).
- `informed_color`: `generic` + full redraw of the hue. A DECLARED prior on the nuisance, presented as an experimental condition.
- `toy_oracle`: exact redraw of the toy background (upper bound of the prior).

Central result on Colored MNIST (and on the toy with a shortcut): present BOTH regimes, `generic` and `informed_color`. The gap measures how much the result depends on a prior about the nuisance.

Admission rule: an operation enters a bank only if domain knowledge, written BEFORE seeing the results, says that it preserves identity. Never chosen or tuned by looking at the labels. The strengths are fixed in advance; their effect is studied by a sensitivity sweep, not by keeping the best value.

## 2. Stopping rule (no labels)
Training runs until convergence (`cibirl/convergence.py`) or `max_epochs`. Convergence = over the last `window` (10) epochs: stable argmax partition (change <= 1 %, up to a renaming of the clusters), constant number of clusters, plateau of twin_acc (+-0.01) and of L_pos+L_neg (2 % relative), after `min_epochs` (40). The model evaluated is that of the last epoch. The labels are used only for monitoring (`eval_*` in metrics.jsonl).

Option `stop.memory: true` (5.25): additionally requires the partition of the memory to be that of the output (`mem_gap` <= 0.01 over the window); otherwise `mem_gap` is only logged.
Option `final_cleanup: true` (5.28): after training, with the policy frozen, merges the duplicate slots of the memory (largest jump in the distances between summaries, guard `cleanup_ratio` 10) and reads the output again; final protocol: stop at the convergence of the output + this last step. `cleanup_mode` (5.29) chooses how: `merge` (5.28), `relabel` (A: output read before the merge, slots renamed), `reintroduce` / `reintroduce_no_open` (C: sweeps over the images, argmax frozen); all four are recorded in each run.

Limit: "converged" = stable, not "good". To be read together with `clusters_used`, `m_std`.

## 3. Findings on the toy (100-epoch suites, 5 seeds; diagnostics on 1 to 2 seeds)
Final results at 100 epochs (last epoch, no selection): `toy` ACC 0.72 +- 0.16; `toy_spur0.9` 0.64 +- 0.08; `toy_no_standardize` 0.39 +- 0.02. These are states AFTER degradation: at 20-40 epochs several runs had 5 pure clusters (ACC 1.00).

Established facts:
- The representation m is good: the mean of each class is the closest centre for 100 % of the images in 9 runs out of 10 (ablation excluded). The encoder is not the defect.
- The policy reproduces the argmax of the reward alone (policy ACC = reward ACC in almost all runs): the merges come from the reward / the memory, not from PPO.
- Two distinct problems:
  1. Degradation of the reward by compression: with beta = 1e-3, KL falls to ~40 and twin_acc goes from ~0.83 to ~0.5, the entropy of the policy rises to ~1.1 (almost random), churn ~0.6. Clear association (KL >= 80: twin 0.79-0.82; KL < 60: 0.50-0.54), confirmed by a control: with beta = 1e-4 or 0, twin_acc stays ~0.8 (seeds 1 and 3).
  2. Instability of the assignment, independent of beta: abrupt merge of two classes (e.g. oracle seed 0: 5 clusters, ACC 1.00 from epoch 18 to 55, then 4 clusters, ACC 0.80, with no sign in twin_acc / KL), or contamination of the clusters (beta = 0: 5 clusters but ACC 0.71-0.77). With a smaller beta the final ACC is not better.
- With an ideal memory (class means), the reward almost never opens a new slot once 2 slots are occupied (right decisions ~0.52-0.60 with 2 occupied slots): it "joins the closest". Caveat: the ideal summaries may differ from those seen in training (mean of 2 views).
- Reproducibility across platforms (Windows / Linux): good for seeds 0 and 1, partial for seed 3 (chaotic dynamics: tiny numerical differences change the trajectory).

There is not yet a label-free criterion that detects problem 2 (twin_acc and KL do not move during a merge). Choosing the best epoch on the labels is forbidden; we report the last epoch AND the trajectory (`eval_*`).

The stopping rule (window 5 -> 10, measure invariant to renaming) was adjusted by looking at runs of seed 0: a choice to flag, to be tested in sensitivity (window 5 / 10 / 20).

## 5. Next steps (decided on the evening of 2026-10-13)
- Do not launch `toy_regimes` as is: it would measure a degraded dynamics (beta = 1e-3).
- Problem 1: DECIDED on 2026-10-13: `beta = 1e-4` becomes the default of `TrainConfig` (cibirl/train.py). The `results/toy_*` runs made with beta = 1e-3 are the previous state. To watch in the runs: a drop in twin_acc (label-free indicator).
- Problem 2: see 5.1 (three variants tested, none fixes it). Next candidate: the gap between the summaries seen by the reward while training on the experts and those of the learner.

### 5.1 Suite `toy_dynamics` (beta = 1e-4, toy without shortcut, oracle bank, 100 fixed epochs, 5 seeds)
Reference against three variants (`configs/suites/toy_dynamics.yml`). ACC at the end of the run, mean +- standard deviation over the seeds; "max ACC" is the best epoch (labels, information only).

| condition | final ACC | mean ACC last 20 | max ACC | final clusters | twin_acc | policy entropy |
|---|---|---|---|---|---|---|
| ref | 0.74 +- 0.17 | 0.77 | 0.92 | 4.2 | 0.81 | 0.49 |
| consolidate (memory = argmax at the end of the epoch) | 0.60 +- 0.00 | 0.64 | 0.84 | 3.0 | 0.81 | 0.11 |
| cold (T = 0.5, alpha = 0) | 0.74 +- 0.08 | 0.80 | 0.96 | 4.0 | 0.81 | 0.21 |
| orbits5 (B = 5) | 0.72 +- 0.20 | 0.76 | 0.92 | 3.6 | 0.65 | 0.72 |

Reading:
- No variant improves the final ACC in a way that is distinguishable from the reference (n = 5, standard deviation 0.1 to 0.2). `cold` reduces the dispersion (0.08 against 0.17) and gains a little on average over the last 20 epochs; to be confirmed on more seeds.
- `consolidate` is worse: 3 clusters on all 5 seeds, ACC 0.60 everywhere. Without random sampling, the merges are never undone.
- The hypothesis "exploration noise contaminates the memory" is not supported: the entropy of the policy is already ~0.1 from epoch ~12 in all conditions, so the sampled actions are almost the argmax while the merges happen. The merge comes from the deterministic reward + memory dynamics.
- `orbits5` makes the expert task harder (twin_acc 0.65 instead of 0.81) with no final gain, with barely fewer abrupt ACC drops (8.0 against 9.8 on average, within the noise).
- Structural finding (3 orbits, 2 views): in the expert states, no slot has more than 2 members, so the reward is trained only on summaries = mean of 1 or 2 views, whereas the learner presents it with class centres averaged over hundreds of images. Not tested at this stage.

### 5.2 Suite `toy_oracle_link` (same conditions as the reference; beta = 1e-4, 100 epochs; 3 seeds, `rho` on 5)
Two questions: (a) does the absence of an attraction signal between orbits of the same class in the expert explain the merges? (b) does the anti-collapse term `rho` (disabled by default) help?
- `expert_oracle_link`: ORACLE DIAGNOSTIC that uses the true classes (never a method; `run.yml` carries `uses_labels_for_training: true`). In an episode, with probability p, the orbits of the same class share a slot (attraction between orbits, no repulsion within the class). This is what a perfect phase-2 relinker would produce with p = 1. It mixes two effects: the attraction between orbits and larger expert summaries (up to 4 views).
- `rho`: -rho x entropy of the marginal of the reward over the slots, measured on the expert states.

Seeds 0 to 2:

| condition | final ACC | mean ACC last 20 | epochs with ACC >= 0.95 | clusters |
|---|---|---|---|---|
| ref | 0.80 +- 0.16 | 0.83 | 31 | 4.0 |
| link 1.0 | 0.74 +- 0.04 | 0.68 | 0 | 4.7 |
| link 0.5 | 0.70 +- 0.08 | 0.79 | 35 | 4.0 |
| rho 0.1 | 0.93 +- 0.09 | 0.94 | 67 | 4.7 |
| rho 1.0 | 0.84 +- 0.12 | 0.94 | 77 | 4.7 |

`rho` on the 5 seeds (ref: 0.74 +- 0.17 / 0.77 / 24 epochs / 4.2 clusters): rho 0.1 -> 0.84 +- 0.13 / 0.81 / 40 / 4.8; rho 1.0 -> 0.82 +- 0.10 / 0.85 / 57 / 4.8.

Reading:
- Perfect experts (link 1.0) do not fix the merges: ACC 0.71 to 0.80, never 0.95, while twin_acc = 1.0 (the reward predicts the expert perfectly). The missing inter-orbit signal is therefore not the main cause.
- The representation m is not at fault: the closest class centre gives 100 % of right assignments in all conditions (seeds 0 to 2).
- Test of the ideal memory (slots = true class centres): with the 5 slots occupied, the reward and the policy place 100 % of the images in their class; with 1 to 4 occupied slots, only 50 to 95 % (deciding whether to open a slot or to join an existing slot under a partial memory). link 1.0 improves the reward alone with 1 occupied slot (0.92 against 0.52 to 0.76), not the policy.
- `rho` is the only lever that moves anything, modestly on 5 seeds (final ACC +0.08 to +0.10, a few more epochs at >= 0.95, closer to 5 clusters), with differences of the order of the standard deviation: a hint, not a proof. Seeds 3 and 4 with rho 1.0 reach 1.0 and then fall back to 0.6 around epoch 100: the late drift remains.

### 5.3 Suite `toy_formation` (same conditions as the reference; 5 seeds; placement diagnostics)
Hypothesis tested: "at the start the memory is poorly informative, the uncertainty contaminates the clusters and the contamination persists".
New: `cibirl/diagnostics.py` (statistics `eval_place`, `eval_purity`, `eval_depleted` per epoch, with the labels, evaluation only; no effect on training: 0 differences over 48 values and same ACC/NMI as the old reference), `oracle_warm_memory` (ORACLE DIAGNOSTIC: 5 images per class in the memory when the learner starts to act), `burn_in_epochs` / `burn_in_fraction` (label-free: the memory is rebuilt from a draw of 10 % of the members of each cluster, counted in epochs from the start of the placements).

Test 1, on the reference:
- The memory is full almost all the time: out of 37,500 placements of epochs 10 to 14 (the learner places nothing before epoch 10, because of the 150-iteration warm-up of the reward), only 63 take place with fewer than 5 occupied slots. The "partial memory" hypothesis is therefore ruled out (it was an extrapolation of the ideal-memory test, which does not describe the states actually visited).
- No slot empties during a run (0 out of 20). The placement errors (21 to 37 % of the placements) come from a full memory; they mostly reflect the merged classes.

Interventions (final ACC, mean +- standard deviation; reference 0.74 +- 0.17): warm 0.80 +- 0.13; burn3 0.78 +- 0.15; burn8 0.69 +- 0.20. None is distinguishable from the reference.
Caveat on warm: the untrained policy overwrites the starting memory in ~25 placements (purity 0.6 to 0.8 at epoch 12 despite a clean start).

What contradicts early contamination as the main cause: 8 runs out of 20 reach a pure state (ACC >= 0.95 for at least 10 consecutive epochs) and 4 lose it afterwards. Seed 4: purity ~1.0, deterministic policy, ACC 1.0 from epoch 13 to 24, then an abrupt break (churn from 0.04 to 0.41, one cluster out of five comes apart) with no sign in twin_acc or in the losses of the reward. It occurs at epochs 24 to 29 in four different conditions (reference, warm, burn3, burn8), while seed 2 stays pure to the end in three of them. The pure equilibrium is therefore unstable for some seeds, independently of the history of the memory.
Proposed next step: autopsy of seed 4 (save m and the policy margins at epochs 22 to 28).

### 5.4 Suite `toy_rho` (10 seeds): correction of 5.2
Same setting as the reference (beta = 1e-4, 100 epochs). Results on 10 seeds (final ACC, mean +- standard deviation; paired difference with the reference +- standard error):

| condition | final ACC | mean ACC last 20 | epochs >= 0.95 | clusters | final difference vs ref |
|---|---|---|---|---|---|
| ref | 0.70 +- 0.16 | 0.73 | 24 | 4.2 | |
| rho 0.1 | 0.77 +- 0.15 | 0.78 | 26 | 4.5 | +0.07 +- 0.06 |
| rho 0.3 | 0.71 +- 0.10 | 0.74 | 14 | 4.0 | +0.01 +- 0.07 |

The gain of `rho` seen on 3 to 5 seeds (5.2) is not confirmed: +0.07 is about 1.2 standard errors for rho 0.1, and rho 0.3 changes nothing (fewer epochs at >= 0.95). Conclusion: `rho` is not a lever against the merges; it stays disabled by default. With 10 seeds the reference is 0.70 +- 0.16 (0.74 on seeds 0 to 4), which gives the order of magnitude of the noise between sets of seeds.

### 5.5 Autopsy of seed 4 of the reference (pure at epochs 13-24, breaks at epoch 25)
Deterministic replay (1 thread, identical to the suite), measurements on epochs 18-30. The labels are used only for evaluation.

* **What does not flip**: the representation (accuracy of the closest class centroid in m = 1.00 at every epoch); the policy and the reward under an ideal memory (true class means) stay 100 % right, even after the break.
* **What flips**: the content of the memory. At epoch 25 two classes end up in the same slot of the sampled memory (dominant slots [3,1,4,2,0] -> [3,3,2,1,0]); the summaries become mixtures, the margin of the policy falls from 4.3 to 0.3-0.9, and afterwards an entire class of 300 images changes prediction from one epoch to the next.
* **Staleness measurement** (before each `recompute`): mean distance between the intent stored at placement time and the current intent, relative to the smallest distance between class means.

| | seed 4 (breaks) | seed 2 (stable) |
|---|---|---|
| absolute drift of the stored intents | 0.10-0.15 | 0.10-0.16 |
| smallest distance between classes (in m) | 0.33 (ep 16) -> 0.14 (ep 25) -> 0.06 (ep 27+) | 0.60 -> 1.2-1.3 (grows) |
| drift / distance | 0.2-2.1 (> 1 from epoch 27) | 0.05-0.23 |

Reading: the drift between epochs is of the same order in the two seeds; what differs is the minimum distance between classes, which collapses in seed 4 (median of the pair distances 0.80 -> 0.32, while the largest stays ~10: one very distant class squeezes the others into a corner of the standardized space). When the drift exceeds the separation, summaries built during an epoch no longer designate the right class. Limits: a single unstable seed and a single control; the drift already exceeds the separation at epochs 19-21 of seed 4 without a break, so it is not a sufficient condition; the cause of the collapse of the separation is not established; the geometry of seed 2 (max/median of the pairs) was not measured.
Leads to test next: (a) recomputing the summaries more often than once per epoch; (b) freezing or slowing the encoder after the formation; (c) understanding why the minimum separation decreases (role of the BN standardization with a very distant class, and of `L_neg` alone on the free moves).

### 5.6 The three analyses of the break: geometry of m on 10 seeds, frequent recompute, frozen encoder
Same setting as the reference (beta = 1e-4, 100 epochs; the results of `ref` are those of `toy_formation`, reproduced identically). The labels are used only for evaluation.

**(c) Geometry of m on the 10 seeds of `toy_rho/ref` (replay of 50 epochs).**
* The minimum separation between classes does not predict the break: seed 9 has a separation of 0.03 to 0.26 between epochs 20 and 49 (often smaller than the drift of the stored intents, 0.15 to 0.26) and stays pure 37 epochs out of 40; seed 2 has a separation of 1.2 and stays pure.
* What is associated with the instability (seed 4: break; seed 3: never pure, ACC oscillates) is the collapse of the MEDIAN distance between class means: 0.57 (seed 3) and 0.80 (seed 4) at epoch 14, against 3.7 to 7.0 for the 8 other seeds; one class stays at distance ~10 while the other four pack together. Association on 2 seeds, not a proof.
* BN is not the direct cause: the compression is already present before the standardization (raw median distance 1.0 at epoch 14 for seed 4, against 8.7 for seed 9). Untested hypothesis: `L_psi` (orbit coherence) has no repulsion term between classes, and the unit variance per dimension imposed by BN is satisfied by a single distant class; `L_neg` acts on the reward and does not go through m (the intent is not trained by the reward).
* Seeds 0, 1, 5, 7, 8 (ACC 0.5 to 0.7, never pure) do not have this collapse (median distance >= 3.7 at epoch 14): their failure is another mode, the merge at formation.

**(a) and (b): `toy_stale`, 5 seeds.** Options added (disabled by default, default path unchanged bit for bit): `recompute_every` (recompute of the summaries every n iterations), `freeze_repr_after` + `repr_lr_scale` (4 epochs after the first placements, the encoder and the intent predictor are frozen, weights and BN statistics, or slowed down).

| condition | final ACC per seed (0,1,2,3,4) | mean | paired diff. vs ref | epochs >= 0.9 per seed |
|---|---|---|---|---|
| ref | 0.60 0.80 1.00 0.52 0.80 | 0.745 | | 0, 5, 89, 0, 28 |
| rec5 | 0.80 1.00 1.00 0.53 0.51 | 0.766 | +0.02 +- 0.09 | 0, 11, 89, 0, 0 |
| rec1 | 0.75 0.80 0.60 0.60 1.00 | 0.749 | +0.01 +- 0.11 | 0, 0, 0, 0, 88 |
| freeze4 | 0.70 0.72 1.00 0.73 1.00 | 0.832 | +0.09 +- 0.06 | 0, 0, 89, 0, 88 |
| slow4 (x0.1) | 0.70 0.70 1.00 0.71 1.00 | 0.824 | +0.08 +- 0.06 | 0, 0, 89, 0, 79 |

* Seed 4 (the one that broke): `freeze4` gives ACC 1.00 with 88 pure epochs and no break (ref.: 28 pure epochs, 11 breaks). The replay is identical to the reference up to epoch 13 (L_psi, L_neg, KL, m_std), so the improvement is caused by the freeze alone. `slow4`: 79 pure epochs, 2 breaks. `rec1`: 88 pure epochs, but `rec1` degrades seed 2, pure in the reference (ACC 0.60, no pure epoch), because the recompute at each iteration modifies the formation; `rec5` does not help seed 4 (0 pure epochs). The two recompute options therefore have a net effect of zero.
* Seed 2 (control): freeze and slowdown change nothing (89 pure epochs). Seeds 0, 1, 3 (never pure in the reference): no variant produces a pure phase (final ACC 0.70 to 0.80; the gain of seed 3, 0.52 -> 0.73 with freeze, does not bring it to the pure regime).
* The mean gains of `freeze4` and `slow4` (+0.09 and +0.08, about 1.5 standard errors) are not significant at 5 seeds; what is established is the mechanism on seed 4.

**Reading.** The late instability of seed 4 is indeed a staleness problem of the stored intents: freezing m (or almost) removes it with no effect on an already stable seed. But it concerns only one seed out of five. The dominant problem remains the merge at formation (seeds 0, 1, 3; 5, 7, 8 of the 10 seeds), which neither the memory, nor the recompute, nor the freeze, nor `rho`, nor oracle experts have fixed. The freeze is a calendar rule (4 epochs after the first placements), valid on the toy where the representation is finished at epoch 12 (centroid accuracy = 1.00); on real data, a label-free criterion would be needed to decide the moment (for example the stability of m between epochs). Consequence: `freeze_repr_after` stays disabled by default.
Proposed next step: address the formation, where the defect lies in the first epochs of actions (number of opened clusters and initial merges), with the hypothesis of the absence of repulsion of `L_psi` between classes as a lead (measure the median distance between classes from epoch 10 on the seeds that merge).

### 5.7 Formation: geometry of m at epochs 0-14, then resampling of the policy from epoch 10
Replay of the 10 seeds of `toy_rho/ref` (1 thread, identical to the suites); labels for evaluation only. Measurement scripts outside the repo.

**Geometry of m at formation.** The hypothesis "`L_psi` has no repulsion between classes, so m is poorly separated when the learner starts to act" is ruled out for the formation merges: the seeds stuck at ACC 0.5 to 0.7 (0, 5, 7) have the BEST geometry at epoch 10 (minimum distance between class means 1.0 to 1.3; minimum separation / intra-class dispersion 150 to 250), while the pure seeds (2, 9) have 0.4 to 0.6 (60 to 130). The collapse of the median distance of seeds 3 and 4 (1.45 and 1.18 at epoch 5, against 4.5 to 7.1) is in place before the learner places an image: it comes from the experts-only phase (and it remains associated with the late instability, see 5.6).

**Resampling from epoch 10.** We train to the end of epoch 9 (the learner has not placed anything yet), copy the whole state (networks, optimizers, experts, memory), then continue for 8 epochs, 8 times, changing only the random generator of torch (draw of the placements, PPO minibatches, expert episodes, views). Number of runs with ACC >= 0.9 at epoch 17:

| seed (fate in the reference) | pure at epoch 17 / 8 | mean ACC | ACC min-max |
|---|---|---|---|
| 0 (stuck) | 1 | 0.69 | 0.5-1.0 |
| 1 (stuck) | 0 | 0.71 | 0.7-0.7 |
| 5 (stuck) | 3 | 0.84 | 0.7-1.0 |
| 7 (stuck) | 2 | 0.69 | 0.5-1.0 |
| 2 (pure) | 3 | 0.80 | 0.5-1.0 |
| 9 (pure) | 4 | 0.86 | 0.6-1.0 |
| 4 (breaks at epoch 25) | 5 | 0.90 | 0.7-1.0 |
| 3 (unstable) | 2 | 0.78 | 0.4-1.0 |
| total | 20 / 64 | | |

Reading: with an identical representation, the fate at formation is largely a lottery of the policy phase. The "pure" seeds are pure in only 3 or 4 resamplings out of 8, and three of the four "stuck" seeds become pure in at least one resampling. The 5 slots are always opened (5 clusters at epoch 17 in 63 resamplings out of 64, 4 in one): the defect is not a lack of clusters but mixed slots. The representation (m) and the reward are therefore ready at epoch 10 (see 5.1 and 5.3, ideal memory at 100 %); what decides is the way an untrained policy fills the memory during its first epochs. Limits: the state at epoch 17 is not the final state (a pure resampling can break later, like seed 4); 8 resamplings per seed; a single configuration of the toy.
Lead, untested: train the policy before it writes into the memory (for example, label-free, bring it closer to softmax(reward) on states of expert episodes, or have it play several epochs without a persistent memory), so that the first placements do not fill the memory at random. To validate on the same resamplings (8 per seed, epoch 10) before any 100-epoch run.

### 5.8 Formation: "teacher" phase of the policy and discarded memory (resamplings from epoch 10)
Same protocol as 5.7: state copied at the end of epoch 9, then resamplings that differ only by chance (6 resamplings per seed and per variant, 8 seeds; reference: 8 resamplings per seed of 5.7). Measurement: ACC >= 0.9 at the end of 8 epochs of persistent placements (epoch 17 for A and the reference; B counts 3 more epochs, discarded). Labels for evaluation only.

Variants (options disabled by default, default path unchanged bit for bit, 117 tests):
* **A, `policy_teacher_iters=30`**: during the first 30 placement iterations (2 epochs), the learner samples its actions from softmax(reward) on its own memory, and the policy is distilled towards softmax(reward) on these states (KL, same number of steps as PPO; the critic is fitted to the returns); afterwards the policy acts alone, with PPO. Chosen rather than a distillation on states of expert episodes so that the grouping of the expert does not enter the policy (principle: the expert does not parametrize the learner; here the policy learns only from the reward, as before).
* **B, `memory_discard_epochs=3`**: the memory written during the first 3 placement epochs is discarded at the end of each one; the policy keeps what it has learned.

| seed | reference (pure / 8) | A (pure / 6) | B (pure / 6) |
|---|---|---|---|
| 0 | 1 | 6 | 0 |
| 1 | 0 | 4 | 2 |
| 2 | 3 | 5 | 2 |
| 3 | 2 | 0 | 1 |
| 4 | 5 | 5 | 3 |
| 5 | 3 | 5 | 0 |
| 7 | 2 | 6 | 0 |
| 9 | 4 | 4 | 0 |
| total | 20 / 64 (31 %) | 35 / 48 (73 %) | 8 / 48 (17 %) |

* **A divides the formation failure rate by more than two** (from 69 % to 27 %) and makes pure, from epoch 10, seeds that almost never were (0: 6/6, 7: 6/6, 1: 4/6). This supports the explanation by the first placements of an untrained policy: giving it the reward as a guide is enough to improve the formation strongly, without touching the representation, the reward or the structure of the memory.
* **B does not help** (17 %, less than the reference): discarding the memory lets the policy learn on contaminated states.
* **Seed 3 is not repaired** (0/6): it is the seed with a collapsed median distance (5.7), where the geometry of m, not the policy, limits the result.
* Limits: state at epoch 17 only (a pure phase can break later, like seed 4 in 5.5); 6 resamplings per seed; no 100-epoch run yet; 30 iterations is a value chosen without tuning.
Proposed follow-up: `configs/suites/toy_teacher.yml` (5 seeds, 100 epochs: reference, `teacher30`, `teacher30_freeze4`), to see whether the pure phases obtained hold up to epoch 100 and whether the freeze of 5.6 removes the late breaks.

### 5.9 Why the reward does not open: analysis of `L_pos` and `L_neg`, then test of `free_debias_scale`
**Analysis (formulas, `cibirl/losses.py`).** Notation: $V$ = valid actions (occupied slots + a single empty slot); $s_k$ = reward score for $k$; $q(k)=e^{s_k}/\sum_{j\in V}e^{s_j}$; $k^+$ = slot of the twin (another view of the same image); $\tau^+=1/K$.
* `L_pos` (forced moves): $L_{pos}=-\log q(k^+)$, hence $\partial L_{pos}/\partial s_{\text{open}}=q(\text{open})\ge 0$: at each forced move the score of opening is pushed down; nothing in `L_pos` raises it (no positive term on the free moves). Measurement: $q(\text{open})\approx 0.04$ on the forced moves.
* `L_neg` (free moves): $\ell(u,v)=-\log(1-q(k(v)))$, $L_{neg}(u)=\max\{(\bar\ell(u)-\tau^+\ell^+)/(1-\tau^+),0\}$ with $\ell^+=\mathbb{E}_{\text{forced}}[-\log(1-q(k^+))]$. The gradient is zero as soon as $\bar\ell\le\tau^+\ell^+$, i.e. (a single wrong slot) $q\le 1-e^{-\tau^+\ell^+}$: a dead zone. $\ell^+$ is measured on twins (easy cases) whereas the hidden false negatives are images of the same class but different (harder): the correction subtracts too much, and all the more so as the model is good on the twins. Measurements (epoch 20, seeds 0, 2, 4): $\ell^+=3.0$ to $5.1$; zero repulsion on 72 to 82 % of the free moves; at epoch 9, 55 to 72 % (mean 68 %).
* The other losses (`L_psi`, KL, `L_inv`, entropy bonus) do not push towards fewer clusters; `L_psi` packs the classes in m (5.7) without merging them; `rho` is the only anti-merge term and it is disabled (5.4).
* What the learned reward says (epoch 20, states rebuilt with the true classes): if a class is missing from the memory, mean $q(\text{open})$ = 0.54 / 0.33 / 0.30 (seeds 2 / 4 / 0), ideal 1; with a slot mixing two classes, the reward prefers the mixture to opening in 75 / 90 / 95 % of the pairs. Caveat: these states have 4 occupied slots, outside the distribution of the expert episodes (at most 2 occupied slots when one has to open with 3 orbits).

**Test: option `free_debias_scale`** ($\tau^+\to s\,\tau^+$ on the free moves only; 1 = unchanged, bit for bit; 0 = no more correction), applied from epoch 0. Protocol of 5.7: 10 epochs, reward probe, copy, 6 resamplings of 8 epochs per seed (8 seeds), only chance changes. Reference: 20/64 pure resamplings (31 %), mean final ACC 0.78.

| | reference ($s=1$) | $s=0.5$ | $s=0$ |
|---|---|---|---|
| free moves without repulsion gradient (epoch 9) | 68 % | 54 to 68 % | 0 to 9 % |
| $q(\text{open})$ when a class is missing (epoch 9, mean of the 8 seeds) | 0.38 | 0.46 | 0.53 |
| pure resamplings at epoch 17 | 20/64 (31 %) | 16/48 (33 %) | 18/48 (38 %) |
| mean final ACC | 0.78 | 0.79 | 0.83 |

* The dead zone disappears with $s=0$ and the reward opens better in the 8 seeds ($q(\text{open})$ rises from 0.38 to 0.53 on average), so the mechanism described does exist.
* But the rate of pure formation does not change in a distinguishable way (38 % against 31 %, a difference of the order of the noise at this number of resamplings), and the reward still opens in only about half of the cases. The dead zone is therefore a real factor but not the sufficient cause. Comparison: the "teacher" phase of 5.8 gives 73 %.
* Next lead, untested: the expert episodes never expose more than 2 occupied slots at the moment of opening (3 orbits), whereas the policy has to open with 4 occupied slots; combine `free_debias_scale=0` with more orbits per episode (`orbits_per_episode`), or add a positive term for opening.

### 5.10 Two tests: repulsion floor with $\rho$, and 5 orbits per expert episode
Protocol of 5.9: option applied from epoch 0, 10 epochs, reward probe, copy, 6 resamplings of 8 epochs per seed (8 seeds), only chance changes. References: 20/64 pure resamplings at epoch 17 (31 %), mean final ACC 0.78, $q(\text{open})$ when a class is missing = 0.38.

* **Test 1, `rho=0.1` + `neg_floor=0.1`.** A constant added to $L_{neg}$ has no gradient; the floor is therefore placed inside the max: $L_{neg}(u)=\max\{\frac{\bar\ell(u)-\tau^+\ell^+}{1-\tau^+},\ \varepsilon\,\bar\ell(u)\}$ with $\varepsilon=0.1$, where $\bar\ell(u)$ is the uncorrected mean repulsion ($\ell(u,v)=-\log(1-q(k(v)))$ averaged over the views $v$ of other orbits). Thus a share $\varepsilon$ of the plain repulsion always remains, including in the dead zone.
* **Test 2, `orbits_per_episode=5`** (instead of 3): the expert opens with up to 4 occupied slots.

| | reference | test 1 | test 2 |
|---|---|---|---|
| pure resamplings at epoch 17 | 20/64 (31 %) | 14/48 (29 %) | 4/48 (8 %) |
| mean final ACC | 0.78 | 0.80 | 0.73 |
| $q(\text{open})$ if a class is missing (epoch 9) | 0.38 | 0.42 | 0.79 |

* **Test 1: no effect** on the formation (29 % against 31 %, ACC 0.80 against 0.78) nor on the reward probe (0.42 against 0.38). The floor and $\rho$ are not enough to change the behaviour.
* **Test 2: the reward does learn to open** (0.79 against 0.38, up in the 8 seeds, from 0.70 to 0.90), which supports the hypothesis that the lack of examples of opening with several occupied slots (at most 2 with 3 orbits) contributes to the defect measured in 5.9; **but the formation is worse** (8 % pure resamplings, ACC 0.73). The partitions obtained are still wrong in both ways: seed 2, two classes merged in one cluster while a cluster remains empty; seed 7, one class split in two clusters (130 + 170) and two others merged. Opening more often therefore did not remove the merges and added splits; with 5 orbits on 5 classes, the probability that two orbits of an episode are of the same class is higher (more false negatives on the expert side), which may explain part of the splits (untested).
* No 100-epoch run was launched for these two variants (the formation, the first condition of stability, is not better); the suite `configs/suites/toy_open.yml` is ready if one wants to measure the stability anyway.
* Summary of 5.8 to 5.10: the "teacher" phase of the policy (73 %) remains the only intervention that clearly improves the formation; the corrections of the reward ($s=0$, floor, 5 orbits) improve the reward probe (especially 5 orbits) without improving the formation.

### 5.11 Change of context: MNIST, digits 1 to 6 (2026-10-14)
To know whether the failure of the toy (class merges) reproduces on well-distinguishable classes, a single run, fixed before the result: `data=mnist`, digits 1 to 6, 300 images per digit drawn at random with the seed (1800 images), bank `digits` (crop + rotation $\pm 15^\circ$ + brightness / contrast, no flip), $K=6$, $B=5$ orbits, $n=2$ views, 2000 expert episodes, toy settings for the rest (`configs/mnist.yml`, suite `configs/suites/mnist_digits.yml`). 3 seeds, stop at convergence (label-free rule) or at 200 epochs.

| seed | epochs | converged | ACC | NMI | clusters | kNN(10) pixels | kNN(10) $z$ | kNN(10) $m$ | k-means $m$ | recovering the twin through $m$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 200 | no | 0.43 | 0.33 | 5 | 0.93 | 0.78 | 0.68 | 0.33 | 8 % |
| 1 | 200 | no | 0.41 | 0.31 | 6 | 0.94 | 0.86 | 0.69 | 0.33 | 3 % |
| 2 | 200 | no | 0.42 | 0.32 | 5 | 0.94 | 0.80 | 0.65 | 0.33 | 3 % |

("kNN": nearest neighbours over the 1800 images, labels used for evaluation only; $z$ = encoder code, $m$ = intent; k-means on the pixels: ACC 0.67 to 0.76.)

* **No seed converges**: ACC between 0.25 and 0.51 from epoch 15 to epoch 200 (around 0.4), and on average 34 to 37 % of the images change cluster from one epoch to the next (from epoch 40 to 200).
* **The digit 1 is isolated** (≈ 290 out of 300 in one cluster, in the 3 seeds); the digits 2 to 6 are spread over 3 to 4 clusters, with no digit-cluster correspondence.
* **It is not the same picture as the toy**: on the toy the representation $m$ was perfect (closest class mean: 1.0) and the failure came from the partition. Here $m$ is already bad before the policy (kNN 0.65 to 0.69 against 0.93 for the pixels; $z$ keeps more: 0.78 to 0.86; the twin of a view is recovered through $m$ in 3 to 8 % of the cases).
* The "twin" accuracy of the reward is ≈ 0.7 at the start, then drops to 0.4–0.5 on seeds 1 and 2 (stays ≈ 0.7 on seed 0); the KL goes from about 900 to 170–540.
* Untested (hypotheses): compression too strong ($\beta$), expert task too hard with $B=5$ (with 6 classes, two orbits of an episode are of the same class in ≈ 91 % of the cases but counted as different), transformation bank. A single setting was tried, with no adjustment.

### 5.12 MNIST: where the information is lost (probe of $z$ and $m$ during training, 2026-10-15)
Same run as 5.11 (same seeds, same config), probe `scripts/probe_representation.py` at epochs 0, 2, 4, 6, 7, 8, 9, 11, 14, 19, 24, 29, 39, 49 (the probe does not disturb the training: at epoch 2 of seed 0 the figures are identical to the full run). Results in `results/mnist_digits/probe/seed{0,1,2}.jsonl`, figure `results/mnist_digits/probe/representation.png`. The learner starts to place at epoch 8: before, only the expert phase acts.

* **From the expert phase on, $m$ is bad and does not improve.** kNN(10) on $m$: 0.71 to 0.78 at epoch 0 (random encoder), then 0.53 to 0.71 at epoch 9, and 0.65 to 0.68 at epoch 49; the raw pixels give 0.93. Training on the experts did not improve it with respect to the initialization. The k-means on $m$ stays between 0.24 and 0.44 over the whole period (k-means on the pixels: 0.67 to 0.76).
* $z$ keeps more class information than $m$ (kNN 0.71 to 0.88, about 0.75 at epoch 49): part of the loss is between $z$ and $m$ (the intent, trained by $L_\psi$ alone).
* The KL of the encoder goes from ≈ 70 (epoch 0) to 700–960 from epoch 2 and then rises to 1300 (seed 1): the code carries many bits. $L_\psi$ goes down to $-32$ to $-41$ at epoch 8 and then to $-43$ to $-45$ at epoch 29, which corresponds (unverified hypothesis) to a variance of $q_\psi$ stuck at its floor rather than to a true orbit coherence.
* Recovering the twin of a view by nearest neighbour in $m$: only 3 to 7 % (chance: 0.06 %).
* The "twin" accuracy of the reward is ≈ 0.7 from epoch 4 and hardly evolves during the expert phase (0.65 to 0.77 from epoch 2 to 9); its drop on seeds 1 and 2 appears later, during the learner phase.
* **Reading.** The MNIST problem lies upstream of the policy and the memory: at epoch 9, before any placement, $m$ already does not separate the digits. The signal "distinguish different images" seems to be learned at the instance level (the KL is high) rather than at the class level; the test of this reading (for example, a larger $\beta$, or 3 orbits) has not been done.

### 5.13 MNIST: neither compression nor the number of orbits explains the bad representation (2026-10-15)
Three variants, applied from epoch 0, probed by `scripts/probe_representation.py` (3 seeds, up to epoch 19): `beta=0.001` (instead of $10^{-4}$) with a ramp of $\beta$ shortened to 60 iterations (`beta_warmup_iters=60`, otherwise $\beta$ reaches its value only at epoch ≈ 16), `orbits_per_episode=3` (instead of 5), and both together. Results in `results/mnist_digits/probe_beta1e-3/`, `probe_orbits3/`, `probe_beta1e-3_orbits3/`; mean over the 3 seeds [min–max].

| epoch 19 | kNN(10) on $m$ | k-means on $m$ | kNN(10) on $z$ | KL | "twin" accuracy | policy ACC |
|---|---|---|---|---|---|---|
| reference ($\beta=10^{-4}$, 5 orbits) | 0.62 [0.60–0.65] | 0.29 [0.24–0.35] | 0.76 | 936 | 0.72 | 0.41 [0.37–0.44] |
| $\beta=10^{-3}$ | 0.63 [0.59–0.65] | 0.33 [0.29–0.37] | 0.74 | 552 | 0.72 | 0.38 [0.36–0.40] |
| 3 orbits | 0.64 [0.63–0.64] | 0.33 [0.30–0.37] | 0.76 | 761 | 0.81 | 0.35 [0.30–0.42] |
| $\beta=10^{-3}$ + 3 orbits | 0.65 [0.53–0.75] | 0.29 [0.24–0.38] | 0.78 | 539 | 0.81 | 0.40 [0.39–0.41] |

(Pixels: kNN 0.93, k-means 0.67–0.76; random encoder: kNN on $m$ 0.71–0.78.) At epoch 9 the differences are of the same order (kNN on $m$: 0.62 to 0.67 for the four).

* **None of the variants improves $m$**: the differences between variants are within the dispersion of the seeds. Reducing the KL from 936 to about 540 does not change the class structure of $m$; reducing the number of orbits increases the "twin" accuracy (0.72 → 0.81, which is expected: fewer slots to separate) with no effect on $m$ or on the ACC.
* The hypotheses "compression too strong" and "too many false negatives of the expert with 5 orbits" are therefore not supported by these tests (for these values, 20 epochs, 3 seeds).
* Still untested: the strength of the transformation bank (hypothesis 3 of the user), the architecture (the intent $m$ is trained only by $L_\psi$), the size of the expert part (30 % of 1800 images).

### 5.14 MNIST: the transformation bank (hypothesis 3), in pixels and with a stronger bank (2026-10-15)
**Geometry of the bank in pixels** (`scripts/probe_bank.py`, seed 0, 1800 images; two views per image, the bank applied 1 or 2 times as in an orbit; median Euclidean distances between pixels; results in `results/mnist_digits/bank_geometry_seed0.json`):

| bank | applications | distance between 2 views of an image | distance between 2 images of the same class | distance between 2 images of different classes | twin = closest view | closest view of the same class |
|---|---|---|---|---|---|---|
| `digits` | 1 | 6.84 | 9.02 | 9.87 | 19 % | 92 % |
| `digits` | 2 | 7.86 | 9.23 | 9.89 | 8 % | 87 % |
| `digits_strong` | 1 | 9.24 | 10.15 | 10.61 | 5 % | 80 % |
| `digits_strong` | 2 | 10.14 | 10.74 | 11.04 | 2 % | 67 % |

* With `digits`, two views of the same image are not much closer (6.8) than two images of the same class (9.0): in pixels, the instance is not trivial to recognize (19 %; 8 % with 2 applications), whereas the class is more so (closest view of the same class: 92 %). The hypothesis "the orbits are resolved too easily at the instance level" is not supported **in pixels** (it remains possible in the learned space).
* **Stronger bank** (`configs/transforms/digits_strong.yml`: crop $\geq 0.5$, rotation $\pm 30^\circ$, jitter 0.3; fixed before the result, a sensitivity step of `digits`), probed as in 5.13 (3 seeds, epoch 19, folder `results/mnist_digits/probe_digits_strong/`):

| epoch 19 | kNN(10) on $m$ | k-means on $m$ | kNN(10) on $z$ | KL | "twin" accuracy | recovering the twin through $m$ | policy ACC |
|---|---|---|---|---|---|---|---|
| `digits` | 0.62 [0.60–0.65] | 0.29 | 0.76 | 936 | 0.72 | 4 % | 0.41 [0.37–0.44] |
| `digits_strong` | 0.56 [0.51–0.61] | 0.30 | 0.66 | 585 | 0.58 | 1 % | 0.38 [0.36–0.40] |

* **A stronger bank degrades $m$ and $z$ instead of improving them** (kNN on $m$ 0.56 against 0.62; on $z$ 0.66 against 0.76) and makes the expert task harder ("twin" accuracy 0.58 against 0.72). The policy ACC does not change (0.38 against 0.41, within the dispersion).
* Partial conclusion: for this direction of variation (stronger bank) hypothesis 3 is not supported. A **weaker** bank has not been tested.

### 5.15 MNIST: the intent $m$ is of rank 1, fixed variance of $L_\psi$ (2026-10-15)
**Extended probe** (`scripts/probe_representation.py`, 3 seeds, epochs 0 to 29, `results/mnist_digits/probe_intent/`):

* **$m$ is of rank 1 from epoch 2**: the first principal axis of $m$ carries 99.4 to 99.5 % of the variance at epoch 2 and 99.9 to 100 % from epoch 9 (participation ratio 1.00; already 93 to 96 % at epoch 0, with an almost random encoder). The 16 dimensions are copies of one and the same variable: the "variance 1 per dimension" constraint of the BatchNorm is satisfied without carrying anything more. At epoch 200, $m$ is almost binary: ≈ 290 images (the digit 1) on one side, ≈ 1500 on the other, which explains why only the 1 is isolated in 5.11. **The toy runs also have an $m$ of rank 1** (`ref_seed0` to `4`, participation ratio 1.00): a single axis is enough to carry 5 classes that are worth one value each; it is therefore not a toy / MNIST difference, but a limit that becomes blocking as soon as the classes have an internal variability.
* **The variance predicted by the intent reaches its floor** ($\log\sigma^2=-6$): share of (image, dimension) pairs at the floor 0 % at epoch 2, 0 to 67 % at epoch 9, 66 to 84 % at epoch 19, 83 to 95 % at epoch 29 (depending on the seed). $L_\psi$ is then ≈ $-40$ to $-45$ (minimum $-48$): the $\log\sigma^2$ term dominates. The coherence of the views is real (RMS error per dimension between two views 0.002 to 0.023 at epoch 200, for a standard deviation of 1.0 between images); in training as in inference, the batch statistic or the running statistic of the BatchNorm gives the same figures.
* **No memorization of the expert images**: $m$ is worth as much on the 30 % of images that build the experts as on as many images drawn among the others (mean gap over epochs ≥ 4: kNN $-0.001$, twin recovered $0.000$, standard deviation 0.02).

**"Fixed variance" test** (`psi_fixed_logvar=0.0`: $L_\psi=\tfrac12\sum_j (m^*_j-\mu_j)^2$ instead of the term with predicted variance; diagnostic option, `None` by default; 3 seeds, `results/mnist_digits/probe_psi_fixed/`), mean [min–max]:

| epoch 19 | kNN(10) on $m$ | k-means on $m$ | kNN(10) on $z$ | first axis of $m$ | KL | policy ACC |
|---|---|---|---|---|---|---|
| reference | 0.62 [0.60–0.65] | 0.29 | 0.76 | 100 % | 936 | 0.41 [0.37–0.44] |
| fixed variance | 0.71 [0.69–0.73] | 0.37 | 0.85 | 100 % | 94 | 0.42 [0.41–0.43] |

* The fixed variance **improves $z$ and $m$** (kNN on $z$ 0.85 against 0.76; on $m$ 0.71 against 0.62, i.e. the level of the random encoder at epoch 0) and divides the KL by ten: the learned variance pushed the encoder to carry many bits to make the views coherent. **It changes neither the rank 1 of $m$ nor the policy ACC** (0.42 against 0.41).
* Partial conclusion: the learned variance of $L_\psi$ degrades the representation, but the bottleneck that bounds the clustering is the rank 1 of $m$; with a single direction, the simplest and most invariant distinction (the 1 against the rest) wins. Untested lead: impose decorrelated dimensions (whitening or covariance penalty instead of the per-dimension standardization alone).

### 5.16 MNIST: decorrelating the dimensions of $m$ (2026-10-15)
Targeted cause: in 5.15, $m$ is of rank 1 (the BatchNorm standardization gives each dimension a variance 1, but they can all be copies of one variable). Two options (`intent_decorrelate`, default `none` = as in the README; both are implementation choices outside the README, like the BatchNorm itself):

* `cov`: BatchNorm kept, plus the penalty $\lambda_{\text{cov}}\,\frac{1}{d}\sum_{i\neq j} C_{ij}^2$ on $m$, where $C$ is the covariance of $m$ over the batch, $d=16$ the number of dimensions and $\lambda_{\text{cov}}=10$ (`intent_cov_weight`, a single value, not tuned). It is worth $d-1$ if all dimensions are copies, 0 if they are decorrelated.
* `whiten`: Cholesky whitening of the batch instead of the BatchNorm (running statistics at inference).

Each alone and with the fixed variance of 5.15; 3 seeds, epochs 0 to 19, folders `results/mnist_digits/probe_dec_*/`; mean [min–max].

| epoch 19 | kNN(10) on $m$ | k-means on $m$ | kNN(10) on $z$ | participation ratio of $m$ | twin recovered through $m$ | "twin" accuracy | policy ACC |
|---|---|---|---|---|---|---|---|
| reference | 0.62 [0.60–0.65] | 0.29 | 0.76 | 1.0 | 4 % | 0.72 | 0.41 [0.37–0.44] |
| fixed variance (5.15) | 0.71 | 0.37 | 0.85 | 1.0 | 5 % | 0.70 | 0.42 |
| **`cov`** | **0.98 [0.98–0.98]** | **0.95 [0.93–0.97]** | **0.98** | **13.6 [12.9–14.4]** | **51 %** | **0.98** | **0.80 [0.71–0.92]** |
| `cov` + fixed variance | 0.95 [0.95–0.96] | 0.70 [0.60–0.79] | 0.96 | 14.8 | 22 % | 0.91 | 0.61 [0.45–0.83] |
| `whiten` | 0.41 [0.39–0.42] | 0.41 | 0.71 | 1.4 [1.0–2.3] | 1 % | 0.58 [0.21–0.77] | 0.38 [0.33–0.42] |
| `whiten` + fixed variance | 0.55 [0.50–0.59] | 0.36 | 0.72 | 5.1 [4.1–6.4] | 3 % | 0.28 | 0.30 [0.25–0.39] |

(Pixels: kNN 0.93, k-means 0.67–0.76. At epoch 9, `cov`: kNN on $m$ 0.96, ACC 0.61 [0.58–0.64].)

* **The covariance penalty alone lifts the bottleneck**: the rank of $m$ goes from 1 to about 14 out of 16, $m$ and $z$ separate the digits almost perfectly (kNN 0.98, above the pixels at 0.93) and the policy ACC at epoch 19 goes from 0.41 to 0.80 (one seed at 0.92, one at 0.71). The cause of the rank 1 of 5.15 is therefore supported causally by this test.
* **With the fixed variance in addition, it is worse** than the penalty alone (ACC 0.61, twin recovered 22 % against 51 %): without the learned variance term, the coherence of the views is less strongly imposed.
* **The Cholesky whitening fails** (rank 1.4 to 5, "twin" accuracy unstable from 0.21 to 0.77): it does not replace the penalty (not studied further: $\varepsilon=10^{-4}$ of regularization, a single setting).
* Limits: 20 epochs, 3 seeds, a single value of $\lambda_{\text{cov}}$, a single dataset; nothing is said yet about convergence or stability (full run launched, next section).

### 5.17 Back to the toy: covariance penalty + $\rho$ + repulsion floor (2026-10-15)
Question asked after 5.15-5.16: if we act "as if there had never been a problem" (intent $m$ corrected), do the three choices judged justified (covariance penalty on $m$, weight 10; $\rho=0.1$; `neg_floor=0.1`) remove the merges of the toy? Protocol fixed before the result: suite `configs/suites/toy_cov.yml`, toy (`configs/toy.yml`, bank `toy_oracle`), the 6 seeds of `toy_rho/ref` that had merged (0, 1, 3, 5, 7, 8), stop at convergence (label-free rule of Section 2, `min_epochs` 40) or at 200 epochs. Two conditions: `ref` = old default (replays the degeneracy), `fix` = `intent_decorrelate=cov` + `rho=0.1` + `neg_floor=0.1`.

| | `ref` | `fix` |
|---|---|---|
| final ACC per seed (0, 1, 3, 5, 7, 8) | 0.80 / 0.80 / 0.51 / 0.80 / 0.80 / 0.60 | 1.00 / 1.00 / 1.00 / 0.70 / 1.00 / 1.00 |
| mean ACC $\pm$ standard deviation | 0.72 $\pm$ 0.12 | 0.95 $\pm$ 0.11 |
| clusters used | 4, 4, 4, 4, 4, 3 | 5 in the 6 seeds |
| converged (label-free rule) | 0/6 (200 epochs) | 5/6 (40, 61, 40, 48, 61 epochs) |
| first epoch with ACC $\ge$ 0.95 | 77 to 150 (3 seeds out of 6 never) | 15 to 21 (5 seeds) |
| rank of final $m$ (participation, 1 = rank 1) | 1.0 in the 6 seeds (share of the 1st component = 1.00) | 2.0 to 4.0 (share of the 1st component 0.26 to 0.52) |

* `ref` reproduces the degeneracy (never converged, oscillating ACC, 4 clusters: two merged classes; seed 8: 3 clusters). On the toy too, the final $m$ is exactly of rank 1 under the old default (5.15 had noted it), whereas the penalty gives it several independent dimensions again.
* `fix` fixes 5 seeds out of 6: ACC 1.00, 5 clusters, convergence between 40 and 61 epochs, with no relapse after convergence. Seed 5 does not converge: ACC 0.70 to 0.80 over 200 epochs, 5 clusters but two merged classes and a third one cut in two (156 + 144 images), so the same failure mode, rarer, is not eliminated.
* Limits. (1) The 6 seeds were chosen because they had failed with `ref`: this measures the correction on hard cases, not an overall failure rate; the 4 other seeds of `toy_rho` are not replayed. (2) `fix` groups three changes; this run does not say which one acts (5.4 and 5.10 had shown that $\rho$ and the floor alone changed nothing, which points to the penalty, without proving it). (3) 6 seeds, a single value of $\lambda_{\text{cov}}$ (10, not tuned). (4) `ref` is stopped at 200 epochs here; the figures of 5.1 to 5.10 are at 100 epochs.
* Off-topic for the toy: the full MNIST run of 5.16 (`mnist_digits_cov`) was interrupted around epoch 30 (process killed, no code error); it is therefore not finished and was not relaunched. The partial logs (epochs 28 to 32) gave a monitoring ACC of 0.96 / 0.95 / 0.70 for seeds 0, 1, 2; these are not final results.

### 5.18 Toy: checks of the covariance penalty (2026-10-15)
Three checks requested after 5.17, suite `configs/suites/toy_cov_check.yml`, toy, same setting as `fix` of 5.17 (covariance penalty, `rho=0.1`, `neg_floor=0.1`) unless stated, stop at convergence (label-free rule, `min_epochs` 40) or at **300 epochs**. To allow different seeds per experiment, `scripts/run_suite.py` now accepts a `seeds` key per experiment (test added in `tests/test_suite.py`).

**(a) The 4 other seeds of `toy_rho` (2, 4, 6, 9), condition `fix`.** All 4 converge at ACC 1.00 with 5 clusters, in 43, 46, 50 and 40 epochs. With the 6 seeds of 5.17, `fix` converges at ACC 1.00 on 9 seeds out of 10; seed 5 is the only one that fails.

**(b) Seed 5, continued to 300 epochs.** The run identically reproduces the first 200 epochs of 5.17 (verified: same values), then does not converge: ACC between 0.70 and 0.80, 4 or 5 clusters, about 5 % of the images change cluster at each epoch (epochs 200 to 300 as well as 100 to 200). The partition stays wrong in the same way: classes 1 and 2 always in the same cluster; class 0 is sometimes whole (4 clusters, ACC 0.80, which is the case at epoch 300), sometimes cut in two (5 clusters, ACC 0.70; at epoch 200: 145 + 155 images). Diagnosis on the final model (epoch 200 of the run of 5.17 and epoch 300 of this run, same findings; labels for evaluation only; measurement scripts outside the repo):
* **The representation is not at fault.** $m$ separates the 5 classes perfectly (kNN(10) = 1.00, k-means = 1.00), its rank is that of the seeds that work (participation 3.9 against 4.0) and the classes are very compact in it (distance between centres $\ge 1500$ times the mean intra-class spread). The penalty therefore does not degrade $m$ on this seed; its final value is low ($L_{cov}=0.010$, i.e. $\lambda L_{cov}=0.10$).
* **The error freezes at formation.** Like seed 0, seed 5 hesitates between epochs 10 and 17 (number of clusters 1 to 5, changing partitions); seed 0 gets out of the hesitation (ACC 1.00 at epoch 18, pure partition at epoch 21), seed 5 settles at epoch 17 on 5 wrong clusters (placement error 0.2 with no change afterwards).
* **The reward knows how to open, but not on a mixed slot.** Clean memory (the 4 other classes in 4 slots, summary = true class centre, class $c$ absent, one free slot): the reward chooses "open" for 100 % of the images of each class (mean probability 1.00). In the real state, the images of classes 1 and 2 join the mixed slot (summary = midpoint of two classes) with a probability of 1.00, and even when a slot is freed by counterfactual, the probability of opening stays 0.00; the images of class 0 split 0.50 / 0.50 between the two halves of the memory (the reward is indifferent between them, hence the argmax partition sometimes whole, sometimes cut); the policy copies the reward. A summary that mixes two classes never appears in the expert (an expert slot contains 1 or 2 views of one orbit), the reward is out of distribution there. Reference: under the old default (`ref`), the same probe gives "open" for a single class out of 5 (probability 0.00 for the 4 others).
* **Combinatorial trap.** Here $K=5$ is also the number of classes: a class cut in two occupies the slot that would be missing to separate the two merged classes. Getting out of the state requires moving together the 150 images of one half of class 0, which image-by-image decisions do not do.

**(c) Sensitivity to $\lambda_{cov}$** (seeds 3 and 8, the two where the old default had failed the most, ACC 0.51 and 0.60; and seed 5 for $\lambda \in \{1, 3, 30\}$). Default value 10 (results of 5.17).

| $\lambda_{cov}$ | seed 3 | seed 8 | seed 5 |
|---|---|---|---|
| 0.1 | 1.00 (46 epochs) | 1.00 (49) | not tested |
| 1 | 1.00 (45) | 1.00 (49) | 0.80, 4 clusters, "converged" at epoch 123 |
| 3 | 1.00 (40) | 1.00 (92) | 1.00 (51) |
| 10 | 1.00 (40) | 1.00 (61) | 0.80 (no convergence at 300) |
| 30 | 1.00 (49) | 0.41, 4 clusters, no convergence | 1.00 (57) |
| 100 | 0.25, 2 clusters, no convergence | 0.44, 5 clusters, no convergence | not tested |

* **Useful window: $\lambda$ between 3 and 10 or so.** From $\lambda=0.1$ the rank of $m$ is restored (participation 4.0) and seeds 3 and 8 converge. At $\lambda=3$ the three seeds converge at ACC 1.00. At $\lambda \ge 30$ failures appear (seed 8 at 30, seeds 3 and 8 at 100).
* **Why a $\lambda$ that is too large fails.** At the failures ($\lambda=100$ seeds 3 and 8, $\lambda=30$ seed 8), $m$ still separates the classes (kNN(10) = 1.00) but they are no longer compact (distance between centres 4 to 22 times the intra-class spread, against 400 to 1700 in the runs that converge), the final measured rank is lower (participation 1.1 to 1.7, against 3 to 4) and 40 to 50 % of the images change cluster at each epoch ($\lambda=100$ seeds 3 and 8, $\lambda=30$ seed 8).
* **Seed 5.** Non-monotonic in $\lambda$ (failure at 1 and 10, success at 3 and 30); it is therefore not a "penalty too strong" nor "too weak" effect at $\lambda=10$, but rather the formation trap of point (b), drawn or not depending on the trajectory. At $\lambda=1$ the label-free stopping rule declares convergence on a wrong and stable partition (ACC 0.80, 4 clusters): the rule does not protect against a stable but bad partition.
* Limits: 2 to 3 seeds per value of $\lambda$, a single run per cell, a single value of $\rho$ and of the floor; seed 5 is the only one where $\lambda=3$ and $\lambda=10$ differ, which is not enough to change the default (10); a check of $\lambda=3$ on the other seeds would be necessary to do so.

### 5.19 Toy with more slots than classes: $K=10$ for 5 classes (2026-10-15)
Working decision: $\lambda_{cov}=3$ becomes the standard weight (`TrainConfig.intent_cov_weight` goes from 10 to 3; the penalty stays activated by `intent_decorrelate=cov`), and $K$ must be larger than the true number of classes, so that convergence is not imposed by the number of slots (with $K=5$ for 5 classes, a partition into 5 clusters can be forced by capacity). Suite `configs/suites/toy_cov_k10.yml`: toy, $K=10$, $\lambda_{cov}=3$, `rho=0.1`, `neg_floor=0.1`, the 10 seeds of `toy_rho` (0 to 9), stop at convergence (label-free rule, `min_epochs` 40) or at 300 epochs.

| seed | epochs | converged | final ACC | final clusters | share of the last 100 epochs with ACC $\ge$ 0.95 | clusters used (min-max, last 100 epochs) | images changing cluster per epoch |
|---|---|---|---|---|---|---|---|
| 0 | 300 | no | 1.00 | 5 | 0.22 | 4-10 | 0.086 |
| 1 | 300 | no | 1.00 | 5 | 0.04 | 5-10 | 0.197 |
| 2 | 300 | no | 0.81 | 8 | 0.19 | 4-10 | 0.115 |
| 3 | 300 | no | 0.71 | 9 | 0.23 | 5-10 | 0.153 |
| 4 | 298 | yes | 1.00 | 5 | 0.66 | 5-7 | 0.061 |
| 5 | 300 | no | 0.70 | 10 | 0.04 | 5-10 | 0.268 |
| 6 | 300 | no | 0.80 | 10 | 0.21 | 5-10 | 0.137 |
| 7 | 300 | no | 0.89 | 9 | 0.15 | 3-10 | 0.135 |
| 8 | 300 | no | 0.86 | 9 | 0.13 | 3-10 | 0.207 |
| 9 | 300 | no | 0.83 | 9 | 0.42 | 5-10 | 0.105 |

Mean: ACC 0.86 $\pm$ 0.11, clusters used 7.9 $\pm$ 2.0, **1 seed out of 10 converges** (seed 4, at epoch 298). Reminder: at $K=5$, `fix` converged on 9 seeds out of 10 (5.17 and 5.18, with $\lambda=10$; $\lambda=3$ was tested only at $K=5$ on seeds 3, 5 and 8, all at ACC 1.00).

* **The final partitions are pure, but too fine.** The 10 final partitions have a purity of 1.00 (no class merge, including seed 5 which merged at $K=5$); the failure has changed in nature: the classes are shattered into several clusters (from 0 to 4 classes cut in at least two clusters of more than 10 % of the class, depending on the seed; 0 for seeds 0, 1 and 4 which end at 5 clusters), with a few very small clusters (4 to 30 images). Seeds 0 and 1 end at 5 clusters and ACC 1.00 but only by passing through epoch 300: the figure shows a number of clusters oscillating between 4 and 10 over the whole duration.
* **ACC underestimates the defect.** ACC is a one-to-one matching, it drops when a class is cut; the relevant measure here is the number of clusters and its stability.
* **Reward probe (seeds 0, 3 and 5; labels for evaluation only).** With the 5 classes present (summary = class centre) and 5 free slots, the reward and the policy attach each image to its class slot with a probability of 1.00 and never open (probability of opening 0.00). It is therefore not the final reward that pushes to open when everything is in place. On the other hand, with a class spread over two slots (summaries = means of two halves of the class), the reward gives exactly 0.50 / 0.50: these two summaries are identical since $m$ is almost constant within a class (compactness of 5.18), so the reward cannot prefer one of the slots, and **nothing in the objective pushes to regroup two slots of the same class**. At $K=5$ it is the capacity that forces the regrouping; at $K=10$ nothing does it anymore.
* **Untested hypotheses** for the origin of the splits during training: (1) `rho` (anti-collapse term: entropy of the marginal of the reward over the slots, which rewards spreading over several slots) and `neg_floor` (minimum repulsion) may favour splits when $K$ is large; (2) the exploration by sampling of the policy opens slots in transient states and nothing closes them again. A run with `rho=0` and `neg_floor=0` at $K=10$ would settle (1).
* Limits: one run per seed, a single $K$ (10), a single $\lambda$ (3), no comparison with $K=5$ on the 10 seeds with $\lambda=3$; the convergence criterion (partition stable to within 1 % for 10 epochs) is strict, but no seed other than seed 4 gets close to it (mean partition change between 6 and 27 % per epoch).

### 5.20 $K=10$ without `rho` or repulsion floor (2026-10-15)
Test of hypothesis (1) of 5.19: do `rho` and `neg_floor` favour splits when $K$ exceeds the number of classes? Suite `configs/suites/toy_cov_k10_norho.yml`: identical to `toy_cov_k10` (toy, $K=10$, $\lambda_{cov}=3$, same 10 seeds, stop at convergence or at 300 epochs) with `rho=0` and `neg_floor=0`.

| seed | epochs | converged | final ACC | final clusters | final purity | share of the last 100 epochs with ACC $\ge$ 0.95 | clusters used (min-max) | images changing cluster per epoch |
|---|---|---|---|---|---|---|---|---|
| 0 | 300 | no | 0.85 | 8 | 1.00 | 0.03 | 4-10 | 0.206 |
| 1 | 300 | no | 0.91 | 9 | 1.00 | 0.14 | 4-10 | 0.126 |
| 2 | 300 | no | 0.63 | 10 | 1.00 | 0.20 | 5-10 | 0.173 |
| 3 | 300 | no | 0.75 | 9 | 1.00 | 0.07 | 5-10 | 0.215 |
| 4 | 300 | no | 0.40 | 2 | 0.40 | 0.37 | 2-10 | 0.114 |
| 5 | 300 | no | 0.83 | 10 | 1.00 | 0.10 | 5-10 | 0.169 |
| 6 | 300 | no | 0.60 | 3 | 0.60 | 0.03 | 2-10 | 0.219 |
| 7 | 70 | yes | 1.00 | 5 | 1.00 | 0.86 | 1-5 | 0.026 |
| 8 | 300 | no | 0.71 | 10 | 1.00 | 0.10 | 1-10 | 0.219 |
| 9 | 300 | no | 0.84 | 9 | 1.00 | 0.09 | 5-10 | 0.217 |

Mean: ACC 0.75 $\pm$ 0.17, clusters used 7.5 $\pm$ 2.9, **1 seed out of 10 converges** (seed 7, in 70 epochs). Reference with `rho` and floor (5.19): ACC 0.86 $\pm$ 0.11, clusters 7.9 $\pm$ 2.0, 1 seed out of 10 (seed 4, at epoch 298).

* **`rho` and the floor are not the cause of the splits.** Without them, the behaviour is the same: 9 seeds out of 10 do not converge, the number of clusters rises to 10 over the last 100 epochs, and the 7 final partitions with more than 5 clusters are pure and shattered (no class merge, 2 to 4 classes cut each). The seed that converges is not the same (7 instead of 4); with one run per seed, this change of seed is not interpretable.
* **Removing `rho` and the floor improves nothing and degrades slightly.** The mean ACC drops (0.75 against 0.86), mainly because seeds 4 and 6 end on 2 and 3 clusters with merged classes (purity 0.40 and 0.60); with both terms, no final partition was impure. This is only a snapshot at epoch 300: these two seeds also pass through 10 clusters during the same period (min-max 2-10), so the difference is mostly the dispersion of the number of clusters. Difference over 10 seeds, one run each: a hint, not a proof.
* There remains hypothesis (2) of 5.19: nothing in the objective closes two slots of the same class (the reward does not distinguish them, 5.19), and the policy samples. Untested.

### 5.21 Why does seed 4 converge at $K=10$? Shapes, colours, formation (2026-10-15)
Question: what distinguishes the only converged seed of `toy_cov_k10` (seed 4) from the seeds that do not converge (3 and 5)? Analysis on the existing runs (20 runs of 5.19 and 5.20), without a new run; labels used for evaluation only. Figures: `results/toy_cov_k10/clusters_seed{3,4,5}.png` (galleries of images per cluster) and `results/toy_cov_k10/shapes_by_seed.png` (shapes of the 5 classes per seed).

* **The clusters of seed 4 are 5 pure clusters of 300 images, one per class, with all the backgrounds mixed in each cluster.** In seeds 3 and 5, the classes are cut in sub-clusters that follow the colour or the brightness of the background (the nuisance), not the shape.
* **Colours: no difference.** The background is drawn uniformly in $[0\,;0.6]^3$ for all seeds. Within a class, the mean colour of the background is predicted linearly with $R^2$ from 0.84 to 0.97 from $m$ and about 1.00 from $z$, over the 20 runs, converged or not (seed 4 with `rho`: 0.94). The leakage of the background is general and does not separate the seeds. (At the scale of $10^{-3}$ of the distance between classes: see 5.23.)
* **Shapes: no separating property.** Seed 4 has the whitest shapes (white share 0.50 to 0.62, against 0.25 to 0.69 elsewhere), hence the least background; over 10 seeds, more white goes with fewer clusters (rank correlation $-0.3$ to $-0.7$ depending on the condition), not significant and not causal. The seed that converges without `rho` (seed 7) has ordinary shapes, and seeds 4 and 7 each fail in the other condition.
* **What differs: the formation.** In 19 runs out of 20, the number of clusters exceeds 5 from epochs 10 to 18 and stays above 5 for 72 to 100 % of the epochs (after epoch 40). The only exception is seed 7 without `rho`: 5 clusters from epoch 12, never more, convergence at epoch 70. Seed 4 with `rho` goes through 8-9 clusters, comes back to 5 only at the very end (5 to 7 clusters and ACC $\ge$ 0.95 for 66 % of the last 100 epochs) and converges at epoch 298.
* **Comparison of the 20 runs.**

| condition | converged seeds | clusters (mean of the last 100 epochs, mean over seeds) | share of epochs with more than 5 clusters |
|---|---|---|---|
| with `rho` and floor (5.19) | 1/10 (seed 4) | 7.4 | 89 % |
| without (5.20) | 1/10 (seed 7) | 7.5 | 86 % |

Without `rho`, two seeds (4 and 6) end at 2 and 3 clusters with merged classes; with `rho`, no final partition is impure.
* **Reading.** No property of the data separates the seeds that converge; convergence looks like an accident of formation (the initial placement directly gives the right partition, or gets back to it narrowly). A mechanism that closes duplicates, or that avoids creating them during formation, is missing. Hypothesis (untested) for the late noise in the number of clusters: the argmax flips between indistinguishable duplicate slots.

### 5.22 $K=10$ with ideal expert orbits (oracle, 2026-10-15)
Question: is the imperfection of the expert (two images of the same class in two different orbits, with no attraction signal between them) what prevents convergence at $K=10$? Suite `configs/suites/toy_cov_k10_link.yml`: identical to `toy_cov_k10` ($K=10$, $\lambda_{cov}=3$, `rho=0.1`, `neg_floor=0.1`, 10 seeds, stop at convergence or at 300 epochs) with `expert_oracle_link=1.0`.
* **ORACLE DIAGNOSTIC, never a method**: it uses the true classes to build the expert trajectories (`run.yml`: `uses_labels_for_training: true`). In each episode, all the orbits of the same class form a single group: same slot, forced movement (join the twin's slot) for the following orbits, no repulsion inside a class. The target of $L_\psi$ stays the orbit (only the slot, the movement type and the repulsion switch to the class). The stopping rule stays label-free.

| seed | epochs | converged | final ACC | final clusters | final purity | share of the last 100 epochs with ACC $\ge$ 0.95 | clusters used (min-max) | images changing cluster per epoch |
|---|---|---|---|---|---|---|---|---|
| 0 | 300 | no | 0.92 | 6 | 0.96 | 0.63 | 5-10 | 0.055 |
| 1 | 300 | no | 0.75 | 9 | 1.00 | 0.05 | 5-10 | 0.211 |
| 2 | 300 | no | 0.90 | 8 | 1.00 | 0.00 | 7-10 | 0.235 |
| 3 | 300 | no | 0.79 | 9 | 1.00 | 0.05 | 6-10 | 0.185 |
| 4 | 300 | no | 0.91 | 8 | 1.00 | 0.08 | 5-10 | 0.171 |
| 5 | 300 | no | 0.75 | 9 | 1.00 | 0.06 | 5-10 | 0.247 |
| 6 | 300 | no | 0.89 | 10 | 1.00 | 0.17 | 5-10 | 0.103 |
| 7 | 300 | no | 0.92 | 6 | 1.00 | 0.30 | 5-10 | 0.134 |
| 8 | 300 | no | 0.63 | 10 | 1.00 | 0.08 | 5-10 | 0.239 |
| 9 | 300 | no | 0.90 | 7 | 1.00 | 0.03 | 5-10 | 0.231 |

Mean: ACC 0.84 $\pm$ 0.10, clusters used 8.2 $\pm$ 1.4, **0 seeds out of 10 converge**. Figure: `results/toy_cov_k10_link/summary.png`.

| condition (10 seeds) | converged | final ACC | clusters (last 100 epochs) | epochs with more than 5 clusters | final $L_{pos}$ | final twin_acc |
|---|---|---|---|---|---|---|
| `toy_cov_k10` (with `rho`, floor) | 1 | 0.86 | 7.4 | 89 % | 0.274 | 0.83 |
| `toy_cov_k10_norho` | 1 | 0.75 | 7.5 | 86 % | 0.276 | 0.83 |
| `toy_cov_k10_link` (oracle) | 0 | 0.84 | 8.0 | 94 % | 0.000 | 1.00 |

* **Ideal orbits do not improve convergence**: no seed converges, the number of clusters stays at 8.0 on average (7.4 without the oracle), and 94 % of the epochs have more than 5 clusters. The expert task is solved perfectly ($L_{pos}$ = 0.000, twin_acc = 1.00 against 0.83), with no effect on the learner: the imperfection of the expert is therefore not the cause of the lack of convergence at $K=10$.
* **Reward probe** (clean memory: the 5 classes present in slots 0 to 4 with, as summary, the mean of $m$ over the whole class, over 20 images or over 5 images, slots 5 to 9 empty; 10 seeds of `toy_cov_k10` and of `toy_cov_k10_link`): the probability of joining the slot of its class is 1.00 and that of opening is 0.00, in the 3 variants and in both suites. The final reward does not push to open when the slots are clean, and this holds with or without ideal orbits (complements the probe of 5.19, which had only whole-class summaries).
* **Placement errors (evaluation).** "Wrongly opens" is rare in the three conditions (3.6 to 9.5 per run and per phase). "Wrongly joins" (joining a slot whose dominant class is another) per run, epochs $<30$ then $\ge 30$: `k10` 4,536 then 9,748; `k10_norho` 4,174 then 3,850; `k10_link` 6,210 then 2,022. Mind the definition (`cibirl/diagnostics.py`): opening a slot is judged "wrong" only if a slot with the same dominant class already exists; during formation, when the slots are still mixed, opening a new slot is therefore judged correct even if it creates a duplicate. Consistent with the hypothesis that duplicates are born in this mixed phase and that nothing closes them; not tested directly.
* **What remains is therefore on the learner side**: the formation (5.21) and the absence of a mechanism that regroups two slots of the same class (5.19). Limits: one run per seed; the oracle mixes attraction between orbits and larger expert summaries (as in 5.2); $L_\psi$ stays at the orbit level; the test does not say what a real (imperfect) phase-2 relinker would give.

### 5.23 Intent $m$: the images of a same class are already merged, but two slots of a class are tied (2026-10-16)
Question (follow-up of 5.22): if the intent and the representation are correct, why are images that are supposed to be together not, even with ideal expert orbits? Analysis with no new run, on the already saved files (`intents.npy`, `preds.npy`, `checkpoint.pt`) of runs of `toy_cov` (seeds 3 and 5), `toy_cov_k10` (3, 4, 5) and `toy_cov_k10_link` (2, 5, 7); labels for evaluation only. Figure: `results/toy_cov_k10_link/intent_scales.png`. Reminder: `m_dim` stayed at 16; what $L_{cov}$ changed is not the dimension of $m$ but its effective rank.

* **$m$ is almost a function of the class.** Share of the variance of $m$ not explained by the class ($1-\eta^2$): from $5\times10^{-7}$ to $2\times10^{-5}$; for $z$: from $5\times10^{-6}$ to $3\times10^{-4}$. Distances in $m$: between classes 3.2 to 5.6; between two images of a same class 0.002 to 0.013; between two views of a same image (orbit, redrawn background) 0.003 to 0.019; between the centres of two sub-clusters of a same class $10^{-4}$ to $4\times10^{-3}$. The intra-class separation is of the order of $10^{-3}$ of the between-class separation, and of the same order as the gap between two views of a same image: $m$ is invariant to the background up to noise. The background remains linearly readable (5.21, $R^2$ 0.84 to 0.97), but at this scale of $10^{-3}$; it is not what separates the clusters in the eyes of the learner.
* **What $L_{cov}$ changed.** Without $L_{cov}$ (6 seeds of `toy_cov`, $K=5$), the first eigenvalue of the covariance of $m$ carries 100 % of the variance (rank 1) and the two closest classes are on average at 0.06 (minimum 0.00): the classes are superposed on a line, which explains the earlier merges. With $L_{cov}$: four directions of similar weight (0.30 / 0.26 / 0.23 / 0.21 at $K=10$, $\lambda=3$), the 5 classes form a regular simplex of $\mathbb{R}^4$, minimum distance between two classes 3.98 on average (minimum 2.89; 4.01 and 2.93 with ideal orbits). This is the maximum possible rank for a code of 5 classes. The defect has therefore moved from "superposed classes" to "one class in two slots".
* **Two slots of a same class are exactly tied for the reward** (14 pairs of duplicate slots in 4 runs: `k10` seeds 3 and 5, `k10_link` seeds 2 and 5). The summaries of the two slots differ by $10^{-4}$ to $10^{-3}$ (a single pair at 0.17, seed 3 class 2, with a small slot of 65 images). The reward logit gap between the two slots is 0.00 on average (at most 0.04) with a standard deviation of 0.00 over the images of the class, against a gap of 19 to 70 with the other occupied slots: the reward is saturated for the class and indifferent between the duplicates. The share of images preferring one of the two slots ranges from 0.27 to 0.58 depending on the pair, with no link to the slot they are in: these are ties broken by noise. The sizes of the two slots are close to 50/50 (150/150, 162/138, 152/148, 160/140), which is what a random draw on a tie would give.
* **Reading (not tested directly).** Ideal orbits improve a point that the representation had already solved (separating the classes), and are neutral on what is missing: a state where two slots carry the same class never appears in the expert episodes (an episode memory has one slot per group), so nothing teaches the reward to prefer one of the two. At $K=5$ the capacity forced the closing; at $K=10$ an exact tie has no reason to resolve itself (a slot of 150 images does not empty by random draw). The leads to discuss therefore concern the tie-breaking between slots of a same class, not the expert nor the representation.
* Limits: 8 runs over 3 conditions for the distances of $m$, 4 runs for the reward gaps between duplicates; invariance is measured with a single background draw per image; I did not separate the role of the reward from that of the policy (both are tied according to the probe of 5.19).

### 5.24 CRP prior on placements (2026-10-16)
Follow-up of 5.23: when the reward is tied between two slots of a same class, nothing breaks the tie; `rho` changes nothing there (it acts on the slot indices of the expert states, not on the learner's slots). The policy's implicit prior (uniform over the valid slots, via the entropy bonus) is replaced by an explicit prior on partitions, the Chinese restaurant process (CRP).

* **Definition.** For the image being placed, with $n_k$ the size of slot $k$ without this image, $n$ the number of placed images and $\alpha>0$ the concentration:
$$\pi_0(k\mid s)=\frac{n_k}{n-1+\alpha}\ \text{(occupied slot)},\qquad \pi_0(\text{open}\mid s)=\frac{\alpha}{n-1+\alpha}.$$
A slot whose only member is the image counts as a new cluster (mass $\alpha$). The policy becomes $\pi(a\mid s)\propto\exp(\mathrm{score}_\omega(s,a))\,\pi_0(a\mid s)$: $\log\pi_0$ is a fixed additive offset of the logits, used for sampling, the PPO ratio, the entropy, $L_{inv}$, the teacher phase and `predict` (each image is removed from its own slot there). No label is involved; the prior only reads the slot sizes.
* **Why a logit offset and not a shaped reward.** The policy has no slot-size input (inputs: $z$, $m$, summaries, empty-slot indicators): it could not represent a reward that depends on $\log n_k$, and it cannot undo the offset. Where the reward is indifferent (equal scores), the prior alone decides, in proportion to the sizes; elsewhere the real logit gaps (19 to 70, 5.23) crush $\log n_k$ (about 5).
* **Code.** `cibirl/memory.py`: `crp_log_prior`; `cibirl/policy.py`: `log_probs` and `act` take `log_prior`; `cibirl/train.py`: `TrainConfig.slot_prior` (`uniform` by default = old behaviour, `crp`) and `crp_alpha` (1.0), `Rollout.log_prior`, wired into `learner_rollout`, `policy_step`, `distill_step`, `_invariance` and `predict`. Tests: `tests/test_slot_prior.py` (exact masses, removal of the image, tie-breaking, sampling proportional to the sizes, a re-placement sweep that keeps the proportions instead of going back to 50/50 under a uniform prior, `predict`); 166 tests pass in total. The default is bit-identical to the old code (same hash of the metrics and of the partition on a 4-epoch training with $L_{cov}$, `rho` and floor).
* **Suite `configs/suites/toy_cov_k10_crp.yml`**: identical to `toy_cov_k10` ($K=10$, $\lambda_{cov}=3$, `rho=0.1`, `neg_floor=0.1`, 10 seeds, stop at label-free convergence or at 300 epochs) with `slot_prior=crp` and $\alpha=1$ or $\alpha=10$.

Result of the evaluated partition (policy argmax, prior included): **20 runs out of 20 converge, ACC 1.000, NMI 1.000, 5 clusters**, in 55.4 epochs on average for $\alpha=1$ and 61.1 for $\alpha=10$ (against 1 seed out of 10 in 5.19 and 0 out of 10 in 5.22, at $K=10$). No visible difference between the two $\alpha$ at this level.

But the same run read without the prior at inference, and the state of the memory, give a different picture (`results/toy_cov_k10_crp/summary.png`):

| seed | $\alpha=1$: epochs | clusters without prior | memory slots | classes cut in memory | $\alpha=10$: epochs | clusters without prior | memory slots | classes cut in memory |
|---|---|---|---|---|---|---|---|---|
| 0 | 41 | 8 | 9 | 1 | 50 | 9 | 10 | 2 |
| 1 | 51 | 6 | 6 | 1 | 51 | 6 | 6 | 0 |
| 2 | 54 | 6 | 6 | 1 | 90 | 7 | 9 | 2 |
| 3 | 61 | 6 | 6 | 1 | 60 | 9 | 9 | 2 |
| 4 | 71 | 9 | 9 | 3 | 47 | 5 | 6 | 1 |
| 5 | 53 | 7 | 7 | 0 | 52 | 10 | 10 | 2 |
| 6 | 85 | 7 | 7 | 2 | 91 | 8 | 8 | 2 |
| 7 | 45 | 6 | 7 | 2 | 73 | 6 | 6 | 1 |
| 8 | 46 | 7 | 8 | 3 | 50 | 8 | 10 | 2 |
| 9 | 47 | 6 | 6 | 1 | 47 | 8 | 10 | 3 |

Means ($\alpha=1$ then $\alpha=10$): clusters without prior at inference 6.8 and 7.6 (ACC 0.91 and 0.90); occupied slots in memory at stop 7.1 and 8.4; classes cut in memory (at least two slots of more than 10 % of the class) 1.5 and 1.7, 18 runs out of 20 with at least one cut class; memory purity 0.99 to 1.00.

* **The effect is mostly a tie-break at reading time.** Over the 20 runs, the output is right because the argmax sends each image to the largest of the duplicate slots (maximum a posteriori under the CRP), not because the memory has closed up: without the prior at inference, the same policy gives 5 to 10 clusters (ACC 0.77 to 1.00), as before.
* **The prior nevertheless changes the formation dynamics.** Means over the 10 seeds: $K_{eff}$ of the memory at epochs 20 to 30: 6.3 ($\alpha=1$) and 6.8 ($\alpha=10$) against 8.1 without prior; occupied slots 7.6 and 8.3 against 9.9; images changing slot per epoch (epochs 40 to 50) 0.19 against 0.35. $K_{eff}$ goes through about 4.8 at epoch 12 with the prior (value closest to 5) and then goes back up to 6-7. $\alpha=1$ opens less than $\alpha=10$, as expected.
* **No closing of the duplicates already formed over the observed duration.** $K_{eff}$ goes from 6.27 (epochs 20 to 30) to 6.04 (epochs 40 to 50) for $\alpha=1$, from 6.80 to 6.76 for $\alpha=10$ (8.08 to 8.06 without prior). This is compatible with the expected neutral drift: at equal reward, the size of a slot follows a walk with no drift, with a fixation time of the order of $n/2$ epochs for a slot of $n$ images (diffusion estimate, untested), i.e. an order of magnitude of 150 epochs for 300 images, longer than the duration of these runs.
* **Limit of the stopping rule.** The rule (stable argmax partition, constant number of clusters, loss plateau) only looks at the evaluated partition, which now includes the prior: it stops as soon as `min_epochs` is reached (41 to 91 epochs) without seeing the duplicates of the memory. The declared convergence is therefore that of the output, not that of the memory.
* **Not tested**: the long-run effect on the memory (runs with no early stop up to 300 epochs), a stopping rule that also looks at the memory, the choice of $\alpha$. One run per seed and per $\alpha$.

### 5.25 Memory convergence, CRP drift, exponent $\nu$ and $\alpha=0.1$ (2026-10-16)
Follow-up of 5.24: the output was right (argmax with prior) but the memory kept duplicate slots, and the stopping rule only looked at the output. Three things done together, measured on 5 seeds (0 to 4), $K=10$, $\lambda_{cov}=3$, `rho=0.1`, `neg_floor=0.1`, one run per seed and per condition.

* **Memory convergence criterion** (`cibirl/convergence.py`, option `stop.memory: true`, disabled by default). $\text{mem\_gap}$ = share of the images that do not keep their cluster between the partition of the memory (placements sampled at the end of the epoch) and the output partition, up to renaming (`partition_change`). The run stops when the rule of Section 2 holds and $\text{mem\_gap}\le 0.01$ over the whole window. This is equivalent to "rebuilding the memory from the output no longer changes anything". Two slots of a class that the output merges but that the memory keeps separate give a large $\text{mem\_gap}$. `mem_gap` and `mem_excess` ($=K_{mem}-K_{out}$, numbers of non-empty slots) are logged at every epoch in all cases; `final_metrics.json` contains `output_converged_epoch` (rule without memory) and `memory_converged_epoch`. No label.
* **Exponent $\nu$ on the prior** (`TrainConfig.crp_power`, 1.0 by default = 5.24, bit-identical). The policy becomes $\pi(a\mid s)\propto e^{\,s_\omega(s,a)}\,\pi_0(a\mid s)^{\nu}$, with $s_\omega$ the policy score and $\pi_0$ the CRP prior of 5.24, renormalised. Up to a constant, $\log P(\text{partition})\approx K\log\alpha-n\,H(C)$ ($K$ occupied slots, $n$ images, $H(C)$ entropy of the cluster sizes; for a deterministic assignment $H(C)=I(X;C)$): $\nu$ weights the compression of the cluster variable. At equal scores, the share of a slot evolves in proportion to $(n_a/n_b)^{\nu}$: $\nu=1$ is a martingale (neutral drift), $\nu>1$ is a positive feedback toward the largest slot. Tests: 173 pass in total (7 new: memory criterion, exponent, re-placement sweep with $\nu=1$ which keeps the share and $\nu=2$ which sends it above 0.8).
* **Suite `configs/suites/toy_cov_k10_mem.yml`** (memory stop, 300 epochs at most), three conditions that each change only one thing: `mem_a1` ($\alpha=1$, $\nu=1$: time alone), `mem_nu2` ($\nu=2$), `mem_a01` ($\alpha=0.1$). Figure: `results/toy_cov_k10_mem/summary.png`.

| seed | $\alpha=1$: output / memory (epoch), clusters, ACC | $\nu=2$: output / memory, clusters, ACC | $\alpha=0.1$: output / memory, clusters, ACC |
|---|---|---|---|
| 0 | 41 / 120, 5, 1.0 | 70 / 70, 3, 0.6 | 42 / 42, 5, 1.0 |
| 1 | 51 / 216, 5, 1.0 | 53 / 53, 2, 0.4 | 40 / no (7 slots), 5, 1.0 |
| 2 | 54 / 289, 4, 0.8 | 40 / 42, 5, 1.0 | 70 / 70, 4, 0.8 |
| 3 | 61 / 240, 5, 1.0 | 61 / 61, 4, 0.8 | 62 / 80, 5, 1.0 |
| 4 | 71 / 171, 4, 0.8 | 67 / 76, 4, 0.8 | 43 / 277, 4, 0.8 |

Means: ACC 0.92 ($\alpha=1$), 0.72 ($\nu=2$), 0.92 ($\alpha=0.1$); stop epochs 207, 60, 154; memory converged 5 out of 5, 5 out of 5, 4 out of 5. Clean outputs (5 clusters, ACC 1.0, memory = output): 3 out of 5, 1 out of 5, 2 out of 5.

* **Time alone closes the duplicates ($\alpha=1$, 5 out of 5).** The output converged at 41–71 epochs, the memory at 120–289 (median 216), of the order of the neutral drift estimated in 5.24 (150–200 epochs, diffusion estimate). The memory criterion therefore requires 2 to 5 times more epochs than the output rule (2.4 to 5.4).
* **But the duration exposes another failure: a class cluster absorbed by another.** $\alpha=1$: 2 seeds out of 5 end at 4 clusters (ACC 0.8); seed 2: 5 pure clusters for 200 epochs, then a burst at epoch 202 ($L_{neg}$ 0.10 → 0.40, churn 0.3) and a class absorbed at epoch 208; seed 4: unstable output between 4 and 5 clusters from epoch 38 to 148 (with a clean window from 61 to 96, where the run of 5.24 stopped), then 4 clusters from epoch 149 on. Same number of cases with $\alpha=0.1$ (seeds 2 and 4); $\nu=2$: 4 out of 5. The runs of 5.24 (stopped at 41–91 epochs by the output rule) showed 5 clusters in 20 out of 20: these are the same runs, reproduced bit by bit (identical `L_pos` at each epoch for the 5 common seeds, only the duration changes), observed earlier. The loss of the class happens after their stop in 5.24 (seed 2: epoch 209; seed 4: epoch 149, seed 4 being already unstable before).
* **What precedes the two permanent losses ($\alpha=1$, seeds 2 and 4), read from `metrics.jsonl`.** Seed 2: 5 pure clusters from epoch 36 to 208, then a burst at epochs 203-204 ($L_{pos}$ 0.29 → 0.45; $L_{neg}$ 0.13 → 0.25 → 0.40; churn 0.05 → 0.30; $L_{cov}$ 0.92 → 0.75), the merged slot appears at epoch 209 and stays until epoch 289. Seed 4: at epoch 38 the memory contains a two-class slot (purity 0.80, 20 % of wrong placements) and the output alternates between 4 and 5 clusters; it is clean from 61 to 96 and from 117 to 148, the memory is entirely closed at epochs 146-148 (5 slots, `mem_gap` 0), then a burst (at epochs 147 to 150: $L_{pos}$ 0.45; $L_{neg}$ 0.22 then 0.46; churn 0.57; $K_{eff}$ 2.0) leaves a two-class slot from epoch 149 to the end. Over the 15 runs ($\alpha=1$ of this suite and suite 5.26), 21 bursts of $L_{neg}$ (more than 1.8 times the median of the previous 10 epochs and more than 0.2) occur while the output is clean: 6 are followed by a class loss within 10 epochs, of which 5 heal in 1 epoch; only seed 2 ($\alpha=1$) is followed by a permanent loss (seed 4 has a less clear burst, at the limit of the threshold). Bursts are therefore frequent and usually without consequence; their cause is not identified (no per-epoch checkpoint to test it). The permanent loss requires a burst and a state where the reward does not oppose the merge (below).
* **The bursts coincide with changes of $L_{cov}$ plateau (2026-10-16, analysis of `metrics.jsonl`, no new run).** $L_{cov}$ does not drift: it stays flat and then changes plateau all at once, and the same plateaus come back from one seed to another (about 0.97, 0.73, 0.53, 0.35, 0.2 for seeds 1, 2, 3; 2.65 then 2.1 and 1.6-1.9 for seed 4 at $\alpha=0.1$ and in gate). Definition of a plateau: median of the next 10 epochs differing by more than 15 % from that of the previous 10, after epoch 30. Over the 25 runs of suites 5.25 ($\alpha=1$ and $\alpha=0.1$) and 5.26, 27 plateaus: 18 (67 %) have a burst of $L_{neg}$ within $\pm2$ epochs of the plateau, against 2.3 % (16 out of 702) for 5-epoch windows far from any plateau; 17 of the 32 bursts of $L_{neg}$ have an $L_{cov}$ plateau within 3 epochs. A plateau does not entail the loss: seeds 1 and 3 ($\alpha=1$) have 1 and 3 of them and end clean; over the 27 plateaus, a single permanent loss (seed 2, epoch 209), 6 transient losses. Seed 0: $L_{cov}$ flat (1.25 to 1.31) but the run stops at epoch 120, before any plateau. Seed 4: $L_{cov}$ flat from epoch 40 to 171 (2.62 to 2.77): its loss (epoch 149) has no $L_{cov}$ plateau and is not explained by this. Hypothesis, untested: $L_{cov}$ penalises the off-diagonal covariances in the basis of the coordinates of $m$, which is not rotation-invariant; the class code (a 4-dimensional simplex in 16) reorients by plateaus toward the axes, which moves all the intents and the slot summaries at once and disturbs the reward and the policy.
* **Test of the code orientation around an $L_{cov}$ plateau (2026-10-16, two reruns with a checkpoint per epoch, `--save-epochs`, `scripts/analyze_code_orientation.py`).** The reruns exactly reproduce the original runs (maximum gap 0.0 on all metrics). Seed 2 ($\alpha=1$, permanent loss at epoch 209), epochs 195 to 212; control: seed 3 (clean), $L_{cov}$ plateau at epoch 155 (0.51 → 0.40), epochs 150 to 162. The class centroids use the true classes: this is an analysis, never a training criterion. From $t-1$ to $t$, for the class centroids of $m$: $\Delta D$ = relative variation of the matrix of distances between centroids (rotation-invariant); $\Delta C$ = relative variation of the centred centroids; $\Delta C_R$ = the same after the best orthogonal alignment (Procrustes); $\theta$ = largest principal angle between the 4-dimensional subspaces spanned by the centroids at $t-1$ and at $t$; $S$ = mean displacement of an image in $m$ divided by the mean distance between centroids. **Result.** Outside an event: $\Delta D$ 0.02 to 0.04, $\Delta C$ 0.10 to 0.15, $\theta$ 7° to 15°. At the plateau: seed 2 (epoch 202) $\Delta D=0.069$, $\Delta C=0.386$, $\Delta C_R=0.084$, $\theta=46.1^\circ$, $S=1.02$; seed 3 (epoch 155) $\Delta D=0.063$, $\Delta C=0.342$, $\Delta C_R=0.084$, $\theta=38.6^\circ$. The plateau is therefore a global rotation of the class subspace (centroids ×3, but most of it disappears after alignment) plus a modest deformation (distances between classes +6 to 10 % against ≈3 %); each image moves by about one inter-class distance ($S=1.0$ to $1.2$ at epochs 202 to 204 of seed 2 and 156-157 of seed 3, against 0.1 to 0.5 usually). The share of variance of the class code carried by its 4 most active coordinates rises from 0.42 to 0.47 (seed 2) and from 0.52 to 0.59 (seed 3): partial alignment on the axes, consistent with the non-rotation-invariant $L_{cov}$ penalty. **What the test excludes.** The event is the same in the clean control as in the run that loses a class, with the same amplitude: a rotation, even of 40° to 46°, is not enough to lose a class. A policy frozen at epoch $t-1$ and applied to the representation of epoch $t$ still gives ACC 1.00 in both runs: the rotation does not break the partition by itself. **What differs (a single comparison, hence indicative).** The perturbation lasts ≈6 epochs in seed 2 ($S$ from 0.7 to 1.2 at epochs 202 to 207, $L_{cov}$ stabilised at 0.74 only at epoch 205) against ≈2 in seed 3 ($S\approx1.15$ at epochs 156-157, back to 0.4 from epoch 158). The class is lost at epoch 208-209, 6 to 7 epochs after the plateau. **Hypothesis, untested:** the plateau does not lose the class; it is the duration of the disorder that follows it (reward and policy catching up with the new code) that leaves a pair of classes the time to merge in the memory, a state that is then stable (see below).
* **It is not geometric.** For the 6 merged runs examined (final checkpoints), the distance between the centroids of $m$ of the two merged classes (4.2 to 6.2) is within the range of all pairs of classes (for example seed 2: 4.19 for the merged pair, 4.19 to 4.83 for the 10 pairs; seed 4 at $\alpha=0.1$: 5.83 for the pair, 5.19 to 5.83 for the pairs).
* **The merged state is stable because the reward itself does not object to it.** For the images of the absorbed class, with the output partition as memory: reward logit (open − join the merged slot) between −16.4 and −0.2 in 5 of the 6 runs (−2.2 to −16.4 for $\alpha=1$; −1.7 to −11.3 for $\alpha=0.1$; −1.3 and −0.2 for $\nu=2$ seed 3), against +19 to +70 in favour of the right slot when the slots are pure (5.23). The summary of the merged slot is halfway between the two classes, a state the expert never shows: the reward extrapolates toward "join". To this is added the size bias of the prior, $\nu\log(n/\alpha)$: +6.4 ($\alpha=1$, $n=600$), +8.7 ($\alpha=0.1$), +12.8 ($\nu=2$). This bias grows as $\log n$; changing $\alpha$ by a factor of $10$ only moves it by $\ln 10\approx2.3$. (In the sixth run, $\nu=2$ seed 1, 4 classes in one slot, the reward prefers to open by +1.7 to +3.0 but the bias of +14.2 wins.)
* **$\nu=2$: under-segmentation.** The runs stay at 1 cluster until epochs 25 to 31 (11 to 16 for the other conditions) and then stop at 2 to 5 clusters (ACC 0.40 to 1.00); converged very fast (42–76 epochs) but wrong. The positive feedback is not usable as is.
* **$\alpha=0.1$: faster when it works, not better.** Memory converged at 42, 70, 80 and 277 epochs, one seed not converged at 300 epochs (7 slots for 5 output clusters, $\text{mem\_gap}$ 0.15); 2 absorptions out of 5, like $\alpha=1$.
* **Reading.** The strength of the size preference is not the right lever: too weak and it lets the memory drift (time ≈ 200 epochs), too strong ($\nu=2$) and it freezes an under-segmented partition, and even at $\nu=1$ it does not separate from the reward in the ambiguous regime (mixed summary). The identified defect is the behaviour of the reward on a contaminated slot, outside the distribution of the expert states (always pure slots).
* **Not tested** (leads, not launched): a prior applied only between slots that the reward does not distinguish (scores within $\delta$) rather than everywhere; expert states with a contaminated slot (an orbit of another class merged into the slot, expert action = open), to be discussed since it touches the expert; a stop at the first $\text{mem\_gap}\le0.01$ with a guard against absorption; the reproducibility of a same seed (one run per cell here).

### 5.26 CRP prior only between slots that the policy does not distinguish (2026-10-16)
Follow-up of 5.25: we retain that $\nu>1$ (under-segmentation) and $\alpha=0.1$ (no better) are not suitable, and keep $\alpha=1$, $\nu=1$. The hypothesis tested: in the ambiguous regime (a slot whose summary is mixed), the size bias of the prior weighs against "open" and maintains the class merges. It is therefore restricted to the slots that the scores do not separate.

* **Definition** (`memory.gated_prior_offset`, `TrainConfig.crp_gate` $=\delta$, negative = disabled = 5.24). Let $s_a$ be the policy score for the valid slot $a$ and $S=\{a:\ s_a\ge\max_b s_b-\delta\}$ the set of slots within $\delta$ nats of the best. The logit offset becomes
$$\text{offset}_a=\begin{cases}\log\pi_0(a)-c & a\in S\\ 0 & a\notin S\end{cases},\qquad c=\log\!\sum_{b\in S}e^{s_b+\log\pi_0(b)}-\log\!\sum_{b\in S}e^{s_b},$$
with $\pi_0$ the CRP prior of 5.24. The constant $c$ is chosen so that the prior only redistributes the mass of $S$: the total probability of $S$ is exactly unchanged. A slot that the scores clearly rank lower is neither favoured nor penalised by its size. The scores are read without gradient to define $S$. Same function for sampling, the PPO ratio, the entropy, $L_{inv}$, the teacher phase and `predict`.
* **Code.** `cibirl/memory.py` (`gated_prior_offset`), `cibirl/policy.py` (`PolicyNet.prior_gate`), `cibirl/train.py` (`crp_gate`, teacher and distillation). Without gate the behaviour is identical to 5.24. Tests in `tests/test_slot_prior.py` (5 new: mass of $S$ conserved, tie-breaking unchanged at equal scores, opening slot not taxed outside $S$ with an exact odds ratio, no gradient through the gate, training with teacher); 178 tests pass.
* **Suite `configs/suites/toy_cov_k10_gate.yml`**: same base as `toy_cov_k10_mem` ($K=10$, $\lambda_{cov}=3$, `rho=0.1`, `neg_floor=0.1`, $\alpha=1$, $\nu=1$, stop with memory criterion, 300 epochs at most), seeds 0 to 4, $\delta=1$ and $\delta=5$; reference: `mem_a1` (same seeds, prior everywhere). Figure: `results/toy_cov_k10_gate/summary.png`.

| seed | $\delta=1$: memory (epoch), memory slots, ACC | $\delta=5$: memory (epoch), memory slots, ACC | reference $\alpha=1$ without gate: memory, clusters, ACC |
|---|---|---|---|
| 0 | 194, 5, 1.0 | no (300), 6, 1.0 | 120, 5, 1.0 |
| 1 | no (300), 10, 1.0 | 246, 5, 1.0 | 216, 5, 1.0 |
| 2 | no (300), 9, 1.0 | no (300), 5 (closed in the last 2 epochs), 1.0 | 289, 4, 0.8 |
| 3 | no (300), 10, 1.0 | 105, 5, 1.0 | 240, 5, 1.0 |
| 4 | no (300), 8, 1.0 | 237, 5, 1.0 | 171, 4, 0.8 |

* **No more class absorption.** The 10 gated runs end at 5 output clusters and ACC 1.000 (NMI 1.000), against 2 absorptions out of 5 without gate (one-sided Fisher exact test on 0/10 against 2/5: $p\approx0.1$, small sample). This is the intended effect: the size bias no longer weighs against "open" in the ambiguous regime.
* **$\delta=5$ closes almost everything.** Memory converged in 3 runs out of 5 (105, 237 and 246 epochs), a fourth is closed at epoch 299 (the criterion requires 10 consecutive epochs, it is short of them), a fifth keeps a duplicate at 300 epochs (6 slots, $\text{mem\_gap}=0.089$). **$\delta=1$ closes poorly**: 1 run out of 5 (epoch 194), the other 4 keep 8 to 10 slots at 300 epochs.
* **Why $\delta=1$ does not close: not because the twins leave $S$.** On the 5 unclosed runs (final checkpoints), the policy score gap between twin slots is 0.00 to 0.01 nat and the reward gap 0.000: the twins are always in $S$ and the size tie-break there is identical to 5.24 (neutral drift). What changes is the creation of duplicates: excess slots ($K_{mem}-K_{out}$) at epoch 40, means over 5 seeds: 4.4 ($\delta=1$), 4.2 ($\delta=5$), 2.8 (without gate). Hypothesis, not tested directly: without gate the bias $\log(\alpha/n)$ also taxes "open" during formation and limits duplicates; with the gate this tax disappears as soon as "open" is more than $\delta$ nats away, so there are more duplicates to close, and the neutral drift is slow (several pairs, sometimes three slots for one class). It is the same tax that, in the ambiguous state, maintained the absorptions.
* **Limits.** 5 seeds, one run per cell. The behaviour between $\delta=5$ and $\delta=\infty$ (= 5.25) is not explored. $\delta$ is one more parameter (in nats of policy score), with no motivated value beyond "distinguished = gap larger than $\delta$"; the reward gaps observed between different classes are 19 to 70 nats (5.23), between twins 0.

### 5.27 Sweep of the CRP concentration $\alpha$, 0.5 to 1.2 (2026-10-17)
Suite `configs/suites/toy_cov_k10_alpha.yml`: same base as `toy_cov_k10_mem` ($K=10$, $\lambda_{cov}=3$, `rho=0.1`, `neg_floor=0.1`, $\nu=1$), $\alpha\in\{0.5,\ 0.6,\ \dots,\ 1.2\}$, seeds 0 to 4, 40 runs, stop when the output is stable and the memory is the output (`stop.memory`, `mem_tol` 0.01) or at 400 epochs. No label. Figure: `results/toy_cov_k10_alpha/summary.png`. $\alpha=1.0$ reproduces `mem_a1` of 5.25 on the 5 seeds (maximum gap 0 on all metrics).

| $\alpha$ | right output (5 clusters, ACC 1.00) | memory converged | memory convergence epochs: median [min-max] | not converged at 400 epochs (seed: memory slots, mem_gap) |
|---|---|---|---|---|
| 0.5 | 5/5 | 5/5 | 262 [61-302] | none |
| 0.6 | 5/5 | 4/5 | 90 [80-198] | seed 2: 6, 0.036 |
| 0.7 | 5/5 | 5/5 | 80 [41-251] | none |
| 0.8 | 5/5 | 4/5 | 205 [77-319] | seed 3: 8, 0.062 |
| 0.9 | 5/5 | 5/5 | 213 [142-279] | none |
| 1.0 | 3/5 | 5/5 | 216 [120-289] | none |
| 1.1 | 5/5 | 4/5 | 180 [47-339] | seed 1: 7, 0.004 |
| 1.2 | 5/5 | 4/5 | 200 [112-229] | seed 0: 7, 0.092 |

* **Output.** Converged between epochs 40 and 80 (median 51) in the 40 runs. Right output in 38 runs out of 40.
* **Memory.** Converged in 36 runs out of 40, median 191 epochs, from 41 to 339. The 4 runs not converged at 400 epochs keep 6 to 8 slots for 5 output clusters (right output in all 4).
* **No trend in $\alpha$ for the closing time**: the medians (262, 90, 80, 205, 213, 216, 180, 200) vary less than the dispersion within a same value (for example 41 to 251 for $\alpha=0.7$).
* **Lost classes**: 2 runs out of 40, both at $\alpha=1.0$ (seeds 2 and 4, 4 clusters, ACC 0.8), none at the 7 other values (0 out of 35). These are the same seeds that absorb at $\alpha=0.1$ (5.25). Neither $\alpha=0.9$ nor $\alpha=1.1$ loses them: the loss is not monotone in $\alpha$. The choice of $\alpha=1$ as the problem case is post hoc (it is the value where the losses had been seen); the frequency of losses is of the order of 2 out of 40 at 5 seeds per value, which does not allow concluding to an effect of $\alpha$.
* **Limits.** 5 seeds per value, one run per cell; the losses depend on the trajectory (seed $\times$ $\alpha$), as shown by the bit-for-bit reproducibility at fixed $\alpha$ and the absence of a monotone link between neighbouring values; the 4 unconverged runs might close beyond 400 epochs.

### 5.28 Final protocol: stop at output convergence, then merge the duplicates of the memory (2026-10-17)
Decision (follow-up of 5.23 to 5.27): convergence is that of the **output** (label-free rule of Section 2, criteria 1 to 3, no memory criterion), then a last step, with policy and encoder frozen, merges the memory slots that carry the same cluster twice. No parameter is added to the training.

* **Why merging rather than reinsertion** (test outside training, 50 final checkpoints of 5.25 and 5.27, frozen policy, labels for evaluation only). (A) Merging by distance: the summaries of duplicate slots are at $\le 0.0021$, those of different clusters at $\ge 3.02$ (ratio $>1400$); any threshold between 0.01 and 3 gives the same partition. (B) Rebuilding the memory from empty by placing the images one by one (frozen argmax, with the CRP prior): 56 exact reconstructions out of 100 (2 image orders per run), 18 runs out of 50 failed in both orders, and class merges (1 to 4 clusters); (B) depends on the order and is not retained.
* **Merge rule** (`cibirl/cleanup.py`, `TrainConfig.final_cleanup`, `cleanup_ratio`). The distances between the summaries of the non-empty slots are sorted; if the largest ratio between two consecutive distances is $\ge r$ (guard $r=10$), all the pairs below this jump are duplicates and are merged (single linkage, the largest slot is kept, summaries recomputed); otherwise nothing is merged. No distance has to be chosen: the jump is $\ge 53$ when duplicates exist and $\le 1.19$ when there are none (see below), the guard 10 is not adjusted. With 2 slots or fewer there is nothing to compare and nothing is merged. Then the output is re-read (`predict`, frozen memory).
* **Protocol.** Suite `configs/suites/toy_cov_k10_final.yml`: toy, $K=10$, $\lambda_{cov}=3$, `rho=0.1`, `neg_floor=0.1`, CRP $\alpha=1$, $\nu=1$, **100 seeds (0 to 99)**, stop at output convergence (`min_epochs` 40), `final_cleanup: true`. No label in the training nor in the stop. Summary: `results/toy_cov_k10_final/summary.txt` and `summary.png`.

| 100 seeds | result |
|---|---|
| output convergence (label-free rule) | 100/100, epoch 40 to 95 (median 51, mean 53) |
| right output at stop (5 clusters, ACC 1.00), before clean-up | 96/100 |
| memory slots, before then after clean-up | 7.13 [3-10] then 4.96 [3-6] |
| memory = output ($\text{mem\_gap}\le0.01$), before then after clean-up | 22/100 then 96/100 |
| runs with duplicates merged / without duplicate | 79 / 21; 217 merges, of which 0 between different classes |
| largest jump: with duplicates / without | min 53 (median 2458) / max 1.19 (median 1.03) |
| distances: merged duplicates / distinct clusters | max 0.094 (per-run median 0.0018) / min 2.36 |
| right output after clean-up (`predict` reading, planned protocol) | **93/100** |

* **The 7 wrong outputs.** (i) Four already were at output convergence (seeds 12, 30, 59, 94: 4, 4, 4 and 3 clusters): the stopping rule stops on a stable but under-segmented partition (5.18). (ii) Three are caused by the clean-up (seeds 19, 80, 81): their output was right (5 clusters, ACC 1.00) before. Mechanism verified on seed 19: the memory had its 10 slots occupied (two per class), so the "open" action was not possible; after the merge, 5 empty slots appear and the policy prefers "open" to "join" for two classes (policy score gap +54 and +29 nats, whereas the reward gap is $-3.4$ and $-3.0$: the reward prefers to join); these images are all sent to the first empty slot (outlier group of `predict`), which groups two classes. Among the 16 runs whose memory was full before clean-up, 3 are in this case (0 were wrong before).
* **Alternative readings, post hoc** (computed after seeing these results, on the already cleaned checkpoints; `scripts/readings_after_cleanup.py`): output read without the "open" action (each image goes into an existing cluster): **96/100**, i.e. the same 4 failures as at convergence and none due to the clean-up any more; partition carried by the cleaned memory: 82/100 (a few images misplaced by the sampled placements of the last epoch, ACC 0.99 for example, plus the 4 wrong outputs). The choice of the reading without "open" is therefore post hoc; validating it requires fresh seeds.
* **Limits.** Toy only, $\alpha=1$, one run per seed. The 4 under-segmentation failures at stop (4 %, 95 % upper bound of 10 % with 4 out of 100) are not addressed. The guard $r=10$ has a margin of a factor 5 (53 against 10) on the "duplicates" side and 8 (10 against 1.19) on the "no duplicate" side over these 100 runs; at stop, the distances between duplicates ($\le0.094$) are larger than at the end of a long run ($\le0.0021$, 5.25).


### 5.29 Closing the memory without distorting the policy: relabel (A) and reintroduction (C) (2026-10-17)
Starting point: the 3 outputs broken by the clean-up in 5.28 (seeds 19, 80, 81). Re-reading the policy on the merged memory puts it in a state it did not see during training (empty slots while the memory was full there); it prefers "open" there. Two ways to close the memory without this, compared to the merge of 5.28 on the **same training** (same 100 seeds, each variant starts from a copy of the same memory; the training is identical to 5.28, verified: same figures for "at stop" and "merge"). No label in the training nor in the stop.

* **A, `relabel`.** The output is the one read at stop (before the merge); the merge of 5.28 (jump, guard 10) is only used to rename its slots. The policy is never queried on the merged memory.
* **C, `reintroduce`** (`cibirl/cleanup.py`, `reintroduce`). From the pre-merge memory, sweeps over the images in a random order (seed of the config): each image is removed from its slot, then placed where the argmax of $\pi(a\mid z,m,C)$ sends it, with the CRP prior of the training (image removed from its slot) and the summaries updated at each placement. This is a training epoch with the argmax instead of sampling, policy frozen. At a tie between two duplicates, the prior favours the larger one: each move enlarges it, the smaller one empties (the neutral drift becomes a ratchet). We stop when a sweep moves no image (cap of 20). No distance, no guard. **The output of C is the partition where the sweeps stop** (memory = output by construction); `reintroduce_no_open` removes the "open" action (the images are reintroduced INTO the existing clusters), `reintroduce` keeps it.
* **Suite** `configs/suites/toy_cov_k10_final_ac.yml` (100 seeds, 0 to 99, same settings as 5.28, `cleanup_mode: relabel`), summary `results/toy_cov_k10_final_ac/summary_variants.txt` (`scripts/summarize_variants.py`). "Right" = 5 clusters and ACC 1.00; "right memory" = the partition carried by the memory is itself right.

| 100 seeds | right output | memory = output | slots (mean) | right memory | lost relative to the stop |
|---|---|---|---|---|---|
| at stop (before any clean-up) | 96 | 22 | 7.13 | 19 | |
| `merge` (5.28) | 93 | 96 | 4.96 | 82 | 19, 80, 81 |
| **A `relabel`** | **96** | 99 | 4.96 | 82 | none |
| C `reintroduce` | 92 | 100 | 5.15 | 92 | 18, 19, 80, 81 |
| **C `reintroduce_no_open`** | **96** | **100** | **4.95** | **96** | none |

* **A** keeps the 96 right outputs and fixes the 3 broken by `merge`. The memory remains that of the sampled placements of the last epoch: 82 right memories, the others have misplaced images (14 runs, memory ACC 0.990 on average, 0.90 at minimum) plus the 4 failures. A single output differs from its memory (seed 55, gap 0.10).
* **C with "open"** does not close the memory when it was full: among the 16 runs with a full memory, 4 (seeds 18, 19, 80, 81) keep 10 slots, with 10 images changing slot at each sweep (cap of 20 sweeps reached in these 4 runs only). Mechanism: as soon as a slot empties, "open" becomes possible again and the policy chooses it for about ten images (same weakness as 5.28). Not retained.
* **C without "open"** converges in at most 2 sweeps in the 100 runs (median of 120 images moved in the first, none in the second). The closed memory carries a partition that is **exactly right** in the 96 runs whose output is right (ACC 1.00 in each, 5 slots), without the misplaced images of A (96 right memories against 82). No output lost, none made wrong from right.
* **The 4 remaining failures are the same for A and C** (seeds 12, 30, 59, 94: under-segmentation already present at stop, 4, 4, 4 and 3 clusters): neither A nor C addresses them, the clean-up does not create clusters.

* **What A and C do not fix.** The policy re-read (`predict`) on the closed memory still fails in 7 runs (4 + 19, 80, 81) for `reintroduce_no_open`: the preference for "open" as soon as slots are empty is a weakness of the policy outside the states seen in training (full memory), which A and C work around without correcting it. To assign NEW images with `predict` on the closed memory, this defect would have to be addressed; not tested.
* **Limits.** Toy, $\alpha=1$, one seed = one run, same seeds (0 to 99) as 5.28. The four readings were planned before the 100 runs were launched, but `reintroduce_no_open` comes from a pilot on 5 seeds (0, 12, 19, 80, 81) and from the reading without "open" of 5.28, made after seeing these same seeds: its advantage over A (right memory 96 against 82) has to be confirmed on fresh seeds (100 to 199). Random sweep order: a single order seed per run (`cfg.seed`).

### 5.30 Validation on 100 fresh seeds: A is retained (2026-10-18)
Rule fixed before the launch (header of `configs/suites/toy_cov_k10_final_ac_val.yml`): C (`reintroduce_no_open`) is retained if its output is right in at least as many runs as A **and** if its memory is right in more runs than A; otherwise A. Suite `toy_cov_k10_final_ac_val`: seeds 100 to 199, same settings, same code as 5.29 (no setting touched). Summary: `results/toy_cov_k10_final_ac_val/summary_variants.txt`.

| seeds 100 to 199 | right output | memory = output | slots (mean) | right memory | lost relative to the stop |
|---|---|---|---|---|---|
| at stop | 98 | 21 | 7.16 | 19 | |
| `merge` | 98 | 97 | 5.02 | 91 | none |
| **A `relabel`** | **98** | 97 | 5.02 | 91 | none |
| C `reintroduce` | 95 | 100 | 5.07 | 95 | 102, 106, 181 |
| C `reintroduce_no_open` | 97 | 100 | 4.97 | 97 | 102 |

* **Decision: A.** C without "open" has the most right memory (97 against 91) but loses one right output (97 against 98): the condition "at least as many right outputs" is not met.
* **Cumulative over the 200 seeds (0 to 199)**, right output / right memory: A 194 / 173; C without "open" 193 / 193; `merge` 191 / 173; C with "open" 187 / 187. A and C without "open" are 1 run apart on the output (194 and 193 out of 200); the clear difference is on the memory (193 against 173), which is not what is evaluated.
* **Failures of A**: 12, 30, 59, 94 (5.29) and 122, 152 (4 clusters, ACC 0.80, already wrong at stop), i.e. 6 out of 200. No merge between different classes (0 out of 214 merges in this suite).
* **Seed 102** (right at stop, lost by C): at stop the memory has 9 slots and a partition right at only 0.68; after the merge, 7 slots, memory right at 0.82; the sweeps of C go down to 4 clusters (ACC 0.80) in 3 sweeps. C re-reads the memory, A only reads the output: C inherits the defects of the memory, A does not.
* **Convergence**: 100/100 according to the output rule (min epoch 40, median 47, max 100). C with "open" reaches the cap of 20 sweeps in 2 runs, C without "open" in none (at most 3 sweeps).
* **Defect of the re-read policy**: `predict` on the memory closed by C: 97/100 (like the output of C here).
* **Limits.** Toy, $\alpha=1$, one run per seed, a single sweep order per run. The rule decides on the output; it says nothing about the use of the closed memory to assign new images.

### 5.31 Evaluation on a separate test partition, results over 200 seeds and README report (2026-10-18)
Follow-up of 5.30 (A is retained). Question: the training evaluation is transductive (the model groups its own 1500 images); what does the frozen model give on images it has never seen? No setting is touched, no training is rerun for the 200 seeds.

* **Test partition.** `scripts/toy_data.py`, `make_toy_test(seed, test_seed, mode)`: same shapes and same palette as the training seed (`_prototypes`, `make_toy` is unchanged: same hash of the images), but 1500 fresh images drawn from a separate generator (`test_seed = seed`). Three modes: `iid` (random backgrounds, as in training), `aligned` (the background of a class is its palette colour, plus 0.03 of noise), `inverted` (the background of class $c$ is the colour of class $c+1$).
* **Reading.** `scripts/eval_heldout.py`: loads `checkpoint.pt` and `memory_pre.pt`, restores the pre-clean-up memory, then assigns each image by the policy argmax (CRP prior included, memory unmodified) and renames the slots by `relabel` (jump, guard 10) as in 5.29. Check: the same reading on the training images reproduces `preds.npy` identically in 200 runs out of 200. Labels only evaluate: ACC (Hungarian), class NMI, background NMI, number of clusters, share of images sent to an empty slot ("open", counted as one more cluster), worst-group ACC (class $\times$ coarse background id, at least 10 images).
* **Results (200 seeds, 0 to 199).** Right output (5 clusters, ACC 1.00): 194/200, i.e. 3.0 % failures (Clopper-Pearson 95 %: 1.1 to 6.4 %); fresh seeds 100 to 199: 98/100 (0.2 to 7.0 %). ACC $0.993\pm0.042$, class NMI $0.9968$, NMI(cluster, background) `iid` $0.0039$; **the four readings (training, `iid`, `aligned`, `inverted`) give the same ACC seed by seed (0 difference)**, hence $\Delta$ACC $=0$; worst-group ACC $0.970$ (1 on the right seeds, 0 on the 6 failures). Clusters $4.965\pm0.209$. In `aligned` and `inverted` the NMI with the background is 0.71 by construction (the background is tied to the class): this is not a leak.
* **Clean-up.** Convergence of the output rule: 200/200, epoch 40 to 100 (median 48.5). Memory slots: 7.14 at stop, 4.99 after clean-up. 431 merges, 0 between different classes. Distances: merged duplicates $\le 0.094$ (634 pairs), distinct clusters $\ge 2.01$ (4067 pairs) except one pair at 0.117 (seed 102, contaminated slot).
* **Pixel reference (difficulty context, not a comparison with the baselines).** K-Means on raw pixels, 200 seeds: $K=5$: ACC 1.000, NMI 1.000, 200/200 right (the toy is solved in pixels if $K$ is known); $K=10$: ACC $0.567\pm0.039$, class NMI 0.843, NMI(cluster, background) 0.159, 0/200 right (each class is cut in two according to the brightness of the background).
* **Failures.** Seeds 12, 30, 59, 94, 122, 152 (4, 4, 4, 3, 4, 4 clusters; ACC 0.8, 0.8, 0.8, 0.6, 0.8, 0.8), all already wrong at stop; in each the clusters are pure or unions of two whole classes (no class is cut). **The replay of seed 12 (same config, one point per epoch) gives a right output**: the training is not deterministic bit for bit (floats, threads), the failure rate is a property of the method over seeds, not of a seed. The replay of seed 20 is right (42 epochs against 46 in the stored run).
* **Special cases.** Seed 106: a whole class (20 % of the images) goes to the "open" group at test time and counts as one more cluster (mean share 0.001). Seed 102: contaminated slot (summary pair at 0.117 between distinct clusters).
* **Report.** `scripts/make_report.py` produces `results/toy_final_report/`: `fig1_task.png` (the task), `fig2_difficulty.png` (pixels and K-Means $K=10$), `fig3_clusters_seed20.png` and `fig4_trace_seed20.png` (one run: clusters at epochs 12, 13 and at the end, then curves), `fig5_dynamics_200seeds.png` (mean dynamics), `fig6_results_200seeds.png` (stop, slots, ACC per seed, distances), `fig7_failures.png`, `heldout_metrics.{json,csv}`, `report_numbers.json`, `trace_seed20.npz`. Test: `tests/test_toy_data.py` (206 tests pass in total).
* **Not done.** Baselines, ablations $\lambda=0$ and "reward + prior as the action rule during training", a step that splits a cluster formed of two classes. Toy only, oracle transformation bank, `spurious=0` in training. Settings chosen on seeds 0 to 9 at most.

### 5.32 Baselines 1 to 3 and the toy benchmark as files, wired and tested, not run (2026-10-18)

Goal: evaluate three baselines and our model under the same conditions (train images, then `iid`, `aligned`, `inverted` test sets), each trained without and with a colour shortcut (`spurious` 0.9), and be able to do the same on Colored MNIST and Waterbirds. Nothing long was run: this step is the setup and its tests (263 tests pass, 1 skipped).

* **`spurious` (training only).** Probability, per training image, that the background (toy) or the colour (Colored MNIST) is the palette colour of the class instead of a random one. At 0 the `aligned` and `inverted` test sets only check robustness at test time; at 0.9 a shortcut is available to learn, and $\Delta\mathrm{ACC}=\mathrm{ACC}(\text{aligned})-\mathrm{ACC}(\text{inverted})$ measures how much the output follows it. All 200 results so far are at 0.
* **What it can and cannot show with the oracle bank.** `toy_oracle` redraws the background independently of the class in every view, so any method that learns from these views (SimCLR, IIC, baseline 3, CIB-IRL) is immune by construction; only methods without views (K-Means, GMM) can follow the shortcut. Isolating the contribution of the bottleneck needs the `generic` bank. Indicative check (2 seeds, K-Means on pixels, spurious 0.9, not a result): $K=5$ stays at ACC $\approx1$ on all test sets; $K=10$ gives ACC 0.93 on the train images, 0.67 to 0.70 on `iid`, $\Delta\mathrm{ACC}$ 0.08 to 0.00. Pixel K-Means with $K=5$ is robust to the shortcut because the shape dominates the pixels.
* **Shared protocol.** `eval/protocol.py` holds `partition_metrics`, `is_right`, `delta_acc` and `TEST_MODES`, moved out of `scripts/eval_heldout.py` (same code, so the stored results are unchanged). Bug found by the tests and fixed: `worst_group_acc` raised a `KeyError` when a method has fewer clusters than classes (K-Means with $K<5$); a class without a cluster now has accuracy 0.
* **Benchmarks.** `scripts/datasets.py`: a `Benchmark` is a train builder, a test builder and the modes. Toy; Colored MNIST (train from MNIST train, test sets from t10k, hue palette, `spurious`); Waterbirds (metadata.csv, official splits, `aligned` = landbird on land or waterbird on water, `inverted` = the other groups). Colored MNIST and Waterbirds are tested on fake data only.
* **Toy as files.** `data/toy/seedNNN.npz` (seeds 0 to 9 tracked, 4 MB; `scripts/make_toy_dataset.py --seeds 0-199` for all), exact `float32`, with `manifest.json` (sha256 of the arrays, not of the zip). `datasets.build("toy", seed, source="auto"|"file"|"generate")`. The generator stays the source of truth (`--verify`, `tests/test_toy_files.py`).
* **Baselines.** `baselines/`: 1 = K-Means / diagonal GMM on pixels or ResNet features (ResNet not tested: torchvision absent); 2 = SimCLR (NT-Xent, K-Means on the backbone output) and IIC, both with the transformation bank of the model and the same `SmallCNN`, fixed number of epochs; MoCo v3 not implemented. 3 = our trainer with `configs/baselines/irl_noib.yml` (`beta: 0`, `lam: 0`) run by `run_experiment.py`; the sequential training of README 5.6 is not implemented. Runner: `scripts/run_baseline.py`; `run_suite.py` accepts `method:`; table: `scripts/compare_methods.py`.
* **Configurations.** `configs/benchmarks/toy.yml` = the final configuration with every value written out, equal to the `TrainConfig` stored in the 200 runs (`tests/test_benchmark_configs.py`); `configs/suites/toy_final.yml` (same 200 seeds as one suite, not run); `configs/suites/toy_compare.yml` (20 experiments: 10 methods x 2 regimes, not run); first-design `configs/colored_mnist.yml`, `configs/waterbirds.yml` (not validated).
* **Decisions for the comparison (2026-10-18).** (1) $K=10$ only for the baselines: the number of classes is not given in practice, and $K=5$ is too easy (K-Means on pixels with $K=5$ is right on 200/200 seeds, kept as a remark in README 11.1.3, not in Table 2). (2) SimCLR, IIC and baseline 3 use the same bank as the model, `toy_oracle`: same dataset, exact comparison; the question of an oracle against a `generic` bank is dropped for this comparison (the view-based methods are therefore immune to the shortcut by construction). (3) `spurious` 0.9 only, besides 0. (4) Seeds 0 to 199 for every experiment, the same everywhere. (5) 300 epochs for SimCLR and IIC (fixed), as for the maximum of the model; baseline 3 has the stopping rule of the model with a maximum of 300. CIB-IRL at spurious 0 is not rerun: its 200 runs (`results/toy_final_report/heldout_metrics.json`) are the clean row. The suite `toy_compare.yml` is now 11 experiments (2200 runs).
* **Cost (single thread).** Measured on one seed: K-Means 4 s, GMM 4 s, CIB-IRL at spurious 0.9 197 s (42 epochs), baseline 3 220 s (50 epochs), held-out evaluation 4 s. SimCLR and IIC cost 0.78 s per epoch each (profiled over 20 epochs; the backward pass dominates), hence about 240 s for 300 epochs. A first single measurement of IIC (563 s) and of K-Means (56 s) came from a noisy moment of the machine and was wrong; there is no speed-up to find in IIC. The model needs 52.8 epochs on average, hence about 250 s per run. Total of the 2200 runs: about 100 core-hours (SimCLR 27, IIC 27, baseline 3 28, CIB-IRL 14, the rest 2).
* **Not done.** Experiments of this step; MoCo v3; ViT features; the real Colored MNIST and Waterbirds data; a ResNet encoder for our own model (`TrainConfig` has none).

### 5.33 $\Delta$ACC with the class-to-cluster matching fixed on the train partition (2026-10-20)

Found while reading the first SimCLR results (117 of 200 seeds, both regimes), not by a failure of a run.

* **Problem.** `partition_metrics` computed the ACC of every partition with its own Hungarian matching between clusters and classes. In the test modes `aligned` (background of class $c$ = colour of $c$) and `inverted` (background of class $c$ = colour of $c+1$) the colour is a one-to-one function of the class, so a method that clusters by background colour alone has ACC 1 in both modes and $\Delta\mathrm{ACC}=0$: the matching absorbs the permutation. A likely contribution to the gap between the `aligned`/`inverted` ACC and the `iid` ACC seen for SimCLR with $K=10$ (0.95 against 0.72 to 0.77): the backgrounds there are nearly uniform within a class (palette colour $\pm 0.03$), which makes the images easier to cluster; not isolated. Every $\Delta\mathrm{ACC}$ computed before this date (README Section 9.3 and the 200 results of 5.31, all equal to 0 for CIB-IRL) is therefore not evidence about the shortcut; the NMI with the background is not affected.
* **Definition adopted.** `class_to_slot(pred_train, y, K)` is the Hungarian matching of the TRAIN partition; `partition_metrics(..., slot_of_class=...)` returns `acc_fixed`, the share of test images sent to the slot of their class under that matching (for the train partition `acc_fixed` = `acc`). $\Delta\mathrm{ACC}=\mathrm{ACC}_\text{fixed}(\text{aligned})-\mathrm{ACC}_\text{fixed}(\text{inverted})$ (`eval.protocol.delta_acc`). A method that clusters by colour has 1 on `aligned` and 0 on `inverted`, hence $\Delta\mathrm{ACC}=1$; a method that follows the classes has 0 (`tests/test_baselines.py::test_fixed_matching_sees_a_method_that_clusters_by_background_colour`). The former quantity stays available as `delta_acc_rematched`, a diagnostic. `acc` (re-matched) is unchanged. The labels still only evaluate.
* **Wired in.** `scripts/eval_heldout.py` (CIB-IRL and baseline 3), `scripts/run_baseline.py` (baselines 1 and 2), `scripts/compare_methods.py` (two columns: fixed matching, then re-matched), `scripts/make_report.py`. A row without `acc_fixed` makes `delta_acc` raise `KeyError` and appears as "-" in the table: the two definitions are never mixed.
* **Decision (user, 2026-10-20).** Re-evaluate what exists with the new metric; do not retrain SimCLR here: the 236 SimCLR runs done (old metric) are removed from `results/toy_compare` and SimCLR is rerun on the user's machine (5 cores) once everything else is done; IIC is run here with the new metric. K-Means and GMM (800 runs, seconds each) are refitted; `irl_noib` (400) and `final_spur90` (200) are re-evaluated from their checkpoints. The 200 clean CIB-IRL runs of 5.31 have no checkpoint in this workspace (`checkpoint.pt` is git-ignored): they are re-evaluated on the user's machine with `python scripts/eval_heldout.py results/toy_cov_k10_final_ac results/toy_cov_k10_final_ac_val --experiment final --out results/toy_final_report`.
* **First readings with the former metric, now to be redone.** K-Means $K=10$, $s=0.9$, 200 seeds: $\Delta\mathrm{ACC}$ $+0.040$ (aligned 1.00, inverted 0.96); GMM $+0.006$; SimCLR (117 seeds) $-0.012$ ($s=0$) and $-0.015$ ($s=0.9$); `irl_noib` and `final` held-out ACC 0.994 on all test sets. Do not quote them.

## 4. Tracking data
`results/` is versioned in git (to document the results). Suites: `configs/suites/*.yml`, launched by `scripts/run_suite.py`.
