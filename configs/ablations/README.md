# Ablations (README Section 9)

Each file only contains the keys it changes. They are applied ON TOP of an experiment config:

    TrainConfig.from_yaml("configs/toy.yml", "configs/ablations/gamma0.yml")

| File | Question |
|---|---|
| `gamma0.yml` | Does the sequential return bring anything beyond a per-image reward (`gamma = 0`)? |
| `no_ib.yml` | Does compression alone remove the spurious features (`beta = 0`)? |
| `no_inv.yml` | Is the decision-consistency term necessary (`lam = 0`)? |
| `no_debias.yml` | How much do false negatives hurt (`tau_plus = 0`)? |
| `reward_sees_only_m.yml` | Does `z` bring more than `m` to the decisions? |
| `no_standardize.yml` | README to the letter: intent without batch standardization (m can collapse) |
| `no_center.yml` | Raw reward in the RL return instead of the per-state centered one |
| `rho_on.yml` | Explicit anti-collapse entropy term |
