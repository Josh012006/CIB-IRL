"""Diagnostic (docs 5.45): scale of the raw advantages and activity of the gradient clip, measured on a COPY of a resume_state.pt.

Nothing is written to the run: the checkpoint is copied, the PPO steps are applied in this process only. For a few rollouts of
the current policy it prints the norm of the gradient of the policy loss, of the value loss, of their sum (what the clip sees),
the clip factor, and the raw GAE advantages.

Usage, from the repository root:  OMP_NUM_THREADS=1 python3 scripts/diag_adv_clip.py <run_dir> [n_rollouts]
(<run_dir> contains config.yml, run.yml and resume_state.pt, i.e. the run was made with --ckpt-every).
"""
import sys, shutil, itertools
import numpy as np, torch
sys.path.insert(0, '.')
from scripts import datasets
from cibirl import transforms
from cibirl.train import Trainer, TrainConfig
import cibirl.losses as L
from cibirl.memory import valid_slot_mask

run = sys.argv[1]
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 6
torch.set_num_threads(1)
cfg = TrainConfig.from_yaml(run + "/config.yml")
import yaml
info = yaml.safe_load(open(run + "/run.yml"))
bank_name = info["transform"]
from pathlib import Path
bank = transforms.build(Path('configs/transforms')/f'{bank_name}.yml')
dp = info["data_params"]
data = datasets.build("mnist", cfg.seed, digits=dp["digits"], n_per_class=dp["n_per_class"])
tr = Trainer(cfg, data.images, bank, device="cpu", oracle_labels=None, eval_labels=data.labels)
tmp = "/tmp/diag_adv_clip_copy.pt"
shutil.copy(run + "/resume_state.pt", tmp)
tr.load_state(tmp)
print(f"{run.split('/')[-1]} : checkpoint epoch {tr.iteration // 30}, alpha={cfg.alpha}, adv_norm={cfg.adv_norm}, grad_clip={cfg.grad_clip}")

def gnorm(params, loss):
    for p in params:
        p.grad = None
    loss.backward(retain_graph=True)
    g = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in params if p.grad is not None))
    return float(g)

pol_params = list(tr.policy.parameters())
val_params = list(tr.value.parameters())
allp = tr.policy_params
perm = torch.randperm(tr.N)
rows = []
for b in range(nb):
    idx = perm[b * cfg.learner_batch:(b + 1) * cfg.learner_batch]
    roll = tr.learner_rollout(idx)
    valid = valid_slot_mask(roll.empty)
    with torch.no_grad():
        scores = tr.reward(roll.z, roll.m, roll.summaries, roll.empty)
        if cfg.center_reward:
            n_valid = valid.sum(-1, keepdim=True)
            scores = scores - (scores * valid).sum(-1, keepdim=True) / n_valid
        r = scores.gather(1, roll.actions.unsqueeze(1)).squeeze(1)
        v_old = tr.value(roll.z, roll.m, roll.summaries, roll.empty)
        adv_raw, ret = L.compute_gae(r, v_old, cfg.gamma, cfg.gae_lambda)
    adv = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8) if cfg.adv_norm else adv_raw
    per_epoch = []
    for e in range(cfg.ppo_epochs):
        logp_all = tr.policy.log_probs(roll.z, roll.m, roll.summaries, roll.empty, roll.log_prior)
        logp_new = logp_all.gather(1, roll.actions.unsqueeze(1)).squeeze(1)
        from cibirl.policy import PolicyNet
        ent = PolicyNet.entropy(logp_all, roll.empty)
        v_new = tr.value(roll.z, roll.m, roll.summaries, roll.empty)
        out = L.ppo_losses(logp_new, roll.logp_old, adv, ent, v_new, ret, cfg.ppo_clip, cfg.alpha)
        # separate norms: policy loss alone, value_coef * value loss alone, and the sum
        gp = gnorm(allp, out.policy_loss)
        gp_pol = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in pol_params if p.grad is not None)))
        gv = gnorm(allp, cfg.value_coef * out.value_loss)
        gj = gnorm(allp, out.policy_loss + cfg.value_coef * out.value_loss)
        scale = min(1.0, cfg.grad_clip / (gj + 1e-6))
        per_epoch.append(dict(gp=gp, gv=gv, gj=gj, scale=scale, kl=float(out.approx_kl), vloss=float(out.value_loss)))
        tr._clip_step(tr.opt_policy, allp, out.policy_loss + cfg.value_coef * out.value_loss)
    rows.append(dict(adv_std=float(adv_raw.std()), adv_absmax=float(adv_raw.abs().max()), adv_mean=float(adv_raw.mean()),
                     r_std=float(r.std()), r_mean=float(r.mean()), ret_mean=float(ret.mean()),
                     v_rmse=float(((v_old - ret) ** 2).mean().sqrt()), ep=per_epoch))
    print(f"rollout {b}: raw adv std {rows[-1]['adv_std']:.2f} largest |adv| {rows[-1]['adv_absmax']:.1f} | r std {rows[-1]['r_std']:.2f} | "
          f"mean return {rows[-1]['ret_mean']:.1f} | rmse(V,G) {rows[-1]['v_rmse']:.1f}")
    for e, d in (enumerate(per_epoch) if b == 0 else []):
        print(f"   ppo-ep {e}: ||g_policy|| {d['gp']:.3g}  ||g_value|| {d['gv']:.3g}  ||g_joint|| {d['gj']:.3g}  clip factor {d['scale']:.3g}  kl {d['kl']:.4f}  L_xi {d['vloss']:.3g}")

gps = [d['gp'] for r_ in rows for d in r_['ep']]
gvs = [d['gv'] for r_ in rows for d in r_['ep']]
gjs = [d['gj'] for r_ in rows for d in r_['ep']]
sc = [d['scale'] for r_ in rows for d in r_['ep']]
print("SUMMARY:")
print(f"  std of the raw advantages (mean over rollouts): {np.mean([r_['adv_std'] for r_ in rows]):.2f} ; mean largest |adv| {np.mean([r_['adv_absmax'] for r_ in rows]):.1f}")
print(f"  ||g_policy|| median {np.median(gps):.3g} ; ||g_value|| median {np.median(gvs):.3g} ; ||g_joint|| median {np.median(gjs):.3g}")
print(f"  clip active (factor < 1) in {100 * np.mean([s < 1 for s in sc]):.0f} % of the steps ; median factor {np.median(sc):.3g}, min {np.min(sc):.3g}")
print(f"  share of the value in the joint norm (median of gv/gj): {np.median([gv_ / gj_ for gv_, gj_ in zip(gvs, gjs)]):.2f}")
print(f"  policy gradient actually applied = g_policy x factor: median {np.median([a * b for a, b in zip(gps, sc)]):.3g}")
npol = sum(p.numel() for p in pol_params)
applied = [a * b for a, b in zip(gps, sc)]
print(f"  policy parameters: {npol} ; applied gradient per parameter (RMS): median {np.median(applied) / np.sqrt(npol):.2e}, min {np.min(applied) / np.sqrt(npol):.2e} (Adam eps = 1e-8)")
print(f"  spread of the clip factor between steps: max/min = {np.max(sc) / np.min(sc):.1f} ; neighbouring steps (median max/min ratio) {np.median([max(a, b) / min(a, b) for a, b in zip(sc[:-1], sc[1:])]):.1f}")
print(f"  ||g_value|| / ||g_policy|| median: {np.median([gv_ / gp_ for gv_, gp_ in zip(gvs, gps)]):.3g}")
