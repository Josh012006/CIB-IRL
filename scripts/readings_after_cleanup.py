"""Three readings of the final clustering, from the saved checkpoint (the memory is already cleaned). The labels only EVALUATE.
  predict   argmax of the policy with the cleaned memory frozen (what final_metrics.json reports)
  no_open   the same argmax with the action "open a new slot" removed: every image goes to an existing cluster
  memory    the partition held by the cleaned memory (the sampled placements of the last epoch)
Usage: python scripts/readings_after_cleanup.py results/toy_cov_k10_final   (writes readings.json there)"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import datasets
from scipy.optimize import linear_sum_assignment
from cibirl import transforms
from cibirl.train import TrainConfig, Trainer
from cibirl.memory import crp_log_prior
torch.set_num_threads(1)
bank = transforms.build(str(ROOT / "configs" / "transforms" / "toy_oracle.yml"))
R = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "results/toy_cov_k10_final"
runs = sorted(glob.glob(R + "/final_seed*/final_metrics.json"), key=lambda p: int(p.split("final_seed")[1].split("/")[0]))
def acc(pred, y):
    ks = np.unique(pred); conf = np.zeros((5, len(ks)))
    for j, k in enumerate(ks): conf[:, j] = np.bincount(y[pred == k], minlength=5)
    r, c = linear_sum_assignment(-conf); return conf[r, c].sum() / len(y)
out = {}
for f in runs:
    d = os.path.dirname(f); name = os.path.basename(d); seed = int(name.split("seed")[1])
    cfg = TrainConfig.from_yaml(d + "/config.yml"); data = datasets.build("toy", seed, spurious=0.0); y = data.labels.numpy()
    tr = Trainer(cfg, data.images, bank, device="cpu", eval_labels=data.labels); ck = torch.load(d + "/checkpoint.pt", map_location="cpu", weights_only=False)
    for k in ("encoder", "intent", "reward", "policy", "value"): getattr(tr, k).load_state_dict(ck[k]); getattr(tr, k).eval()
    tr.memory.assign, tr.memory.member_m, tr.memory.counts, tr.memory.summaries = ck["assign"], ck["member_m"], ck["counts"], ck["summaries"]
    with torch.no_grad():
        z = tr.encoder(data.images, sample=False).z; m = tr.intent.mean(z); summ, empty = tr.memory.state(); N = len(y)
        lp0 = crp_log_prior(tr.memory.counts, empty.unsqueeze(0).expand(N, -1), cfg.crp_alpha, tr.memory.assign, cfg.crp_power)
        lp = tr.policy.log_probs(z, m, summ.unsqueeze(0).expand(N, -1, -1), empty.unsqueeze(0).expand(N, -1), lp0)
    a = lp.argmax(-1).numpy(); lp2 = lp.clone(); lp2[:, empty] = -float("inf"); b = lp2.argmax(-1).numpy(); c = ck["assign"].numpy()
    out[name] = {k: (acc(v, y), int(sum(1 for q in np.unique(v) if (v == q).sum() >= 10))) for k, v in (("predict", a), ("no_open", b), ("memory", c))}
json.dump(out, open(os.path.join(R, "readings.json"), "w"), indent=1)
for k in ("predict", "no_open", "memory"):
    ok = sum(1 for v in out.values() if v[k][1] == 5 and v[k][0] > 0.999)
    print(f"{k:>8s}: {ok}/{len(out)} runs with 5 clusters and ACC 1.00")
