"""Orientation of the class code of the intent m around an event (an L_cov step): epoch t-1 -> t, from per-epoch checkpoints
(run_experiment.py --save-epochs A-B). The class centroids use the TRUE classes: this is an analysis, never training.

    python scripts/analyze_code_orientation.py results/<suite>/<run> <seed>

For every epoch t of the checkpoints it prints: the accuracy of the output partition, and, from t-1 to t,
  pairwise-dist change   relative change of the 5 x 5 matrix of distances between class centroids (rotation invariant),
  raw centroid change    relative change of the centred centroids in the coordinates of m,
  after best rotation    the same after the best orthogonal alignment (Procrustes): small = the code only turned,
  max principal angle    largest angle between the 4-dimensional subspaces spanned by the centroids at t-1 and t,
  per-image shift        mean displacement of an image in m, divided by the mean distance between class centroids,
  policy(t-1) on rep(t)  accuracy of the partition obtained with the policy of epoch t-1 on the representation of epoch t,
                         and the share of images whose slot differs from the policy of epoch t (a stale policy).
Then the share of the class-code variance carried by its 4 busiest coordinates (a code aligned with the axes: near 1)."""
import sys, json, glob, os
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import datasets
from cibirl import transforms
from cibirl.train import TrainConfig, Trainer
from eval.metrics import cluster_acc
run_dir, seed = sys.argv[1], int(sys.argv[2])
cfg = TrainConfig.from_yaml(run_dir + "/config.yml"); data = datasets.build("toy", seed, spurious=0.0); y = data.labels.numpy()
bank = transforms.build(str(ROOT / "configs" / "transforms" / "toy_oracle.yml"))
tr = Trainer(cfg, data.images, bank, device="cpu", eval_labels=data.labels)
files = sorted(glob.glob(run_dir + "/ckpts/ep*.pt")); eps = [int(os.path.basename(f)[2:5]) for f in files]
metrics = {r["epoch"] + 1: r for r in map(json.loads, open(run_dir + "/metrics.jsonl"))}
cks = {e: torch.load(f, map_location="cpu", weights_only=False) for e, f in zip(eps, files)}
def load(e, policy_from=None):
    ck = cks[e]
    for k in ("encoder", "intent", "reward", "value"): getattr(tr, k).load_state_dict(ck[k]); getattr(tr, k).eval()
    tr.policy.load_state_dict(cks[policy_from or e]["policy"]); tr.policy.eval()
    tr.memory.assign = ck["assign"].clone()
def reps():
    with torch.no_grad(): m = tr.intents()
    tr.memory.recompute(m); return m.numpy()
def cstats(m):
    C = np.stack([m[y == c].mean(0) for c in range(5)]); Cc = C - C.mean(0)
    D = np.linalg.norm(C[:, None] - C[None], axis=-1)
    var = Cc.var(0); return C, Cc, D, var
M, P = {}, {}
for e in eps:
    load(e); M[e] = reps(); P[e] = tr.predict().numpy()
print("epoch | L_cov(logged) | acc | pairwise-dist change | raw centroid change | after best rotation | max principal angle (deg) | per-image shift / class dist | policy(t-1) on rep(t): acc, share of images with another cluster than policy(t)")
prev = None
for e in eps:
    m = M[e]; C, Cc, D, var = cstats(m); acc = cluster_acc(P[e], y, cfg.num_slots)
    line = f"{e:3d} | {metrics[e]['L_cov']:.3f} | {acc:.2f} |"
    if prev is not None:
        Cp, Ccp, Dp, varp = cstats(M[prev])
        dD = np.linalg.norm(D - Dp) / np.linalg.norm(Dp)
        raw = np.linalg.norm(Cc - Ccp) / np.linalg.norm(Cc)
        U, S, Vt = np.linalg.svd(Ccp.T @ Cc); R = U @ Vt; rot = np.linalg.norm(Cc - Ccp @ R) / np.linalg.norm(Cc)
        qa = np.linalg.qr(Ccp.T)[0][:, :4]; qb = np.linalg.qr(Cc.T)[0][:, :4]
        ang = np.degrees(np.arccos(np.clip(np.linalg.svd(qa.T @ qb)[1], -1, 1))).max()
        shift = np.linalg.norm(m - M[prev], axis=1).mean() / D[np.triu_indices(5, 1)].mean()
        load(e, policy_from=prev); reps(); pc = tr.predict().numpy()
        line += f" {dD:.3f} | {raw:.3f} | {rot:.3f} | {ang:5.1f} | {shift:.3f} | {cluster_acc(pc, y, cfg.num_slots):.2f}, {np.mean(pc != P[e]):.3f}"
    print(line); prev = e
# axis alignment of the code: share of centroid variance on the 4 busiest coordinates
print("\nepoch | share of the class-code variance on its 4 busiest coordinates | busiest coordinate indices")
for e in eps:
    _, Cc, _, var = cstats(M[e]); o = np.argsort(-var); print(f"{e:3d} | {var[o[:4]].sum() / var.sum():.3f} | {sorted(o[:4].tolist())}")
