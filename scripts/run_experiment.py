"""One run of the whole pipeline: a dataset, a transformation bank, a config, and a stopping rule.

    python scripts/run_experiment.py --data toy --transform generic
    python scripts/run_experiment.py --data toy --data-set spurious=0.9 --transform informed_color --set seed=3
    python scripts/run_experiment.py --data toy --transform toy_oracle --config configs/toy.yml configs/ablations/no_inv.yml
    python scripts/run_experiment.py --data toy --transform generic --fixed-epochs 30        # no convergence test
    python scripts/run_experiment.py --data toy --transform generic --stop-set max_epochs=500 min_epochs=60

By default training runs UNTIL CONVERGENCE (cibirl/convergence.py: stable argmax partition, constant number of
clusters, plateau of the expert objective; no label is used) or until `max_epochs`.

Writes <out-root>/<name>/ :
    config.yml           resolved TrainConfig (defaults + files + --set)
    transform.yml        the transformation bank as it was used
    run.yml              dataset and parameters, stopping rule, command line
    metrics.jsonl        one line per epoch (losses, twin_acc, churn, part_change, n_clusters...; eval_* = labels,
                         for monitoring only)
    final_metrics.json   ACC, NMI(cluster, class), NMI(cluster, background), epochs_run, converged, stop_reason
    preds.npy            slot of every image under the policy with the memory frozen
    intents.npy          m of every image
    checkpoint.pt        networks + memory
    figures/             curves.png, clusters.png (+ clusters_epXXX.png snapshots), confusion.png, embedding.png

The model that is evaluated is the one of the LAST epoch: no epoch is picked by looking at the labels.
"""
import argparse
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets                                                     # noqa: E402
from cibirl import transforms                                       # noqa: E402
from cibirl.cleanup import clone_memory, merge_twin_slots, reintroduce, relabel, use_memory   # noqa: E402
from cibirl.convergence import ConvergenceMonitor, StopConfig, partition_change   # noqa: E402
from cibirl.train import TrainConfig, Trainer                       # noqa: E402
from eval import visualize as viz                                   # noqa: E402
from eval.metrics import cluster_acc, nmi                           # noqa: E402


def parse_kv(items) -> dict:
    """['a=1', 'b=[1, 2]', 'c=true'] -> {'a': 1, 'b': [1, 2], 'c': True} (values read as YAML)."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"expected KEY=VALUE, got '{item}'")
        k, v = item.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def resolve_file(p: str) -> Path:
    """cwd first, then the repo root (so that commands work from any folder)."""
    return Path(p) if Path(p).exists() else ROOT / p


def resolve_bank(arg: str) -> Path:
    """A bank is a file path or a name in configs/transforms/."""
    if arg.endswith((".yml", ".yaml")) or Path(arg).exists():
        return resolve_file(arg)
    path = ROOT / "configs" / "transforms" / f"{arg}.yml"
    if not path.exists():
        names = sorted(p.stem for p in (ROOT / "configs" / "transforms").glob("*.yml"))
        raise SystemExit(f"unknown transform bank '{arg}'. Available: {names}")
    return path


def default_name(data: str, data_params: dict, bank: str, configs: list[str], seed: int) -> str:
    parts = [data] + [f"{k}{v:g}" if isinstance(v, float) else f"{k}{v}" for k, v in data_params.items()]
    parts.append(bank)
    parts += [Path(c).stem for c in configs[1:]]                    # ablation files
    parts.append(f"seed{seed}")
    return "_".join(parts)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="toy", help=f"dataset: {sorted(datasets.DATASETS)}")
    ap.add_argument("--data-set", nargs="*", default=[], metavar="KEY=VALUE", help="dataset parameters, e.g. spurious=0.9")
    ap.add_argument("--transform", required=True, help="transformation bank: a name in configs/transforms/ or a .yml path")
    ap.add_argument("--config", nargs="+", default=None,
                    help="TrainConfig YAML files merged left to right (default: the dataset's default config)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="TrainConfig overrides, e.g. gamma=0 seed=3")
    ap.add_argument("--stop-set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="stopping rule, e.g. max_epochs=500 min_epochs=60 (see cibirl/convergence.py)")
    ap.add_argument("--fixed-epochs", type=int, default=None, help="run exactly N epochs, no convergence test")
    ap.add_argument("--device", default="cpu", help="cpu (default: faster on tiny models) or cuda")
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--name", default=None, help="run folder name (default built from the options)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run folder")
    ap.add_argument("--snapshot-every", type=int, default=10, help="save a cluster grid every N epochs (0 = never)")
    ap.add_argument("--ckpt-every", type=int, default=0, metavar="N",
                    help="write <run>/resume_state.pt (the FULL state: networks, optimizers, memory, random numbers, convergence "
                         "monitor) every N epochs and when training stops, so that the run can be continued with --resume. "
                         "0 = never (default). Does not touch the training.")
    ap.add_argument("--resume", action="store_true",
                    help="continue the run <out-root>/<name> from its resume_state.pt (same --data/--data-set/--transform/--config/"
                         "--set as the original command; give the NEW total in --stop-set max_epochs=...). The final files of the "
                         "previous stop are moved to <run>/at_ep<N>/.")
    ap.add_argument("--save-epochs", default=None, metavar="A-B",
                    help="also save the networks and the memory at the end of epochs A..B (1-based, e.g. 195-212) in ckpts/epNNN.pt, "
                         "to study what happens around an event. Does not touch the training (no random number is drawn).")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    data_params, overrides, stop_overrides = parse_kv(args.data_set), parse_kv(args.set), parse_kv(args.stop_set)
    if args.data not in datasets.DATASETS:
        raise SystemExit(f"unknown dataset '{args.data}'. Available: {sorted(datasets.DATASETS)}")
    config_files = [resolve_file(c) for c in (args.config or [datasets.DEFAULT_CONFIG[args.data]])]
    cfg = TrainConfig.from_yaml(*config_files, **overrides)
    stop = StopConfig.from_dict(stop_overrides)
    if args.fixed_epochs is not None:
        stop = replace(stop, max_epochs=args.fixed_epochs, enabled=False)
    bank = transforms.build(resolve_bank(args.transform))
    data = datasets.build(args.data, cfg.seed, **data_params)
    images, y = data.images, data.labels.numpy()
    bg = data.background.numpy() if data.background is not None else None
    if images.shape[1] != cfg.in_channels or images.shape[2] != cfg.image_size or images.shape[3] != cfg.image_size:
        raise SystemExit(f"data are {tuple(images.shape[1:])} but the config says in_channels={cfg.in_channels}, "
                         f"image_size={cfg.image_size}")

    name = args.name or default_name(args.data, data_params, bank.name, args.config or [""], cfg.seed)
    out = Path(args.out_root) if Path(args.out_root).is_absolute() else ROOT / args.out_root
    out = out / name
    state_file = out / "resume_state.pt"
    if args.resume:
        if not state_file.exists():
            sys.exit(f"--resume: {state_file} does not exist (the run was not made with --ckpt-every)")
        if args.overwrite:
            sys.exit("--resume and --overwrite are incompatible")
        saved_cfg = TrainConfig.from_yaml(out / "config.yml")
        if saved_cfg != cfg:
            sys.exit(f"--resume: the configuration differs from {out / 'config.yml'}; repeat the options of the original command")
    else:
        if out.exists():
            if not args.overwrite:
                sys.exit(f"{out} already exists: choose --name, or pass --overwrite")
            shutil.rmtree(out)
        (out / "figures").mkdir(parents=True)
        cfg.to_yaml(out / "config.yml")
        (out / "transform.yml").write_text(yaml.safe_dump(bank.spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    run_info = {"data": args.data, "data_params": data.info, "transform": bank.name, "stop": stop.to_dict(),
                "device": args.device, "uses_labels_for_training": cfg.expert_oracle_link > 0 or cfg.oracle_warm_memory > 0,
                "command": " ".join(sys.argv)}
    if args.resume:                                   # keep the original run.yml, add the continuation
        prev = yaml.safe_load((out / "run.yml").read_text(encoding="utf-8"))
        prev.setdefault("resumed", []).append({"stop": stop.to_dict(), "command": run_info["command"]})
        run_info = prev
    (out / "run.yml").write_text(yaml.safe_dump(run_info, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"run folder: {out}\n{bank!r}\nstop: {stop}", flush=True)

    oracle = cfg.expert_oracle_link > 0 or cfg.oracle_warm_memory > 0
    if oracle:
        print("WARNING: expert_oracle_link / oracle_warm_memory use the TRUE CLASSES to train. "
              "Diagnostic only, not a result of the method.", flush=True)
    # eval_labels only feed the eval_* placement statistics (cibirl/diagnostics.py), never the training
    tr = Trainer(cfg, images, bank, device=args.device, oracle_labels=data.labels if oracle else None,
                 eval_labels=data.labels)
    monitor = ConvergenceMonitor(stop)
    save_lo, save_hi = (None, None)
    if args.save_epochs:
        lo, _, hi = args.save_epochs.partition("-")
        save_lo, save_hi = int(lo), int(hi or lo)
        (out / "ckpts").mkdir(exist_ok=True)
    history: list[dict] = []
    start_epoch = 0
    if args.resume:
        monitor.load_state_dict(tr.load_state(str(state_file)).pop("monitor"))
        history = list(monitor.rows)
        start_epoch = len(history)
        mfile = out / "metrics.jsonl"                  # a run that crashed after its last state has epochs the state does not: drop them
        mfile.write_text("".join(l + "\n" for l in mfile.read_text(encoding="utf-8").splitlines()[:start_epoch]), encoding="utf-8")
        if start_epoch >= stop.max_epochs:
            sys.exit(f"--resume: the run already has {start_epoch} epochs; give a larger max_epochs in --stop-set")
        monitor.converged, monitor.reason = False, "running"
        old_final = out / f"at_ep{start_epoch}"        # the files of the previous stop are kept, not overwritten
        old_final.mkdir(exist_ok=True)
        for f in ("final_metrics.json", "preds.npy", "intents.npy", "checkpoint.pt", "variants.npz", "memory_pre.pt"):
            if (out / f).exists():
                shutil.move(str(out / f), str(old_final / f))
        if (out / "figures").exists():
            shutil.copytree(out / "figures", old_final / "figures", dirs_exist_ok=True)
        print(f"resumed from epoch {start_epoch}", flush=True)
    (out / "figures").mkdir(exist_ok=True)
    metrics_file = open(out / "metrics.jsonl", "a" if args.resume else "w", encoding="utf-8")
    t0 = time.time()

    def write_state() -> None:
        tr.save_state(str(state_file), extra={"monitor": monitor.state_dict()})

    def log(row: dict) -> bool:
        pred = tr.predict().cpu().numpy()
        stop_now = monitor.observe(row, pred, tr.memory.assign.cpu().numpy())   # adds part_change, n_clusters, mem_gap to the row
        # EVALUATION ONLY: the labels are used to follow the dynamics, never to train nor to stop (the monitor
        # does not read these keys)
        row["eval_acc"], row["eval_nmi_class"] = cluster_acc(pred, y, cfg.num_slots), nmi(pred, y)
        if bg is not None:
            row["eval_nmi_background"] = nmi(pred, bg)
        history.append(row)
        if save_lo is not None and save_lo <= row["epoch"] + 1 <= save_hi:
            tr.save(str(out / "ckpts" / f"ep{row['epoch'] + 1:03d}.pt"))
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()
        if args.ckpt_every and (row["epoch"] + 1) % args.ckpt_every == 0 and not stop_now:
            write_state()                                    # after the monitor and the log: the state of the end of the epoch
        print(f"ep {row['epoch']:3d} it {row['iter']:5d} | L_pos {row.get('L_pos', 0):5.3f} L_neg {row.get('L_neg', 0):5.3f} "
              f"twin {row.get('twin_acc', 0):.2f} KL {row.get('KL', 0):6.1f} L_inv {row.get('L_inv', 0):.3f} "
              f"m_std {row.get('m_std', 0):.2f} | ent {row.get('pi_entropy', 0):.2f} | clusters {row['n_clusters']} "
              f"K_eff {row['K_eff']:.2f} churn {row['churn']:.2f} dpart {row['part_change']:.3f} gap {row['mem_gap']:.3f} "
              f"| acc* {row['eval_acc']:.2f} | {time.time() - t0:5.0f}s", flush=True)
        e = args.snapshot_every
        if e and not stop_now and row["epoch"] % e == e - 1:
            viz.plot_cluster_grid(images, pred, out / "figures" / f"clusters_ep{row['epoch'] + 1:03d}.png", cfg.num_slots,
                                  labels=y, title=f"Clusters formed by the policy, after epoch {row['epoch'] + 1}")
        return stop_now

    tr.fit(stop.max_epochs, log_fn=log, start_epoch=start_epoch)
    metrics_file.close()
    if args.ckpt_every:
        write_state()                                        # the state at the stop, BEFORE the final clean-up: a run can be prolonged
    print(f"\nstopped after {len(history)} epochs: {monitor.reason}", flush=True)

    # ---- final clustering: argmax of the policy with the memory frozen (README 6.7) ----------------
    pred = tr.predict().cpu().numpy()
    m = tr.intents().cpu().numpy()
    cleanup = None
    if cfg.final_cleanup:
        # last step, nothing is trained (docs 5.28, 5.29). Every way of closing the memory starts from the SAME trained model and
        # the same memory (copies), so they are compared on the same runs; `cleanup_mode` says which one gives the output.
        pred_pre = pred.copy()
        mem_pre = clone_memory(tr.memory)
        mem_before = mem_pre.assign.cpu().numpy().copy()
        m_t = torch.from_numpy(m).to(tr.memory.device)

        def describe(p, mem):
            ma = mem.assign.cpu().numpy()
            return {"acc": cluster_acc(p, y, cfg.num_slots), "clusters": int(len(np.unique(p))),
                    "clusters_ge10": int(sum(1 for q in np.unique(p) if (p == q).sum() >= 10)),
                    "memory_clusters": int(len(np.unique(ma[ma >= 0]))), "memory_acc": cluster_acc(ma, y, cfg.num_slots),
                    "mem_gap": partition_change(ma, p)}
        before = describe(pred_pre, mem_pre)
        variants: dict[str, tuple] = {}
        # merge: merge the twin slots, then read the output again on the merged memory
        mem_m = clone_memory(mem_pre)
        info = merge_twin_slots(mem_m, m_t, cfg.cleanup_ratio)
        with use_memory(tr, mem_m):
            variants["merge"] = (tr.predict().cpu().numpy(), mem_m, {})
        # A, relabel: the output read before, with its slots renamed by the merge
        variants["relabel"] = (relabel(torch.from_numpy(pred_pre), info["rep"]).numpy(), clone_memory(mem_m), {})
        # C, reintroduce: sweep the images from the memory of before (no distance, no guard). The OUTPUT of C is the partition
        # the sweeps end in (a fixed point of the policy when a sweep moves nothing); `predict_*` is the policy read once more on
        # that memory, which can open a cluster again once empty slots exist (kept for information).
        for name, allow in (("reintroduce", True), ("reintroduce_no_open", False)):
            mem_c = clone_memory(mem_pre)
            with use_memory(tr, mem_c):
                rinfo = reintroduce(tr, allow_open=allow, seed=cfg.seed)
                p_read = tr.predict().cpu().numpy()
            variants[name] = (mem_c.assign.cpu().numpy().copy(), mem_c,
                              {"sweeps": rinfo["sweeps"], "moved": rinfo["moved"], "predict_acc": cluster_acc(p_read, y, cfg.num_slots),
                               "predict_clusters": int(len(np.unique(p_read)))})
        # EVALUATION ONLY: did a merge join two different classes?
        dominant = {int(k): int(np.bincount(y[mem_before == k], minlength=int(y.max()) + 1).argmax()) for k in np.unique(mem_before[mem_before >= 0])}
        wrong = sum(1 for k, r in info["rep"].items() if k != r and dominant[k] != dominant[r])
        described = {n: {**describe(p, mem), **extra} for n, (p, mem, extra) in variants.items()}
        np.savez(out / "variants.npz", pre=pred_pre, mem_pre=mem_before,
                 **{n: p for n, (p, _, _) in variants.items()}, **{f"mem_{n}": mem.assign.cpu().numpy() for n, (_, mem, _) in variants.items()})
        torch.save({"assign": mem_pre.assign, "member_m": mem_pre.member_m, "counts": mem_pre.counts, "summaries": mem_pre.summaries},
                   out / "memory_pre.pt")
        pred, chosen, _ = variants[cfg.cleanup_mode]
        tr.memory = chosen
        cleanup = {"before": before, "mode": cfg.cleanup_mode, "merged_slots": info["merged_pairs"], "jump": info["jump"],
                   "threshold": info["threshold"], "slot_distances": info["distances"], "merges_across_classes": wrong,
                   "after": described[cfg.cleanup_mode], "variants": described}
    final = {"name": out.name, "seed": cfg.seed, "data": args.data, **data.info, "transform": bank.name,
             "epochs_run": len(history), "converged": monitor.converged, "stop_reason": monitor.reason,
             "output_converged_epoch": monitor.output_converged_epoch, "memory_converged_epoch": monitor.memory_converged_epoch,
             "mem_gap": history[-1].get("mem_gap"), "memory_clusters": history[-1].get("n_clusters", 0) + history[-1].get("mem_excess", 0),
             "acc": cluster_acc(pred, y, cfg.num_slots), "nmi_class": nmi(pred, y)}
    if cleanup is not None:
        final["before_cleanup"] = cleanup["before"]
        final["cleanup"] = {k: v for k, v in cleanup.items() if k != "before"}
        final["mem_gap"], final["memory_clusters"] = cleanup["after"]["mem_gap"], cleanup["after"]["memory_clusters"]
    if oracle:
        final["oracle_labels_used"] = True                          # diagnostic run: not comparable to the method
    if bg is not None:
        final["nmi_background"] = nmi(pred, bg)
        final["nmi_class_background_in_data"] = nmi(y, bg)
    final["clusters_used"] = int(len(np.unique(pred)))
    (out / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    np.save(out / "preds.npy", pred)
    np.save(out / "intents.npy", m)
    tr.save(str(out / "checkpoint.pt"))

    fig = out / "figures"
    viz.plot_curves(history, fig / "curves.png")
    viz.plot_cluster_grid(images, pred, fig / "clusters.png", cfg.num_slots, labels=y,
                          title="Clusters formed by the policy (argmax, memory frozen)")
    viz.plot_confusion(pred, y, fig / "confusion.png", cfg.num_slots)
    viz.plot_embedding(m, pred, fig / "embedding.png", labels=y, background=bg)

    print("\n" + "\n".join(f"{k:>30s}: {v:.3f}" if isinstance(v, float) else f"{k:>30s}: {v}" for k, v in final.items()))
    print(f"\nfigures and metrics in {out}")


if __name__ == "__main__":
    main()
