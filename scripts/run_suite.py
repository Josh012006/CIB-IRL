"""A suite of experiments x seeds from a YAML file, run in parallel, then the table mean +- std.

    python scripts/run_suite.py configs/suites/toy_regimes.yml --cpu-count 3
    python scripts/run_suite.py configs/suites/toy_regimes.yml --seeds 0 1 --only generic_spur0.9 --dry-run
    python scripts/run_suite.py configs/suites/toy_regimes.yml --skip-existing        # resume after an interruption

Each run trains until convergence (or max_epochs) with scripts/run_experiment.py, in its own process; its console
output goes to <out-root>/_logs/<name>.log and its results to <out-root>/<name>/. The table can be printed again with
    python -m eval.aggregate <out-root>

Suite file (all keys optional except experiments):

    name: toy_regimes                  # default out-root: results/<name>
    seeds: [0, 1, 2, 3, 4]
    common:                            # merged under every experiment (dicts are merged, the experiment wins)
      data: toy
      stop: {max_epochs: 300}
    experiments:
      - name: generic_spur0.9          # runs are named <name>_seed<k>
        transform: generic             # a name in configs/transforms/ or a path
        data_params: {spurious: 0.9}   # dataset parameters
        configs: [configs/toy.yml, configs/ablations/no_inv.yml]    # TrainConfig files (default: the dataset's)
        set: {gamma: 0}                # TrainConfig overrides
        stop: {max_epochs: 500}        # stopping rule (cibirl/convergence.py)
        fixed_epochs: 30               # run exactly N epochs instead
        seeds: [2, 4]                  # seeds of this experiment only (default: the suite's `seeds`)
        method: simclr                 # cibirl (default: scripts/run_experiment.py) or a baseline: kmeans, gmm, simclr, iic
                                       # (scripts/run_baseline.py; `configs` are then BaselineConfig files and `set` their keys)

`seeds` is a list, or a range written as a string: "0-199" is 0, 1, ..., 199 (also "0-9,20-22").
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _parallel import Job, run_jobs, say          # noqa: E402
from eval.aggregate import main as aggregate      # noqa: E402

SUITE_KEYS = {"name", "description", "seeds", "out_root", "common", "experiments"}
EXPERIMENT_KEYS = {"name", "data", "data_params", "transform", "configs", "set", "stop", "fixed_epochs", "seeds", "method"}
DICT_KEYS = {"data_params", "set", "stop"}
BASELINES = ("kmeans", "gmm", "simclr", "iic")           # run by scripts/run_baseline.py; any other method is our trainer
NO_BANK = ("kmeans", "gmm")                              # baselines that never draw a view: no transformation bank needed


def parse_seeds(value):
    """A list of integers, or a string "0-199" / "0-9,20-22". Returns None if the value is neither."""
    if isinstance(value, str):
        try:
            out = []
            for part in value.split(","):
                lo, _, hi = part.strip().partition("-")
                out += list(range(int(lo), int(hi or lo) + 1))
            return out or None
        except ValueError:
            return None
    return value if isinstance(value, list) and value and all(isinstance(s, int) for s in value) else None


def load_suite(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or not isinstance(raw.get("experiments"), list) or not raw["experiments"]:
        raise SystemExit(f"{path}: a suite is a mapping with a non-empty 'experiments' list")
    extra = set(raw) - SUITE_KEYS
    if extra:
        raise SystemExit(f"{path}: unknown key(s) {sorted(extra)} (allowed: {sorted(SUITE_KEYS)})")
    raw.setdefault("name", path.stem)
    if "seeds" in raw:
        raw["seeds"] = parse_seeds(raw["seeds"])
        if raw["seeds"] is None:
            raise SystemExit(f"{path}: 'seeds' must be a non-empty list of integers (or a range like \"0-199\")")
    common = raw.get("common") or {}
    experiments, seen = [], set()
    for exp in raw["experiments"]:
        extra = set(exp) - EXPERIMENT_KEYS
        if extra:
            raise SystemExit(f"{path}: experiment {exp.get('name')!r} has unknown key(s) {sorted(extra)} "
                             f"(allowed: {sorted(EXPERIMENT_KEYS)})")
        seeds = exp.get("seeds")
        if seeds is not None:
            exp = {**exp, "seeds": parse_seeds(seeds)}
            if exp["seeds"] is None:
                raise SystemExit(f"{path}: experiment {exp.get('name')!r}: 'seeds' must be a non-empty list of integers (or a range like \"0-199\")")
        merged = {k: v for k, v in common.items() if k not in DICT_KEYS}
        for k in DICT_KEYS:
            merged[k] = {**(common.get(k) or {}), **(exp.get(k) or {})}
        merged.update({k: v for k, v in exp.items() if k not in DICT_KEYS})
        method = merged.get("method", "cibirl")
        if method not in ("cibirl", *BASELINES):
            raise SystemExit(f"{path}: experiment {merged.get('name')!r}: unknown method {method!r} (cibirl, {', '.join(BASELINES)})")
        if "name" not in merged or ("transform" not in merged and method not in NO_BANK):
            raise SystemExit(f"{path}: every experiment needs a 'name' and a 'transform'")
        if merged["name"] in seen:
            raise SystemExit(f"{path}: duplicate experiment name {merged['name']!r}")
        seen.add(merged["name"])
        experiments.append(merged)
    raw["experiments"] = experiments
    return raw


def kv(d: dict) -> list[str]:
    """{'a': 1, 'b': 'x'} -> ['a=1', 'b="x"'] (JSON is valid YAML, so the runner reads it back unchanged)."""
    return [f"{k}={json.dumps(v)}" for k, v in d.items()]


def baseline_command(exp: dict, seed: int, name: str, out_root: Path, max_epochs: int | None) -> list[str]:
    cmd = [sys.executable, str(ROOT / "scripts" / "run_baseline.py"), "--method", exp["method"], "--data", exp.get("data", "toy"),
           "--name", name, "--out-root", str(out_root), "--overwrite"]
    if exp.get("transform"):
        cmd += ["--transform", str(exp["transform"])]
    if exp["data_params"]:
        cmd += ["--data-set", *kv(exp["data_params"])]
    if exp.get("configs"):
        cmd += ["--config", *exp["configs"]]
    overrides = dict(exp["set"])
    if max_epochs is not None and exp["method"] in ("simclr", "iic"):
        overrides["epochs"] = max_epochs                        # quick checks: --max-epochs also shortens the contrastive training
    return cmd + ["--set", f"seed={seed}", *kv(overrides)]


def command(exp: dict, seed: int, name: str, out_root: Path, max_epochs: int | None) -> list[str]:
    if exp.get("method", "cibirl") in BASELINES:
        return baseline_command(exp, seed, name, out_root, max_epochs)
    cmd = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--data", exp.get("data", "toy"),
           "--transform", str(exp["transform"]), "--name", name, "--out-root", str(out_root), "--overwrite"]
    if exp["data_params"]:
        cmd += ["--data-set", *kv(exp["data_params"])]
    if exp.get("configs"):
        cmd += ["--config", *exp["configs"]]
    cmd += ["--set", f"seed={seed}", *kv(exp["set"])]
    stop = dict(exp["stop"])
    if max_epochs is not None:
        stop["max_epochs"] = max_epochs
    if exp.get("fixed_epochs") is not None:
        cmd += ["--fixed-epochs", str(exp["fixed_epochs"])]
    elif stop:
        cmd += ["--stop-set", *kv(stop)]
    return cmd


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", help="suite YAML file")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override the seeds of the file")
    ap.add_argument("--only", nargs="+", default=None, metavar="NAME", help="run only these experiments")
    ap.add_argument("--max-epochs", type=int, default=None, help="override max_epochs (quick checks)")
    ap.add_argument("--out-root", default=None, help="default: results/<suite name>")
    ap.add_argument("--skip-existing", action="store_true", help="skip runs that already have final_metrics.json")
    ap.add_argument("--cpu-count", type=int, default=1, metavar="N", help="number of runs executed in parallel (default 1)")
    ap.add_argument("--threads-per-run", type=int, default=None,
                    help="torch threads of each run (default: machine cores // cpu-count, at least 1)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and stop")
    args = ap.parse_args()
    if args.cpu_count < 1:
        ap.error("--cpu-count must be >= 1")

    suite_path = Path(args.suite) if Path(args.suite).exists() else ROOT / args.suite
    suite = load_suite(suite_path)
    default_seeds = suite.get("seeds") or [0]
    experiments = suite["experiments"]
    if args.only:
        unknown = set(args.only) - {e["name"] for e in experiments}
        if unknown:
            ap.error(f"unknown experiment(s) {sorted(unknown)}; the suite has {[e['name'] for e in experiments]}")
        experiments = [e for e in experiments if e["name"] in args.only]
    out_arg = args.out_root or suite.get("out_root") or f"results/{suite['name']}"
    out_root = Path(out_arg) if Path(out_arg).is_absolute() else ROOT / out_arg

    def seeds_of(exp):
        return args.seeds or exp.get("seeds") or default_seeds

    seeds = list(dict.fromkeys(s for exp in experiments for s in seeds_of(exp)))    # order of first appearance
    jobs = []
    for seed in seeds:
        for exp in experiments:
            if seed not in seeds_of(exp):
                continue
            name = f"{exp['name']}_seed{seed}"
            if args.skip_existing and (out_root / name / "final_metrics.json").exists():
                say(f"skip   {name} (already done)")
                continue
            jobs.append(Job(name, command(exp, seed, name, out_root, args.max_epochs), out_root / "_logs" / f"{name}.log"))

    if args.dry_run:
        for j in jobs:
            print(" ".join(j.cmd))
        print(f"\n{len(jobs)} runs would be launched in {out_root}")
        sys.exit(0)

    failed = run_jobs(jobs, ROOT, args.cpu_count, args.threads_per_run)
    print()
    aggregate(str(out_root))
    if failed:
        say(f"\n{len(failed)} run(s) failed: {', '.join(failed)} (logs in {out_root / '_logs'})")
        sys.exit(1)
