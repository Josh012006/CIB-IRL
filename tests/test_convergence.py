"""The stopping rule: label-free, strict about its config, and wired into Trainer.fit.
Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cibirl.convergence import ConvergenceMonitor, StopConfig, partition_change    # noqa: E402

N = 200


def row(epoch, twin=0.8, l_pos=0.5, l_neg=0.3):
    return {"epoch": epoch, "twin_acc": twin, "L_pos": l_pos, "L_neg": l_neg}


def run(cfg, pred_fn, row_fn=row, epochs=200):
    mon = ConvergenceMonitor(cfg)
    for e in range(epochs):
        if mon.observe(row_fn(e), pred_fn(e)):
            return mon, e + 1
    raise AssertionError("never stopped: the monitor must stop at max_epochs at the latest")


def test_config_is_strict_and_casts():
    cfg = StopConfig.from_dict({"part_tol": "1e-2", "max_epochs": 50.0, "enabled": False})
    assert cfg.part_tol == 0.01 and cfg.max_epochs == 50 and cfg.enabled is False
    with pytest.raises(KeyError, match="did you mean 'window'"):
        StopConfig.from_dict({"windw": 3})
    with pytest.raises(ValueError):
        StopConfig.from_dict({"window": 0})
    with pytest.raises(ValueError):
        StopConfig.from_dict({"enabled": 1})
    assert StopConfig.from_dict(None) == StopConfig()


def test_stable_run_stops_as_soon_as_allowed():
    cfg = StopConfig(max_epochs=100, min_epochs=20, window=5)
    stable = np.arange(N) % 4
    mon, stopped_at = run(cfg, lambda e: stable)
    assert mon.converged and mon.reason == "converged" and stopped_at == 20     # min_epochs is the binding constraint
    cfg = StopConfig(max_epochs=100, min_epochs=3, window=5)
    mon, stopped_at = run(cfg, lambda e: stable)
    assert stopped_at == 11 and mon.converged                                   # two windows + the first epoch have no predecessor


def test_a_moving_partition_is_never_converged():
    rng = np.random.default_rng(0)
    mon, stopped_at = run(StopConfig(max_epochs=60, min_epochs=10), lambda e: rng.integers(0, 4, N))
    assert not mon.converged and mon.reason == "max_epochs" and stopped_at == 60


def test_small_partition_change_is_tolerated_up_to_part_tol():
    base = np.arange(N) % 4

    def flip(k):
        def f(e):
            p = base.copy()
            p[:k * (e % 2)] = 3 - p[:k * (e % 2)]       # k images change cluster every other epoch (cluster count unchanged)
            return p
        return f
    cfg = StopConfig(max_epochs=80, min_epochs=10, part_tol=0.01)
    assert run(cfg, flip(2))[0].converged                # 1 % of 200
    assert not run(cfg, flip(6))[0].converged            # 3 %


def test_changing_cluster_count_delays_convergence_by_a_window():
    def pred(e):
        p = np.arange(N) % 4
        if e == 30:
            p[0] = 9                                     # one image alone in a new cluster, for one epoch
        return p
    cfg = StopConfig(max_epochs=100, min_epochs=31, window=5, part_tol=0.05)
    mon, stopped_at = run(cfg, pred)
    assert mon.converged and stopped_at == 36            # allowed from epoch 31, but the odd epoch stays in the window until 35


def test_still_improving_objective_blocks_convergence():
    stable = np.arange(N) % 4
    cfg = StopConfig(max_epochs=60, min_epochs=10, window=5)
    rising = lambda e: row(e, twin=0.3 + 0.01 * e)       # +0.05 per window > twin_tol
    assert not run(cfg, lambda e: stable, rising)[0].converged
    falling = lambda e: row(e, l_pos=2.0 * 0.95 ** e)    # loss still dropping fast
    assert not run(cfg, lambda e: stable, falling)[0].converged


def test_missing_or_nan_metrics_never_trigger_a_stop():
    stable = np.arange(N) % 4
    cfg = StopConfig(max_epochs=40, min_epochs=10)
    assert not run(cfg, lambda e: stable, lambda e: {"epoch": e})[0].converged
    assert not run(cfg, lambda e: stable, lambda e: row(e, l_pos=float("nan")))[0].converged


def test_disabled_monitor_runs_exactly_max_epochs():
    mon, stopped_at = run(StopConfig(max_epochs=25, min_epochs=1, enabled=False), lambda e: np.zeros(N, dtype=int))
    assert stopped_at == 25 and not mon.converged and mon.reason == "fixed_epochs"


def test_diagnostics_are_written_into_the_row():
    mon = ConvergenceMonitor(StopConfig(max_epochs=5, min_epochs=5))
    r0, r1 = row(0), row(1)
    mon.observe(r0, np.array([0, 0, 1, 1]))
    mon.observe(r1, np.array([0, 1, 1, 1]))
    assert r0["part_change"] == 1.0 and r1["part_change"] == 0.25 and r1["slot_change"] == 0.25 and r1["n_clusters"] == 2


def test_trainer_fit_stops_when_the_callback_says_so():
    import datasets
    from cibirl.train import TrainConfig, Trainer
    d = datasets.build("toy", seed=0, n_classes=3, n_per_class=20)
    cfg = TrainConfig.from_yaml(ROOT / "configs" / "toy.yml", num_slots=3, expert_fraction=0.5, num_expert_episodes=20,
                                episodes_per_batch=4, learner_batch=20, reward_warmup_iters=2, beta_warmup_iters=2)
    tr = Trainer(cfg, d.images, datasets.background_redraw)
    seen = []
    history = tr.fit(10, log_fn=lambda r: seen.append(r["epoch"]) or r["epoch"] == 1)
    assert len(history) == 2 and seen == [0, 1]
    assert len(tr.fit(2, log_fn=lambda r: None)) == 2     # a callback returning None never stops (old behaviour)


def test_partition_change_ignores_renaming_but_not_real_moves():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert partition_change(a, np.array([2, 2, 0, 0, 1, 1])) == 0.0           # same partition, other slot ids
    assert abs(partition_change(a, np.array([0, 0, 1, 1, 1, 2])) - 1 / 6) < 1e-12   # one image moved
    assert abs(partition_change(a, np.array([0, 0, 0, 0, 2, 2])) - 2 / 6) < 1e-12   # two clusters merged: the smaller one counts
    assert partition_change(np.zeros(5, dtype=int), np.zeros(5, dtype=int)) == 0.0   # single cluster, both sides


def test_clusters_swapping_slot_ids_do_not_block_convergence():
    base = np.arange(N) % 4

    def swapping(e):
        return np.where(base < 2, base, base) if e % 2 else np.where(base < 2, 1 - base, base)   # slots 0 and 1 trade ids
    mon, stopped_at = run(StopConfig(max_epochs=60, min_epochs=15, window=5), swapping)
    assert mon.converged and stopped_at == 15
    assert mon.rows[-1]["slot_change"] > 0.4 and mon.rows[-1]["part_change"] == 0.0


def test_the_stop_decision_never_depends_on_evaluation_keys():
    stable = np.arange(N) % 4
    cfg = StopConfig(max_epochs=100, min_epochs=20, window=5)
    rng = np.random.default_rng(0)
    plain = run(cfg, lambda e: stable)[1]
    noisy = run(cfg, lambda e: stable, lambda e: dict(row(e), eval_acc=rng.random(), eval_nmi_class=rng.random()))[1]
    assert plain == noisy
