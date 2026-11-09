"""C6 experiments: every number in docs/RESULTS.md comes from this script.

    python scripts/compare_models.py noise   # how noisy is an EER measured on 8 users?
    python scripts/compare_models.py dev     # encoder variants, development users only
    python scripts/compare_models.py stateless   # can a classic method fit the Embedder protocol?
    python scripts/compare_models.py engine  # the shipped checkpoint through the real engine, val users
    python scripts/compare_models.py final   # the only command that reads test users
    python scripts/compare_models.py table   # print the RESULTS.md tables from the JSON files

Results are aggregate metrics, written as JSON to checkpoints/c6/ (gitignored).

Test users stay out of every decision. `noise` and `dev` never score the 8
test users of split seed 0. `dev` splits the other 43 users into 27 train /
8 val / 8 holdout, so the val-to-holdout gap can be studied without reading a
test user. Training choices were made from those two commands, then frozen,
then `final` was run.

CMU only: there is no synthetic path here, so these numbers are always CMU numbers.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from keyauth.auth import EngineConfig, InvalidTimings, KeyAuthEngine, TemplateStore
from keyauth.auth.embedder import TorchEmbedder
from keyauth.baselines import Detector, Mahalanobis, ScaledManhattan, detector_scores, timing_vectors
from keyauth.data import KeystrokeData, Normalizer, load_cmu, split_users
from keyauth.data.preprocess import EPS_S, valid_mask
from keyauth.evaluate import summarize, verification_scores
from keyauth.model import RhythmEncoder
from keyauth.train import TrainConfig, embed, evaluate_split, load_checkpoint, train

ROOT = Path(__file__).resolve().parent.parent
CMU_PATH = ROOT / "data" / "DSL-StrongPasswordData.csv"
OUT_DIR = ROOT / "checkpoints" / "c6"

N_ENROLL = 10
THREADS_PER_WORKER = 2
FINAL_SEEDS = (0, 1, 2, 3, 4)
INNER_SPLITS = (1, 2, 3)
TRAIN_SEEDS = (0, 1, 2, 3, 4)
SHRINKAGE_GRID = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
NOISE_GROUPS = 200
NOISE_GROUP_SIZE = 8
EMA_ALPHA = 0.3

# The encoder as it was before C6, spelled out so it stays reproducible if defaults change.
BEFORE: dict = {"loss": "batch_hard", "lr": 1e-3, "epochs": 40, "patience": 10}

DEV_VARIANTS: dict[str, dict] = {
    "batch_hard": {"loss": "batch_hard"},
    "batch_hard_lr1e-4": {"loss": "batch_hard", "lr": 1e-4},
    "supcon_t0.1": {"loss": "supcon", "temperature": 0.1},
    "supcon_t0.3": {"loss": "supcon", "temperature": 0.3},
}
DEV_EPOCHS = 30


def dev_pool() -> KeystrokeData:
    """All CMU users except the test users of split seed 0."""
    data = load_cmu(CMU_PATH)
    _, _, locked = split_users(data, seed=0)
    return data.for_users([u for u in data.users if u not in set(locked.users)])


def write_json(name: str, payload: object) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=1))
    return path


def read_json(name: str) -> object:
    return json.loads((OUT_DIR / name).read_text())


def select_epoch(values: list[float], patience: int) -> int:
    """Index of the epoch early stopping keeps, using train.py's rule on any metric."""
    best = float("inf")
    best_i = 0
    stale = 0
    for i, v in enumerate(values):
        if v < best:
            best = v
            best_i = i
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    return best_i


def ema(values: list[float], alpha: float = EMA_ALPHA) -> list[float]:
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1.0 - alpha) * out[-1])
    return out


def mean_std(values: list[float]) -> tuple[float, float]:
    """Sample std (ddof=1); 0 for a single value."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


# ---- noise: sampling variance of an 8-user EER ---------------------------


def cmd_noise(args: argparse.Namespace) -> None:
    pool = dev_pool()
    users = np.array(pool.users, dtype=object)
    rng = np.random.default_rng(123)
    groups = [rng.choice(users, NOISE_GROUP_SIZE, replace=False) for _ in range(NOISE_GROUPS)]
    result: dict[str, dict] = {}
    detectors: dict[str, Callable[[], Detector]] = {
        "scaled_manhattan": ScaledManhattan,
        "mahalanobis_s0.9": lambda: Mahalanobis(shrinkage=0.9),
    }
    for name, make in detectors.items():
        eers = [summarize(detector_scores(make(), pool.for_users(g), N_ENROLL))["eer"] for g in groups]
        arr = np.array(eers)
        # consecutive independent draws act as (val, test) pairs
        gaps = np.abs(arr[::2] - arr[1::2])
        result[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "gap_mean": float(gaps.mean()),
            "gap_p90": float(np.percentile(gaps, 90)),
            "frac_gap_ge_11pts": float(np.mean(gaps >= 0.11)),
            "all_43_users_eer": summarize(detector_scores(make(), pool, N_ENROLL))["eer"],
        }
    print(write_json("noise.json", {"n_users_pool": len(users), "group_size": NOISE_GROUP_SIZE, "detectors": result}))


# ---- dev: encoder variants on development users ---------------------------


def dev_splits(inner: int) -> tuple[KeystrokeData, KeystrokeData, KeystrokeData | None]:
    """(train, val, holdout). inner=0 is the real seed-0 split with its test users dropped, so no holdout."""
    if inner == 0:
        train_data, val_data, _ = split_users(load_cmu(CMU_PATH), seed=0)
        return train_data, val_data, None
    return split_users(dev_pool(), val_frac=8 / 43, test_frac=8 / 43, seed=1000 + inner)


def dev_job(job: dict) -> dict:
    """One full training run with no early stopping; stopping rules are replayed afterwards from history."""
    torch.set_num_threads(THREADS_PER_WORKER)
    train_data, val_data, holdout = dev_splits(job["inner"])
    kwargs = {"epochs": DEV_EPOCHS, "patience": DEV_EPOCHS + 1, "n_enroll": N_ENROLL, "seed": job["train_seed"]}
    kwargs.update(job["config"])
    cfg = TrainConfig(**kwargs)
    fitted = Normalizer.fit(train_data)  # the same fit train() makes, needed by the hook

    def hook(row: dict, model: RhythmEncoder) -> None:
        if holdout is not None:
            row["holdout"] = evaluate_split(model, fitted, holdout, cfg, threshold=row["eer_threshold"])

    model, normalizer, history = train(train_data, val_data, cfg, log=lambda s: None, epoch_hook=hook)
    z = embed(model, normalizer, val_data)
    return {
        "variant": job["variant"],
        "inner": job["inner"],
        "train_seed": job["train_seed"],
        "history": history,
        "resultant": float(np.linalg.norm(z.mean(axis=0))),
    }


def dev_detector_job(inner: int) -> dict:
    """Classic detectors on the same val and holdout users as the encoder variants."""
    _, val_data, holdout = dev_splits(inner)
    out: dict = {"inner": inner}
    detectors: dict[str, Callable[[], Detector]] = {
        "scaled_manhattan": ScaledManhattan,
        "mahalanobis_s0.9": lambda: Mahalanobis(shrinkage=0.9),
    }
    for name, make in detectors.items():
        out[name] = {
            "val": summarize(detector_scores(make(), val_data, N_ENROLL))["eer"],
            "holdout": summarize(detector_scores(make(), holdout, N_ENROLL))["eer"],
        }
    return out


def cmd_dev(args: argparse.Namespace) -> None:
    jobs = [
        {"variant": name, "inner": inner, "train_seed": 0, "config": config}
        for name, config in DEV_VARIANTS.items()
        for inner in INNER_SPLITS
    ]
    for seed in TRAIN_SEEDS:
        for name in ("batch_hard", args.seed_variant):
            jobs.append({"variant": f"{name}@seed0split", "inner": 0, "train_seed": seed, "config": DEV_VARIANTS[name]})
    with ProcessPoolExecutor(args.workers) as pool:
        runs = list(pool.map(dev_job, jobs))
    print(write_json("dev.json", runs))
    print(write_json("dev_detectors.json", [dev_detector_job(inner) for inner in INNER_SPLITS]))


# ---- stateless: classic features squeezed through the Embedder protocol ------


def stateless_candidates(normalizer: Normalizer) -> dict[str, Callable[[KeystrokeData], np.ndarray]]:
    """Per-attempt embeddings that need no per-user state, which is all the Embedder protocol allows."""

    def unit(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def channel_normalized(d: KeystrokeData) -> np.ndarray:
        return unit(normalizer(d)[valid_mask(d.lengths, d.max_len)].reshape(len(d), -1))

    def speed_normalized(d: KeystrokeData) -> np.ndarray:
        t = timing_vectors(d)
        v = np.concatenate([np.log(np.maximum(t[:, 0::3], EPS_S)), np.log(np.maximum(t[:, 1::3], EPS_S))], axis=1)
        return unit(v - v.mean(axis=1, keepdims=True))

    def raw_seconds(d: KeystrokeData) -> np.ndarray:
        return unit(timing_vectors(d))

    return {
        "channel-normalized timings": channel_normalized,
        "log timings, speed-normalized": speed_normalized,
        "raw seconds": raw_seconds,
    }


def cmd_stateless(args: argparse.Namespace) -> None:
    pool = dev_pool()
    train_data, val_data, _ = split_users(load_cmu(CMU_PATH), seed=0)
    candidates = stateless_candidates(Normalizer.fit(train_data))
    rows: dict[str, dict] = {
        "per-user scaled Manhattan (reference, not an Embedder)": {
            "val_8_users": summarize(detector_scores(ScaledManhattan(), val_data, N_ENROLL))["eer"],
            "all_43_dev_users": summarize(detector_scores(ScaledManhattan(), pool, N_ENROLL))["eer"],
        }
    }
    for name, make in candidates.items():
        rows[name] = {
            "val_8_users": summarize(verification_scores(make(val_data), val_data.user_ids, N_ENROLL))["eer"],
            "all_43_dev_users": summarize(verification_scores(make(pool), pool.user_ids, N_ENROLL))["eer"],
        }
    print(write_json("stateless.json", rows))


# ---- engine: the shipped checkpoint through the real engine, val users only --


def cmd_engine(args: argparse.Namespace) -> None:
    _, val_data, _ = split_users(load_cmu(CMU_PATH), seed=0)
    embedder, threshold = TorchEmbedder.from_checkpoint(args.checkpoint)
    counts = {"refused": 0, "genuine": 0, "genuine_accepted": 0, "impostor": 0, "impostor_accepted": 0, "enroll_failed": 0}
    latencies: list[float] = []

    def attempts(user: str) -> list[np.ndarray]:
        rows = np.flatnonzero(val_data.user_ids == user)
        return [val_data.values[i, : val_data.lengths[i]] for i in rows]

    def attempt_login(engine: KeyAuthEngine, user: str, timings: np.ndarray, kind: str) -> None:
        start = time.perf_counter()
        try:
            result = engine.verify("app", user, timings)
        except InvalidTimings:
            counts["refused"] += 1
            return
        latencies.append(time.perf_counter() - start)
        counts[kind] += 1
        counts[f"{kind}_accepted"] += int(result.accepted)

    for user in val_data.users:
        config = EngineConfig(enroll_samples=N_ENROLL, max_failures=10**9, adapt=False)
        engine = KeyAuthEngine(embedder, TemplateStore(), threshold, config)
        enrolled = [engine.enroll("app", user, a) for a in attempts(user)[:N_ENROLL]]
        if enrolled[-1].status != "complete":
            counts["enroll_failed"] += 1
            continue
        for timings in attempts(user)[N_ENROLL:]:
            attempt_login(engine, user, timings, "genuine")
        for other in val_data.users:
            if other != user:
                for timings in attempts(other)[:5]:
                    attempt_login(engine, user, timings, "impostor")
    print(write_json("engine.json", {"threshold": threshold, "median_verify_ms": float(np.median(latencies) * 1000), **counts}))


# ---- final: multi-seed comparison, the only code that reads test users -----


def final_detector(detector: Detector, val_data: KeystrokeData, test_data: KeystrokeData) -> dict:
    val = summarize(detector_scores(detector, val_data, N_ENROLL))
    test = summarize(detector_scores(detector, test_data, N_ENROLL), threshold=val["eer_threshold"])
    return {"val": val, "test": test}


def final_job(job: dict) -> dict:
    torch.set_num_threads(THREADS_PER_WORKER)
    data = load_cmu(CMU_PATH)
    train_data, val_data, test_data = split_users(data, seed=job["seed"])
    out: dict = {"seed": job["seed"], "method": job["method"]}
    if job["method"] == "scaled_manhattan":
        out.update(final_detector(ScaledManhattan(), val_data, test_data))
    elif job["method"] == "mahalanobis":
        # shrinkage is picked on val users only
        val_eers = [summarize(detector_scores(Mahalanobis(shrinkage=s), val_data, N_ENROLL))["eer"] for s in SHRINKAGE_GRID]
        shrinkage = SHRINKAGE_GRID[int(np.argmin(val_eers))]
        out.update(final_detector(Mahalanobis(shrinkage=shrinkage), val_data, test_data))
        out["shrinkage"] = shrinkage
    else:
        kwargs = {"n_enroll": N_ENROLL, "seed": job["seed"]}
        kwargs.update(job["config"])
        cfg = TrainConfig(**kwargs)
        model, normalizer, history = train(train_data, val_data, cfg, log=lambda s: None)
        val = evaluate_split(model, normalizer, val_data, cfg)
        test = evaluate_split(model, normalizer, test_data, cfg, threshold=val["eer_threshold"])
        out.update({"val": val, "test": test, "epochs_run": len(history)})
    return out


def old_checkpoint_metrics(path: Path) -> dict:
    """Re-score the pre-C6 checkpoint on split seed 0, to confirm the 'before' numbers."""
    data = load_cmu(CMU_PATH)
    _, val_data, test_data = split_users(data, seed=0)
    model, normalizer, _ = load_checkpoint(path)
    cfg = TrainConfig(n_enroll=N_ENROLL)
    val = evaluate_split(model, normalizer, val_data, cfg)
    return {"val": val, "test": evaluate_split(model, normalizer, test_data, cfg, threshold=val["eer_threshold"])}


def cmd_final(args: argparse.Namespace) -> None:
    jobs: list[dict] = []
    for seed in FINAL_SEEDS:
        jobs.append({"seed": seed, "method": "scaled_manhattan"})
        jobs.append({"seed": seed, "method": "mahalanobis"})
        jobs.append({"seed": seed, "method": "encoder_before", "config": BEFORE})
        jobs.append({"seed": seed, "method": "encoder_after", "config": {}})
    with ProcessPoolExecutor(args.workers) as pool:
        runs = list(pool.map(final_job, jobs))
    payload: dict = {"seeds": list(FINAL_SEEDS), "n_enroll": N_ENROLL, "runs": runs}
    if args.before_checkpoint is not None:
        payload["old_checkpoint_seed0"] = old_checkpoint_metrics(args.before_checkpoint)
    print(write_json("final.json", payload))


# ---- table ------------------------------------------------------------------


def pct(mean: float, std: float) -> str:
    return f"{mean * 100:.2f} +/- {std * 100:.2f}"


def print_noise() -> None:
    noise = read_json("noise.json")
    print(f"\n### EER noise of a {noise['group_size']}-user group ({noise['n_users_pool']} development users, {NOISE_GROUPS} draws)\n")
    print("| detector | mean EER | std | min | max | mean gap between two groups | p90 gap | P(gap >= 11 pts) | EER on all 43 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, r in noise["detectors"].items():
        print(
            f"| {name} | {r['mean'] * 100:.2f} | {r['std'] * 100:.2f} | {r['min'] * 100:.2f} | {r['max'] * 100:.2f} "
            f"| {r['gap_mean'] * 100:.2f} | {r['gap_p90'] * 100:.2f} | {r['frac_gap_ge_11pts'] * 100:.1f}% "
            f"| {r['all_43_users_eer'] * 100:.2f} |"
        )


def print_dev() -> None:
    runs = read_json("dev.json")
    names = sorted({r["variant"] for r in runs})
    print("\n### Encoder variants (development users only, 27 train / 8 val / 8 holdout, 3 inner splits, full 30 epochs)\n")
    print("| variant | stop rule | val EER | holdout EER | holdout minus val | mean epoch kept | resultant length |")
    print("|---|---|---|---|---|---|---|")
    for name in names:
        group = [r for r in runs if r["variant"] == name and r["inner"] != 0]
        if not group:
            continue
        for rule in ("raw min, patience 10", "EMA min, patience 10", "last epoch"):
            vals, holds, kept = [], [], []
            for r in group:
                eers = [h["eer"] for h in r["history"]]
                i = len(eers) - 1
                if rule.startswith("raw"):
                    i = select_epoch(eers, 10)
                elif rule.startswith("EMA"):
                    i = select_epoch(ema(eers), 10)
                vals.append(eers[i])
                holds.append(r["history"][i]["holdout"]["eer"])
                kept.append(i + 1)
            gap = float(np.mean(np.array(holds) - np.array(vals)))
            print(
                f"| {name} | {rule} | {pct(*mean_std(vals))} | {pct(*mean_std(holds))} | {gap * 100:+.2f} "
                f"| {np.mean(kept):.1f} | {np.mean([r['resultant'] for r in group]):.3f} |"
            )
    detectors = read_json("dev_detectors.json")
    print("\n### Paired per inner split (same val and holdout users, encoder at its last epoch, EER %)\n")
    print("| method | " + " | ".join(f"inner {d['inner']} val / holdout" for d in detectors) + " | mean val / holdout |")
    print("|---|" + "---|" * (len(detectors) + 1))
    for key in ("scaled_manhattan", "mahalanobis_s0.9"):
        cells = [f"{d[key]['val'] * 100:.1f} / {d[key]['holdout'] * 100:.1f}" for d in detectors]
        val_mean = np.mean([d[key]["val"] for d in detectors]) * 100
        hold_mean = np.mean([d[key]["holdout"] for d in detectors]) * 100
        print(f"| {key} | " + " | ".join(cells) + f" | {val_mean:.2f} / {hold_mean:.2f} |")
    for name in names:
        group = [r for r in runs if r["variant"] == name and r["inner"] != 0]
        if not group:
            continue
        cells = []
        val_all, hold_all = [], []
        for d in detectors:
            last = next(r for r in group if r["inner"] == d["inner"])["history"][-1]
            val_all.append(last["eer"])
            hold_all.append(last["holdout"]["eer"])
            cells.append(f"{last['eer'] * 100:.1f} / {last['holdout']['eer'] * 100:.1f}")
        print(f"| {name} | " + " | ".join(cells) + f" | {np.mean(val_all) * 100:.2f} / {np.mean(hold_all) * 100:.2f} |")
    print("\n### Training-seed variance on the fixed seed-0 split (val users only)\n")
    print("| variant | seeds | best val EER | final-epoch val EER |")
    print("|---|---|---|---|")
    for name in names:
        group = [r for r in runs if r["variant"] == name and r["inner"] == 0]
        if not group:
            continue
        best = [min(h["eer"] for h in r["history"]) for r in group]
        last = [r["history"][-1]["eer"] for r in group]
        print(f"| {name} | {len(group)} | {pct(*mean_std(best))} | {pct(*mean_std(last))} |")


def print_stateless() -> None:
    rows = read_json("stateless.json")
    print("\n### Classic features through the stateless Embedder protocol (development users only, global EER %)\n")
    print("| embedding | seed-0 val (8 users) | all 43 development users |")
    print("|---|---|---|")
    for name, r in rows.items():
        print(f"| {name} | {r['val_8_users'] * 100:.2f} | {r['all_43_dev_users'] * 100:.2f} |")


def print_engine() -> None:
    r = read_json("engine.json")
    print("\n### Shipped checkpoint through the engine (seed-0 val users, val threshold, no adaptation)\n")
    print("| genuine accepted | impostors accepted | enrollments failed | attempts refused by validation | median verify ms |")
    print("|---|---|---|---|---|")
    print(
        f"| {r['genuine_accepted']}/{r['genuine']} ({r['genuine_accepted'] / r['genuine'] * 100:.1f}%) "
        f"| {r['impostor_accepted']}/{r['impostor']} ({r['impostor_accepted'] / r['impostor'] * 100:.1f}%) "
        f"| {r['enroll_failed']} | {r['refused']} | {r['median_verify_ms']:.2f} |"
    )


def print_final() -> None:
    final = read_json("final.json")
    print(f"\n### CMU, {final['n_enroll']} enrollment attempts, split seeds {final['seeds']} (mean +/- sample std, global EER, %)\n")
    print("| method | val EER | test EER | mean test minus val | test per-user EER | FAR at val threshold | FRR at val threshold |")
    print("|---|---|---|---|---|---|---|")
    methods = sorted({r["method"] for r in final["runs"]})
    for m in methods:
        group = [r for r in final["runs"] if r["method"] == m]
        cells = [
            pct(*mean_std([r["val"]["eer"] for r in group])),
            pct(*mean_std([r["test"]["eer"] for r in group])),
            f"{np.mean([r['test']['eer'] - r['val']['eer'] for r in group]) * 100:+.2f}",
            pct(*mean_std([r["test"]["per_user_eer"] for r in group])),
            pct(*mean_std([r["test"]["far"] for r in group])),
            pct(*mean_std([r["test"]["frr"] for r in group])),
        ]
        print(f"| {m} | " + " | ".join(cells) + " |")
    print("\n### Per seed (global EER, %)\n")
    print("| method | seed | val | test | test minus val |")
    print("|---|---|---|---|---|")
    for m in methods:
        for r in sorted((r for r in final["runs"] if r["method"] == m), key=lambda x: x["seed"]):
            gap = (r["test"]["eer"] - r["val"]["eer"]) * 100
            print(f"| {m} | {r['seed']} | {r['val']['eer'] * 100:.2f} | {r['test']['eer'] * 100:.2f} | {gap:+.2f} |")
    print("\n### encoder_after minus each method, paired by seed (test EER, points; negative = encoder better)\n")
    print("| compared with | per seed | mean | sample std | seeds encoder wins |")
    print("|---|---|---|---|---|")
    after = {r["seed"]: r["test"]["eer"] for r in final["runs"] if r["method"] == "encoder_after"}
    for m in methods:
        if m != "encoder_after":
            other = {r["seed"]: r["test"]["eer"] for r in final["runs"] if r["method"] == m}
            diffs = [(after[seed] - other[seed]) * 100 for seed in final["seeds"]]
            mean, std = mean_std(diffs)
            wins = sum(d < 0 for d in diffs)
            print(f"| {m} | {' '.join(f'{d:+.2f}' for d in diffs)} | {mean:+.2f} | {std:.2f} | {wins}/{len(diffs)} |")
    if "old_checkpoint_seed0" in final:
        old = final["old_checkpoint_seed0"]
        print(f"\nOld checkpoint re-scored on split seed 0: val {old['val']['eer'] * 100:.2f}, test {old['test']['eer'] * 100:.2f}")


def cmd_table(args: argparse.Namespace) -> None:
    printers = (
        ("noise.json", print_noise),
        ("dev.json", print_dev),
        ("stateless.json", print_stateless),
        ("engine.json", print_engine),
        ("final.json", print_final),
    )
    for name, printer in printers:
        if (OUT_DIR / name).exists():
            printer()
        else:
            print(f"\n({name} not found; run the matching command first)", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("noise").set_defaults(run=cmd_noise)
    sub.add_parser("stateless").set_defaults(run=cmd_stateless)
    engine = sub.add_parser("engine")
    engine.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "model.pt")
    engine.set_defaults(run=cmd_engine)
    dev = sub.add_parser("dev")
    dev.add_argument("--workers", type=int, default=4)
    dev.add_argument("--seed-variant", default="supcon_t0.1", choices=sorted(DEV_VARIANTS))
    dev.set_defaults(run=cmd_dev)
    final = sub.add_parser("final")
    final.add_argument("--workers", type=int, default=4)
    final.add_argument("--before-checkpoint", type=Path, default=None)
    final.set_defaults(run=cmd_final)
    sub.add_parser("table").set_defaults(run=cmd_table)
    args = ap.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
