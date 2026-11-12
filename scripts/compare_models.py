"""C6 experiments: every number in docs/RESULTS.md comes from this script.

    python scripts/compare_models.py noise   # how noisy is an EER measured on 8 users?
    python scripts/compare_models.py dev     # encoder variants, development users only
    python scripts/compare_models.py stateless   # can a classic method fit the Embedder protocol?
    python scripts/compare_models.py engine  # the shipped checkpoint through the real engine, val users
    python scripts/compare_models.py final   # multi-seed comparison, reads test users
    python scripts/compare_models.py peruser # C6b per-user thresholds, rule fitted on val, reads test users
    python scripts/compare_models.py table   # print the RESULTS.md tables from the JSON files
    python scripts/compare_models.py c11 STAGE   # C11 Aalto training + CMU login test (stages: see the C11 section)

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
from keyauth.auth.scorers import MahalanobisScorer, ScaledManhattanScorer, UserScorer, enrollment_loo_scores
from keyauth.baselines import Detector, Mahalanobis, ScaledManhattan, detector_scores, timing_vectors
from keyauth.data import KeystrokeData, Normalizer, load_cmu, split_users
from keyauth.data.preprocess import EPS_S, valid_mask
from keyauth.evaluate import (
    Scores,
    compute_eer,
    far_frr,
    make_template,
    score_protocol,
    summarize,
    verification_scores,
)
from keyauth.model import EncoderConfig, RhythmEncoder
from keyauth.thresholds import (
    WEIGHT_GRID,
    PerUserThresholdParams,
    embedding_loo_scores,
    enrollment_spread,
    fit_params,
    normalize_scores,
    references,
)
from keyauth.train import (
    AALTO_N_ENROLL,
    KEY_VOCAB,
    TrainConfig,
    aalto_splits,
    embed,
    evaluate_split,
    load_aalto_cached,
    load_checkpoint,
    save_checkpoint,
    train,
)

ROOT = Path(__file__).resolve().parent.parent
CMU_PATH = ROOT / "data" / "DSL-StrongPasswordData.csv"
AALTO_ROOT = ROOT / "data" / "aalto"
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


# ---- peruser: C6b per-user thresholds, fitted on val, reported on test ------


def enroll_rows(user_ids: np.ndarray, n_enroll: int = N_ENROLL) -> dict[str, np.ndarray]:
    """Each user's enrollment rows, the same "first n_enroll" rows that score_protocol uses."""
    user_ids = np.asarray(user_ids)
    return {u: np.flatnonzero(user_ids == u)[:n_enroll] for u in sorted(set(user_ids.tolist()))}


def peruser_eval(val_scores: Scores, val_spreads: dict, test_scores: Scores, test_spreads: dict) -> dict:
    """Before (one global threshold) and after (per-user thresholds) on val and test.

    Every threshold and rule parameter comes from val; test only reports.
    """
    val = summarize(val_scores)
    before = {"val": val, "test": summarize(test_scores, threshold=val["eer_threshold"])}
    params, weight_eers = fit_params(val_scores, val_spreads)

    def after_for(p: PerUserThresholdParams) -> dict:
        out: dict = {"params": p.to_dict()}
        for name, scores, spreads in (("val", val_scores, val_spreads), ("test", test_scores, test_spreads)):
            refs = references(spreads, p)
            clipped = [abs(refs[u] - (p.weight * s + (1 - p.weight) * p.reference_spread)) > 1e-12 for u, s in spreads.items()]
            out[name] = summarize(normalize_scores(scores, refs), threshold=p.scale)
            out[name]["clipped_users"] = int(sum(clipped))
            out[name]["thresholds"] = [p.scale * r for r in refs.values()]
        return out

    # fixed-weight rows are reported for reading only; the rule uses the val-selected weight
    fixed = {str(w): after_for(fit_params(val_scores, val_spreads, weights=[w])[0]) for w in WEIGHT_GRID}
    return {"before": before, "after": after_for(params), "val_eer_by_weight": weight_eers, "fixed_weight": fixed}


def detector_peruser(scorer: UserScorer, data: KeystrokeData) -> tuple[Scores, dict]:
    vectors = timing_vectors(data)
    spreads = {u: enrollment_spread(enrollment_loo_scores(scorer, vectors[idx])) for u, idx in enroll_rows(data.user_ids).items()}
    return detector_scores(as_detector(scorer), data, N_ENROLL), spreads


def as_detector(scorer: UserScorer) -> Detector:
    """The baselines.Detector with the same settings as `scorer`, for the batch protocol."""
    if isinstance(scorer, MahalanobisScorer):
        return Mahalanobis(shrinkage=scorer.shrinkage, min_scale=scorer.min_scale)
    return ScaledManhattan(min_scale=scorer.min_scale)


def peruser_job(seed: int) -> dict:
    torch.set_num_threads(THREADS_PER_WORKER)
    train_data, val_data, test_data = split_users(load_cmu(CMU_PATH), seed=seed)
    out: dict = {"seed": seed}

    # encoder: the shipped recipe (defaults), same run as `final`'s encoder_after
    cfg = TrainConfig(n_enroll=N_ENROLL, seed=seed)
    model, normalizer, _ = train(train_data, val_data, cfg, log=lambda s: None)
    parts = []
    for d in (val_data, test_data):
        z = embed(model, normalizer, d)
        spreads = {u: enrollment_spread(embedding_loo_scores(z[idx])) for u, idx in enroll_rows(d.user_ids).items()}
        parts.extend([verification_scores(z, d.user_ids, N_ENROLL), spreads])
    out["encoder"] = peruser_eval(*parts)

    out["scaled_manhattan"] = peruser_eval(
        *detector_peruser(ScaledManhattanScorer(), val_data), *detector_peruser(ScaledManhattanScorer(), test_data)
    )
    # shrinkage picked on val users only, as in `final`
    val_eers = [summarize(detector_scores(Mahalanobis(shrinkage=s), val_data, N_ENROLL))["eer"] for s in SHRINKAGE_GRID]
    scorer = MahalanobisScorer(shrinkage=SHRINKAGE_GRID[int(np.argmin(val_eers))])
    out["mahalanobis"] = peruser_eval(*detector_peruser(scorer, val_data), *detector_peruser(scorer, test_data))
    out["mahalanobis"]["shrinkage"] = scorer.shrinkage
    return out


def cmd_peruser(args: argparse.Namespace) -> None:
    with ProcessPoolExecutor(args.workers) as pool:
        runs = list(pool.map(peruser_job, FINAL_SEEDS))
    print(write_json("peruser.json", {"seeds": list(FINAL_SEEDS), "n_enroll": N_ENROLL, "runs": runs}))


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


PERUSER_METHODS = ("encoder", "scaled_manhattan", "mahalanobis")


def print_peruser() -> None:
    data = read_json("peruser.json")
    runs = data["runs"]
    by_seed = {r["seed"]: r for r in runs}
    cols = (("val", "eer"), ("val", "far"), ("val", "frr"), ("test", "eer"), ("test", "far"), ("test", "frr"))
    header = "| method | thresholds | val EER | val FAR | val FRR | test EER | test FAR | test FRR |"
    for title, group in (
        (f"split seeds {data['seeds']} (mean +/- sample std, %)", runs),
        ("split seed 0 only (%)", [by_seed[0]]),
    ):
        print(f"\n### C6b before / after, CMU, {data['n_enroll']} enrollment attempts, {title}\n")
        print(header)
        print("|---|---|---|---|---|---|---|---|")
        for m in PERUSER_METHODS:
            for label, key in (("global (before)", "before"), ("per-user (after)", "after")):
                cells = []
                for split, metric in cols:
                    mean, std = mean_std([r[m][key][split][metric] for r in group])
                    cell = f"{mean * 100:.2f}"
                    if len(group) > 1:
                        cell = pct(mean, std)
                    cells.append(cell)
                print(f"| {m} | {label} | " + " | ".join(cells) + " |")
    print("\n### Rule chosen on val, per seed\n")
    print("| method | seed | weight | scale | reference spread | global-equivalent threshold | users clipped val / test | test FAR before -> after | test FRR before -> after |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in PERUSER_METHODS:
        for r in sorted(runs, key=lambda x: x["seed"]):
            p = r[m]["after"]["params"]
            b, a = r[m]["before"]["test"], r[m]["after"]["test"]
            print(
                f"| {m} | {r['seed']} | {p['weight']:.2f} | {p['scale']:.4f} | {p['reference_spread']:.4f} "
                f"| {p['scale'] * p['reference_spread']:.4f} | {r[m]['after']['val']['clipped_users']} / {a['clipped_users']} "
                f"| {b['far'] * 100:.2f} -> {a['far'] * 100:.2f} | {b['frr'] * 100:.2f} -> {a['frr'] * 100:.2f} |"
            )
    print("\n### Fixed weight (not used for any choice; seeds 0-4 mean, %)\n")
    print("| method | weight | val EER | test EER | test FAR | test FRR |")
    print("|---|---|---|---|---|---|")
    for m in PERUSER_METHODS:
        for w in WEIGHT_GRID:
            rows = [r[m]["fixed_weight"][str(w)] for r in runs]
            print(
                f"| {m} | {w:.2f} | {pct(*mean_std([x['val']['eer'] for x in rows]))} "
                f"| {pct(*mean_std([x['test']['eer'] for x in rows]))} | {pct(*mean_std([x['test']['far'] for x in rows]))} "
                f"| {pct(*mean_std([x['test']['frr'] for x in rows]))} |"
            )


# ---- C11: Aalto (phrase-independent) training, CMU login test ---------------
#
# Stages, in order. Only `final` reads test users (Aalto test participants and CMU test users), once,
# after every choice below was frozen from the val-only stages:
#   c11 aalto   train Aalto variants (split seed 0; --seeds = training seeds), Aalto val  -> c11/aalto.json
#   c11 b1      every Aalto model on CMU val users, no CMU training, both normalizers    -> c11/b1.json
#   c11 b2      fine-tune the selected Aalto model on each split's CMU train users, grid -> c11/b2.json
#   c11 b3      the shipped CMU-only recipe re-run with the same replicates              -> c11/b3.json
#   c11 final   test users, once, for every variant                                      -> c11/final.json
#   c11 table   print the docs/RESULTS.md tables

C11_DIR = ROOT / "checkpoints" / "c11"
AALTO_SPLIT_SEED = 0  # Aalto participants / sentences split; training seeds vary separately
AALTO_TRAIN: dict = {"batches_per_epoch": 200, "p": 64, "k": 4, "lr": 1e-3}
AALTO_BASE: dict = {"width": 64, "layers": 1, "temperature": 0.1, "corrections": "keep", "keys": False, "epochs": 15}
# One change from the base each (the round-1 grid), then the round-2 combinations.
AALTO_VARIANTS: dict[str, dict] = {
    "base": {},
    "drop_corrections": {"corrections": "drop"},
    "keys": {"keys": True},
    "t0.05": {"temperature": 0.05},
    "t0.025": {"temperature": 0.025},
    "wide": {"width": 128},
    "deep": {"layers": 2},
    "e30": {"epochs": 30},
    # round 2: combinations of the round-1 / round-2 winners
    "t0.05_e30": {"temperature": 0.05, "epochs": 30},
    "t0.05_e30_keys": {"temperature": 0.05, "epochs": 30, "keys": True},  # privacy cost at the selected config
}
CMU_REPLICATES = (0, 1, 2)  # training seed = split seed + 100 * replicate; Aalto model of seed = replicate
# JavaScript keyCodes of CMU's .tie5Roanl + Return (Shift dropped), for the key-identity model only.
CMU_KEYCODES = (190, 84, 73, 69, 53, 82, 79, 65, 78, 76, 13)
B2_LRS = (1e-4, 3e-4, 1e-3)
B2_EPOCHS = (5, 15, 30)
# Decisions made from the val-only stages (see docs/RESULTS.md, C11); `final` and `b2` use them.
C11_SELECTED: dict = {"aalto": "t0.05_e30", "normalizer": "cmu", "b2_lr": 1e-3, "b2_epochs": 15}
# C11b (B2e: fine-tune with CMU's Return kept). Chosen on CMU val only (`c11 b2 --keep-enter` grid, then the
# normalizer at the chosen cell); `c11 final_b2e` reads test users once with these. Val, 3 replicates: B2e 19.84,
# B2 18.72, B3 19.10, so B2e is not selected for shipping; its test is scored only to report the comparison.
C11B_SELECTED: dict = {"aalto": "t0.05_e30", "normalizer": "cmu", "b2_lr": 1e-3, "b2_epochs": 30}
T_CRIT_95_DF4 = 2.776  # two-sided 95% t quantile, 4 degrees of freedom (5 paired splits)


def aalto_config(variant: str) -> dict:
    out = dict(AALTO_BASE)
    out.update(AALTO_VARIANTS[variant])
    return out


def aalto_ckpt(variant: str, seed: int) -> Path:
    return C11_DIR / f"aalto_{variant}_s{seed}.pt"


def c11_write(name: str, payload: object) -> Path:
    C11_DIR.mkdir(parents=True, exist_ok=True)
    path = C11_DIR / name
    path.write_text(json.dumps(payload, indent=1))
    return path


def c11_read(name: str) -> object:
    return json.loads((C11_DIR / name).read_text())


def c11_merge(name: str, rows: list[dict], key: Callable[[dict], tuple]) -> Path:
    """Add rows to a JSON list, replacing rows with the same key (re-runs overwrite)."""
    old = []
    if (C11_DIR / name).exists():
        old = c11_read(name)
    new_keys = {key(r) for r in rows}
    return c11_write(name, [r for r in old if key(r) not in new_keys] + rows)


def subset_metrics(scores: dict[str, tuple[list, list]], threshold: float | None) -> dict:
    """EER on the pooled genuine / impostor lists of each subset, and FAR / FRR at `threshold`."""
    out = {}
    for name, (gen, imp) in scores.items():
        g = np.concatenate(gen)
        i = np.concatenate(imp)
        row = {"n_genuine": int(g.size), "n_impostor": int(i.size), "eer": compute_eer(g, i).eer}
        if threshold is not None:
            row["far"], row["frr"] = far_frr(g, i, threshold)
        out[name] = row
    return out


def aalto_metrics(z: np.ndarray, data: KeystrokeData, n_enroll: int, threshold: float | None = None) -> dict:
    """Aalto verification: enroll each participant's first n_enroll held-out sentences, genuine = the rest,
    impostors = the first 5 of every other participant. EER over all attempts, and over clean / corrected
    attempts only (both genuine and impostor attempts filtered by has_correction; templates unchanged).
    threshold: the val threshold, for FAR / FRR (None = val itself: its own EER threshold)."""
    flags = data.has_correction.astype(np.float64)[:, None]
    subsets: dict[str, tuple[list, list]] = {"all": ([], []), "clean": ([], []), "corrected": ([], [])}

    def score_user(enroll: np.ndarray, genuine: np.ndarray, impostor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        template = make_template(enroll[:, :-1])
        dg = np.linalg.norm(genuine[:, :-1] - template, axis=1)
        di = np.linalg.norm(impostor[:, :-1] - template, axis=1)
        for name, keep_g, keep_i in (
            ("all", np.ones(len(dg), bool), np.ones(len(di), bool)),
            ("clean", genuine[:, -1] < 0.5, impostor[:, -1] < 0.5),
            ("corrected", genuine[:, -1] > 0.5, impostor[:, -1] > 0.5),
        ):
            subsets[name][0].append(dg[keep_g])
            subsets[name][1].append(di[keep_i])
        return dg, di

    scores = score_protocol(np.hstack([z, flags]), data.user_ids, score_user, n_enroll, 5)
    summary = summarize(scores, threshold)
    out = {"summary": summary, "users": len(data.users), "attempts": len(data)}
    out.update(subset_metrics(subsets, summary["threshold"]))
    return out


def aalto_job(job: dict) -> dict:
    """Train one Aalto variant on train participants' seen sentences; score val participants only."""
    torch.set_num_threads(job.get("threads", 1))
    t0 = time.time()
    v = aalto_config(job["variant"])
    data = load_aalto_cached(AALTO_ROOT, C11_DIR / "aalto.npz")
    train_data, val_data, _ = aalto_splits(data, v["corrections"], AALTO_N_ENROLL, seed=AALTO_SPLIT_SEED)
    del data
    cfg = TrainConfig(
        epochs=v["epochs"], temperature=v["temperature"], n_enroll=AALTO_N_ENROLL, seed=job["seed"], **AALTO_TRAIN
    )
    enc = EncoderConfig(conv_channels=v["width"], hidden_size=v["width"], num_layers=v["layers"])
    if v["keys"]:
        enc.key_vocab = KEY_VOCAB
    log_path = C11_DIR / f"aalto_{job['variant']}_s{job['seed']}.log"
    with log_path.open("w") as log_file:

        def log(line: str) -> None:
            log_file.write(line + "\n")
            log_file.flush()

        model, normalizer, history = train(train_data, val_data, cfg, enc, log=log)
    val = aalto_metrics(embed(model, normalizer, val_data), val_data, AALTO_N_ENROLL)
    save_checkpoint(
        aalto_ckpt(job["variant"], job["seed"]), model, normalizer, val["summary"]["eer_threshold"],
        {"val": val, "config": v}, final_enter=False,
    )
    return {
        "variant": job["variant"],
        "seed": job["seed"],
        "config": v,
        "train_sentences": len(train_data),
        "train_users": len(train_data.users),
        "val": val,
        "history": history,
        "seconds": time.time() - t0,
    }


def run_jobs(fn: Callable[[dict], dict], jobs: list[dict], workers: int) -> list[dict]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, jobs))


def cmd_c11_aalto(args: argparse.Namespace) -> None:
    variants = args.variants.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = [{"variant": v, "seed": s, "threads": args.threads} for v in variants for s in seeds]
    rows = run_jobs(aalto_job, jobs, args.workers)
    print(c11_merge("aalto.json", rows, lambda r: (r["variant"], r["seed"])))


def cmu_data(final_enter: bool, keys: bool = False) -> KeystrokeData:
    """CMU, optionally with keyCodes (key-identity model) and without the final Return (Aalto-trained models)."""
    data = load_cmu(CMU_PATH)
    if keys:
        data = dataclasses.replace(data, key_codes=np.tile(np.array(CMU_KEYCODES, dtype=np.int64), (len(data), 1)))
    if not final_enter:
        data = data.drop_final_key()
    return data


def cmu_eval(model: RhythmEncoder, normalizer: Normalizer, val_data: KeystrokeData, test_data: KeystrokeData | None) -> dict:
    """CMU login protocol: 10 enrollment attempts, global threshold at the val EER; test (if given) at that threshold."""
    cfg = TrainConfig(n_enroll=N_ENROLL)
    out = {"val": evaluate_split(model, normalizer, val_data, cfg)}
    if test_data is not None:
        out["test"] = evaluate_split(model, normalizer, test_data, cfg, threshold=out["val"]["eer_threshold"])
    return out


def cmu_normalizer(mode: str, model_normalizer: Normalizer, train_data: KeystrokeData) -> Normalizer:
    if mode not in ("aalto", "cmu"):
        raise ValueError(f"normalizer mode must be 'aalto' or 'cmu', got {mode!r}")
    if mode == "aalto":
        return model_normalizer
    return Normalizer.fit(train_data)


def b1_job(job: dict) -> dict:
    """An Aalto model on one CMU split, no CMU training. Reads test users only if job["test"]."""
    torch.set_num_threads(job.get("threads", 1))
    model, normalizer, _ = load_checkpoint(job["checkpoint"])
    keys = model.config.key_vocab > 0
    train_data, val_data, test_data = split_users(cmu_data(final_enter=False, keys=keys), seed=job["split"])
    if not job.get("test"):
        test_data = None
    out = {k: job[k] for k in ("variant", "seed", "split", "normalizer")}
    out.update(cmu_eval(model, cmu_normalizer(job["normalizer"], normalizer, train_data), val_data, test_data))
    return out


def cmd_c11_b1(args: argparse.Namespace) -> None:
    rows = c11_read("aalto.json")
    jobs = [
        {"checkpoint": str(aalto_ckpt(r["variant"], r["seed"])), "variant": r["variant"], "seed": r["seed"],
         "split": s, "normalizer": n}
        for r in rows for s in FINAL_SEEDS for n in ("aalto", "cmu")
    ]
    print(c11_write("b1.json", run_jobs(b1_job, jobs, args.workers)))


def b2_ckpt(variant: str, lr: float, epochs: int, split: int, replicate: int, final_enter: bool = False,
            normalizer: str = "cmu") -> Path:
    """B2 checkpoints in c11/b2/; B2e (Return kept, C11b) in c11/b2e/, with the normalizer in the name if not CMU's."""
    name = f"{variant}_lr{lr:g}_e{epochs}_split{split}_r{replicate}.pt"
    if not final_enter:
        return C11_DIR / "b2" / name
    if normalizer != "cmu":
        name = f"{variant}_norm{normalizer}_lr{lr:g}_e{epochs}_split{split}_r{replicate}.pt"
    return C11_DIR / "b2e" / name


def b2_job(job: dict) -> dict:
    """Fine-tune the Aalto model of seed `replicate` on one split's CMU train users: Return dropped (B2), or kept
    if job["final_enter"] (B2e, C11b; the Aalto model never saw an Enter, the fine-tuned one expects it)."""
    torch.set_num_threads(job.get("threads", 1))
    t0 = time.time()
    final_enter = bool(job.get("final_enter", False))
    init, init_normalizer, _ = load_checkpoint(aalto_ckpt(job["variant"], job["replicate"]))
    train_data, val_data, _ = split_users(cmu_data(final_enter=final_enter), seed=job["split"])
    normalizer = cmu_normalizer(job["normalizer"], init_normalizer, train_data)
    cfg = TrainConfig(epochs=job["epochs"], lr=job["lr"], n_enroll=N_ENROLL, seed=job["split"] + 100 * job["replicate"])
    model, normalizer, history = train(train_data, val_data, cfg, init=init, normalizer=normalizer, log=lambda s: None)
    path = b2_ckpt(job["variant"], job["lr"], job["epochs"], job["split"], job["replicate"], final_enter, job["normalizer"])
    save_checkpoint(path, model, normalizer, history[-1]["eer_threshold"], {}, final_enter=final_enter)
    out = {k: job[k] for k in ("variant", "lr", "epochs", "split", "replicate", "normalizer")}
    out["final_enter"] = final_enter
    out.update(cmu_eval(model, normalizer, val_data, None))
    out["seconds"] = time.time() - t0
    return out


def cmd_c11_b2(args: argparse.Namespace) -> None:
    """b2: Return dropped -> b2.json. b2 --keep-enter (C11b, B2e): Return kept -> b2e.json."""
    replicates = [int(r) for r in args.replicates.split(",")]
    grid = [(lr, e) for lr in B2_LRS for e in B2_EPOCHS]
    variant = C11_SELECTED["aalto"]
    selected = C11_SELECTED
    out_name = "b2.json"
    if args.keep_enter:
        selected = C11B_SELECTED
        out_name = "b2e.json"
    normalizer = selected["normalizer"]
    if args.b2_base is not None:
        variant = args.b2_base
    if args.normalizer is not None:
        normalizer = args.normalizer
    if args.selected_only:
        grid = [(selected["b2_lr"], selected["b2_epochs"])]
    jobs = [
        {"variant": variant, "normalizer": normalizer, "lr": lr, "epochs": e,
         "split": s, "replicate": r, "final_enter": args.keep_enter}
        for lr, e in grid for s in FINAL_SEEDS for r in replicates
    ]
    rows = run_jobs(b2_job, jobs, args.workers)
    print(c11_merge(out_name, rows, lambda r: (r["variant"], r["normalizer"], r["lr"], r["epochs"], r["split"], r["replicate"])))


def b3_ckpt(split: int, replicate: int) -> Path:
    return C11_DIR / "b3" / f"split{split}_r{replicate}.pt"


def b3_job(job: dict) -> dict:
    """The shipped CMU-only recipe (train.py defaults, Return kept); replicate 0 = docs/RESULTS.md's C6 seeds."""
    torch.set_num_threads(job.get("threads", 1))
    train_data, val_data, _ = split_users(load_cmu(CMU_PATH), seed=job["split"])
    cfg = TrainConfig(n_enroll=N_ENROLL, seed=job["split"] + 100 * job["replicate"])
    model, normalizer, history = train(train_data, val_data, cfg, log=lambda s: None)
    save_checkpoint(b3_ckpt(job["split"], job["replicate"]), model, normalizer, history[-1]["eer_threshold"], {})
    out = {k: job[k] for k in ("split", "replicate")}
    out.update(cmu_eval(model, normalizer, val_data, None))
    return out


def cmd_c11_b3(args: argparse.Namespace) -> None:
    jobs = [{"split": s, "replicate": r} for s in FINAL_SEEDS for r in CMU_REPLICATES]
    print(c11_write("b3.json", run_jobs(b3_job, jobs, args.workers)))


def cmu_ckpt_job(job: dict) -> dict:
    """Val and test of a saved CMU-split model (B2 / B3) on its own split."""
    torch.set_num_threads(job.get("threads", 1))
    model, normalizer, meta = load_checkpoint(job["checkpoint"])
    _, val_data, test_data = split_users(cmu_data(final_enter=meta["final_enter"]), seed=job["split"])
    out = {k: v for k, v in job.items() if k != "checkpoint"}
    out.update(cmu_eval(model, normalizer, val_data, test_data))
    return out


def aalto_test_job(job: dict) -> dict:
    """One Aalto model on Aalto val (threshold) and test participants."""
    torch.set_num_threads(job.get("threads", 1))
    model, normalizer, _ = load_checkpoint(aalto_ckpt(job["variant"], job["seed"]))
    data = load_aalto_cached(AALTO_ROOT, C11_DIR / "aalto.npz")
    _, val_data, test_data = aalto_splits(data, "keep", AALTO_N_ENROLL, seed=AALTO_SPLIT_SEED)
    del data
    val = aalto_metrics(embed(model, normalizer, val_data), val_data, AALTO_N_ENROLL)
    test = aalto_metrics(embed(model, normalizer, test_data), test_data, AALTO_N_ENROLL, val["summary"]["eer_threshold"])
    return {"variant": job["variant"], "seed": job["seed"], "val": val, "test": test}


def cmd_c11_final(args: argparse.Namespace) -> None:
    """Reads TEST users. Run once, after C11_SELECTED is frozen."""
    aalto_rows = c11_read("aalto.json")
    aalto_jobs = [{"variant": r["variant"], "seed": r["seed"]} for r in aalto_rows]
    b1_jobs = [
        {"checkpoint": str(aalto_ckpt(r["variant"], r["seed"])), "variant": r["variant"], "seed": r["seed"],
         "split": s, "normalizer": C11_SELECTED["normalizer"], "test": True}
        for r in aalto_rows for s in FINAL_SEEDS
    ]
    lr, epochs = C11_SELECTED["b2_lr"], C11_SELECTED["b2_epochs"]
    b2_jobs = [
        {"checkpoint": str(b2_ckpt(C11_SELECTED["aalto"], lr, epochs, s, r)), "variant": "b2", "split": s, "replicate": r}
        for s in FINAL_SEEDS for r in CMU_REPLICATES
    ]
    b3_jobs = [
        {"checkpoint": str(b3_ckpt(s, r)), "variant": "b3", "split": s, "replicate": r}
        for s in FINAL_SEEDS for r in CMU_REPLICATES
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        aalto = list(pool.map(aalto_test_job, aalto_jobs))
        b1 = list(pool.map(b1_job, b1_jobs))
        cmu = list(pool.map(cmu_ckpt_job, b2_jobs + b3_jobs))
    payload = {"selected": C11_SELECTED, "aalto": aalto, "b1": b1, "b2": cmu[: len(b2_jobs)], "b3": cmu[len(b2_jobs) :]}
    print(c11_write("final.json", payload))


def cmd_c11_final_b2e(args: argparse.Namespace) -> None:
    """C11b: reads TEST users for B2e only (B2 / B3 test numbers come from final.json). Run once, after
    C11B_SELECTED is frozen."""
    sel = C11B_SELECTED
    jobs = [
        {"checkpoint": str(b2_ckpt(sel["aalto"], sel["b2_lr"], sel["b2_epochs"], s, r, True, sel["normalizer"])),
         "variant": "b2e", "split": s, "replicate": r}
        for s in FINAL_SEEDS for r in CMU_REPLICATES
    ]
    rows = run_jobs(cmu_ckpt_job, jobs, args.workers)
    print(c11_write("final_b2e.json", {"selected": sel, "b2e": rows}))


def ms(values: list[float]) -> str:
    """'mean +/- sample std' in percent (just the mean for one value)."""
    mean, std = mean_std(values)
    if len(values) < 2:
        return f"{mean * 100:.2f}"
    return pct(mean, std)


def group_by(rows: list[dict], key: Callable[[dict], object]) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(key(r), []).append(r)
    return out


def split_means(rows: list[dict], metric: Callable[[dict], float]) -> list[float]:
    """Average replicates within each CMU split, one value per split (the std across these is the split std)."""
    return [float(np.mean([metric(r) for r in g])) for _, g in sorted(group_by(rows, lambda r: r["split"]).items())]


def replicate_means(rows: list[dict], metric: Callable[[dict], float]) -> list[float]:
    """Average splits within each replicate / training seed (the std across these is the training-seed std)."""
    key = "replicate"
    if "replicate" not in rows[0]:
        key = "seed"
    return [float(np.mean([metric(r) for r in g])) for _, g in sorted(group_by(rows, lambda r: r[key]).items())]


def print_c11_val() -> None:
    aalto = c11_read("aalto.json")
    print("\n### C11 Aalto variants, VAL participants (unseen people, unseen sentences; EER %)\n")
    print("| variant | config | seeds | train sentences | val EER | val EER clean | val EER corrected | best epoch (EER) | minutes |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, g in group_by(aalto, lambda r: r["variant"]).items():
        c = g[0]["config"]
        keys = "off"
        if c["keys"]:
            keys = "on"
        cfg = f"w{c['width']} l{c['layers']} T{c['temperature']} {c['corrections']} keys={keys} {c['epochs']}ep"
        best = [min(range(len(r["history"])), key=lambda i: r["history"][i]["eer"]) for r in g]
        print(
            f"| {name} | {cfg} | {len(g)} | {g[0]['train_sentences']} | {ms([r['val']['all']['eer'] for r in g])} "
            f"| {ms([r['val']['clean']['eer'] for r in g])} | {ms([r['val']['corrected']['eer'] for r in g])} "
            f"| {best[0] + 1} ({g[0]['history'][best[0]]['eer'] * 100:.2f}) | {np.mean([r['seconds'] for r in g]) / 60:.0f} |"
        )
    print("\nval EER by epoch (seed 0): " )
    for r in aalto:
        if r["seed"] == 0:
            print(f"- {r['variant']}: " + " ".join(f"{h['eer'] * 100:.1f}" for h in r["history"]))
    if (C11_DIR / "b1.json").exists():
        b1 = c11_read("b1.json")
        print("\n### C11 B1: Aalto model on CMU, no CMU training, VAL users (split seeds 0-4, EER %)\n")
        print("| variant | Aalto seed | normalizer | CMU val EER |")
        print("|---|---|---|---|")
        for (v, seed, n), g in group_by(b1, lambda r: (r["variant"], r["seed"], r["normalizer"])).items():
            print(f"| {v} | {seed} | {n} | {ms([r['val']['eer'] for r in g])} |")
    if (C11_DIR / "b2.json").exists():
        b2 = c11_read("b2.json")
        print("\n### C11 B2: fine-tuned on each split's CMU train users, VAL users (EER %, mean over splits)\n")
        print("| Aalto model | lr | epochs | replicates | CMU val EER | minutes per run |")
        print("|---|---|---|---|---|---|")
        for (v, lr, e), g in sorted(group_by(b2, lambda r: (r["variant"], r["lr"], r["epochs"])).items()):
            reps = sorted({r["replicate"] for r in g})
            print(
                f"| {v} | {lr:g} | {e} | {reps} | {ms(split_means(g, lambda r: r['val']['eer']))} "
                f"| {np.mean([r['seconds'] for r in g]) / 60:.1f} |"
            )
    if (C11_DIR / "b3.json").exists():
        b3 = c11_read("b3.json")
        print("\n### C11 B3: CMU-only recipe re-run, VAL users (EER %)\n")
        print(f"replicate 0: {ms([r['val']['eer'] for r in b3 if r['replicate'] == 0])}; "
              f"all replicates, split means: {ms(split_means(b3, lambda r: r['val']['eer']))}")


def print_c11_final() -> None:
    final = c11_read("final.json")
    sel = final["selected"]
    print(f"\n### C11 final (selected on val: {sel})\n")
    b1_sel = [r for r in final["b1"] if r["variant"] == sel["aalto"]]
    variants = [
        ("B3: CMU-only encoder (re-run, Return kept)", final["b3"]),
        (f"B1: Aalto {sel['aalto']}, no CMU training", b1_sel),
        (f"B2: Aalto {sel['aalto']} + CMU fine-tune", final["b2"]),
    ]
    for name in sorted({r["variant"] for r in final["b1"]} - {sel["aalto"]}):
        variants.append((f"B1: Aalto {name}, no CMU training", [r for r in final["b1"] if r["variant"] == name]))
    print("**CMU login test, 10 enrollment attempts, split seeds 0-4 (mean +/- std across the 5 splits of the "
          "replicate-averaged values; %)**\n")
    print("| variant | runs | val EER | test EER | test FAR at val threshold | test FRR at val threshold | test EER, std across training seeds | test EER, replicate 0 only |")
    print("|---|---|---|---|---|---|---|---|")
    for name, rows in variants:
        r0 = [r for r in rows if r.get("replicate", r.get("seed")) == 0]
        print(
            f"| {name} | {len(rows)} | {ms(split_means(rows, lambda r: r['val']['eer']))} "
            f"| {ms(split_means(rows, lambda r: r['test']['eer']))} | {ms(split_means(rows, lambda r: r['test']['far']))} "
            f"| {ms(split_means(rows, lambda r: r['test']['frr']))} | {ms(replicate_means(rows, lambda r: r['test']['eer']))} "
            f"| {ms([r['test']['eer'] for r in r0])} |"
        )
    print("\n**Per split (test EER %, replicate mean)**\n")
    print("| variant | " + " | ".join(f"split {s}" for s in FINAL_SEEDS) + " |")
    print("|---|" + "---|" * len(FINAL_SEEDS))
    for name, rows in variants[:3]:
        print(f"| {name} | " + " | ".join(f"{v * 100:.2f}" for v in split_means(rows, lambda r: r["test"]["eer"])) + " |")
    print("\n**Aalto, unseen participants typing unseen sentences, 5 enrollment sentences (EER %; FAR / FRR at the "
          "Aalto val threshold)**\n")
    print("| variant | seeds | val EER | val clean | val corrected | test EER | test clean | test corrected | test FAR | test FRR |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, g in group_by(final["aalto"], lambda r: r["variant"]).items():
        print(
            f"| {name} | {len(g)} | {ms([r['val']['all']['eer'] for r in g])} | {ms([r['val']['clean']['eer'] for r in g])} "
            f"| {ms([r['val']['corrected']['eer'] for r in g])} | {ms([r['test']['all']['eer'] for r in g])} "
            f"| {ms([r['test']['clean']['eer'] for r in g])} | {ms([r['test']['corrected']['eer'] for r in g])} "
            f"| {ms([r['test']['all']['far'] for r in g])} | {ms([r['test']['all']['frr'] for r in g])} |"
        )
    t = final["aalto"][0]["test"]
    print(f"\nAalto test: {t['users']} participants, {t['attempts']} sentences, {t['all']['n_genuine']} genuine / "
          f"{t['all']['n_impostor']} impostor comparisons (clean {t['clean']['n_genuine']} / corrected {t['corrected']['n_genuine']} genuine).")


def paired_diffs(a: list[dict], b: list[dict], metric: Callable[[dict], float]) -> list[float]:
    """Per-split differences a - b of the replicate means (same splits, so paired)."""
    return [x - y for x, y in zip(split_means(a, metric), split_means(b, metric))]


def print_paired(name: str, diffs: list[float]) -> None:
    """Paired split differences in points, with the 95% t interval half-width (4 df)."""
    mean, std = mean_std(diffs)
    half = T_CRIT_95_DF4 * std / np.sqrt(len(diffs))
    wins = sum(d < 0 for d in diffs)
    print(f"| {name} | {' '.join(f'{d * 100:+.2f}' for d in diffs)} | {mean * 100:+.2f} | {std * 100:.2f} "
          f"| +/- {half * 100:.2f} | {wins}/{len(diffs)} |")


def print_c11b() -> None:
    b2e = c11_read("b2e.json")
    b2 = c11_read("b2.json")
    sel = C11B_SELECTED
    print("\n### C11b B2e (Return kept) vs B2 (Return dropped), fine-tuning grid, CMU VAL (Aalto t0.05_e30 seed 0, "
          "replicate 0, CMU normalizer, split seeds 0-4; EER %)\n")
    print("| lr | epochs | B2e val EER (Return kept) | B2 val EER (Return dropped) | B2e - B2 per split (points) | mean |")
    print("|---|---|---|---|---|---|")
    for lr in B2_LRS:
        for e in B2_EPOCHS:
            def cell(rows: list[dict]) -> list[dict]:
                return [r for r in rows if r["variant"] == sel["aalto"] and r["normalizer"] == "cmu"
                        and r["lr"] == lr and r["epochs"] == e and r["replicate"] == 0]
            kept, dropped = cell(b2e), cell(b2)
            if not kept or len(dropped) != len(kept):
                continue
            d = paired_diffs(kept, dropped, lambda r: r["val"]["eer"])
            print(f"| {lr:g} | {e} | {ms(split_means(kept, lambda r: r['val']['eer']))} "
                  f"| {ms(split_means(dropped, lambda r: r['val']['eer']))} | {' '.join(f'{x * 100:+.2f}' for x in d)} "
                  f"| {np.mean(d) * 100:+.2f} |")
    print("\n**B2e normalizer and replicates at the chosen cell (CMU VAL, %)**\n")
    print("| normalizer | lr | epochs | replicates | val EER (mean +/- std across splits) |")
    print("|---|---|---|---|---|")
    for (n, lr, e), g in sorted(group_by(b2e, lambda r: (r["normalizer"], r["lr"], r["epochs"])).items()):
        if lr == sel["b2_lr"] and e == sel["b2_epochs"]:
            for reps in ([0], [0, 1, 2]):
                rows = [r for r in g if r["replicate"] in reps]
                if len(rows) == len(reps) * len(FINAL_SEEDS):
                    print(f"| {n} | {lr:g} | {e} | {reps} | {ms(split_means(rows, lambda r: r['val']['eer']))} |")
    if not (C11_DIR / "final_b2e.json").exists():
        return
    final = c11_read("final.json")
    b2e_test = c11_read("final_b2e.json")["b2e"]
    variants = (
        ("B3: CMU-only recipe (Return kept; C11)", final["b3"]),
        ("B2: Aalto t0.05_e30 + CMU fine-tune, Return dropped (C11)", final["b2"]),
        (f"B2e: Aalto t0.05_e30 + CMU fine-tune, Return kept (lr {sel['b2_lr']:g}, {sel['b2_epochs']} ep, "
         f"{sel['normalizer']} normalizer)", b2e_test),
    )
    print("\n### C11b final: CMU login test, 10 enrollment attempts, split seeds 0-4 (mean +/- std across the 5 splits "
          "of the 3-replicate means, %; FAR / FRR at each split's val threshold)\n")
    print("| variant | runs | val EER | test EER | test FAR | test FRR | test EER std across training seeds | test EER, replicate 0 |")
    print("|---|---|---|---|---|---|---|---|")
    for name, rows in variants:
        r0 = [r for r in rows if r["replicate"] == 0]
        print(
            f"| {name} | {len(rows)} | {ms(split_means(rows, lambda r: r['val']['eer']))} "
            f"| {ms(split_means(rows, lambda r: r['test']['eer']))} | {ms(split_means(rows, lambda r: r['test']['far']))} "
            f"| {ms(split_means(rows, lambda r: r['test']['frr']))} "
            f"| {mean_std(replicate_means(rows, lambda r: r['test']['eer']))[1] * 100:.2f} "
            f"| {ms([r['test']['eer'] for r in r0])} |"
        )
    print("\n**Per split (test EER %, replicate mean)**\n")
    print("| variant | " + " | ".join(f"split {s}" for s in FINAL_SEEDS) + " |")
    print("|---|" + "---|" * len(FINAL_SEEDS))
    for name, rows in variants:
        print(f"| {name.split(':')[0]} | " + " | ".join(f"{v * 100:.2f}" for v in split_means(rows, lambda r: r["test"]["eer"])) + " |")
    print("\n**Paired by split (points; negative = B2e better)**\n")
    print("| difference | per split | mean | sample std | 95% t half-width | splits B2e better |")
    print("|---|---|---|---|---|---|")
    for metric_name, split in (("test", "test"), ("val", "val")):
        print_paired(f"B2e - B3, {metric_name} EER", paired_diffs(b2e_test, final["b3"], lambda r: r[split]["eer"]))
        print_paired(f"B2e - B2, {metric_name} EER", paired_diffs(b2e_test, final["b2"], lambda r: r[split]["eer"]))


def cmd_c11_table(args: argparse.Namespace) -> None:
    print_c11_val()
    if (C11_DIR / "final.json").exists():
        print_c11_final()
    if (C11_DIR / "b2e.json").exists():
        print_c11b()


def cmd_c11(args: argparse.Namespace) -> None:
    stages: dict[str, Callable[[argparse.Namespace], None]] = {
        "aalto": cmd_c11_aalto,
        "b1": cmd_c11_b1,
        "b2": cmd_c11_b2,
        "b3": cmd_c11_b3,
        "final": cmd_c11_final,
        "final_b2e": cmd_c11_final_b2e,
        "table": cmd_c11_table,
    }
    stages[args.stage](args)


def cmd_table(args: argparse.Namespace) -> None:
    printers = (
        ("noise.json", print_noise),
        ("dev.json", print_dev),
        ("stateless.json", print_stateless),
        ("engine.json", print_engine),
        ("final.json", print_final),
        ("peruser.json", print_peruser),
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
    peruser = sub.add_parser("peruser")
    peruser.add_argument("--workers", type=int, default=4)
    peruser.set_defaults(run=cmd_peruser)
    sub.add_parser("table").set_defaults(run=cmd_table)
    c11 = sub.add_parser(
        "c11", help="C11 stages: aalto, b1, b2, b3 (val only), final (test, once), final_b2e (C11b test, once), table"
    )
    c11.add_argument("stage", choices=["aalto", "b1", "b2", "b3", "final", "final_b2e", "table"])
    c11.add_argument("--workers", type=int, default=4)
    c11.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    c11.add_argument("--variants", default="base,drop_corrections,keys,t0.05", help="aalto: comma separated AALTO_VARIANTS")
    c11.add_argument("--seeds", default="0", help="aalto: comma separated training seeds")
    c11.add_argument("--replicates", default="0", help="b2: comma separated replicates (Aalto model seeds)")
    c11.add_argument("--selected-only", action="store_true", help="b2: only C11_SELECTED's lr / epochs")
    c11.add_argument("--b2-base", default=None, help="b2: Aalto variant to fine-tune (default C11_SELECTED's)")
    c11.add_argument("--keep-enter", action="store_true", help="b2: keep CMU's Return (C11b, B2e) -> c11/b2e.json")
    c11.add_argument("--normalizer", choices=["aalto", "cmu"], default=None, help="b2: default the selected one")
    c11.set_defaults(run=cmd_c11)
    args = ap.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
