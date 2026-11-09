"""Train the rhythm encoder, pick the accept threshold, evaluate on unseen users.

    python -m keyauth.train --data cmu --cmu-path data/DSL-StrongPasswordData.csv
    python -m keyauth.train --data synthetic          # no dataset needed

Users are split train / val / test (no overlap). The model trains on train
users; after each epoch, validation users are enrolled and verified to get a
val EER, and the best epoch is kept. The accept threshold is chosen on val
users, then FAR / FRR / EER are reported on test users, who influenced
neither the weights nor the threshold.

The checkpoint holds everything the API needs: weights, encoder config,
normalizer and threshold.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from keyauth.data import KeystrokeData, Normalizer, load_cmu, make_synthetic_dataset, split_users
from keyauth.data.torch_dataset import KeystrokeDataset, PKBatchSampler
from keyauth.evaluate import summarize, verification_scores
from keyauth.model import EncoderConfig, RhythmEncoder, batch_hard_triplet_loss


@dataclass
class TrainConfig:
    epochs: int = 40
    batches_per_epoch: int = 100
    p: int = 8  # users per batch
    k: int = 8  # attempts per user per batch
    lr: float = 1e-3
    weight_decay: float = 1e-4
    margin: float = 0.2
    patience: int = 10  # stop after this many epochs without val improvement
    n_enroll: int = 10  # enrollment attempts per user during evaluation
    impostors_per_user: int = 5
    seed: int = 0
    device: str = "cpu"


def embed(model: RhythmEncoder, normalizer: Normalizer, data: KeystrokeData, batch_size: int = 1024) -> np.ndarray:
    """(N, embed_dim) embeddings for every attempt in `data`."""
    device = next(model.parameters()).device
    x_all = torch.from_numpy(normalizer(data))
    lengths_all = torch.from_numpy(data.lengths.astype(np.int64))
    was_training = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            x = x_all[start : start + batch_size].to(device)
            lengths = lengths_all[start : start + batch_size]
            out.append(model(x, lengths).cpu().numpy())
    model.train(was_training)
    return np.concatenate(out, axis=0)


def evaluate_split(model, normalizer, data: KeystrokeData, cfg: TrainConfig, threshold: float | None = None) -> dict:
    scores = verification_scores(
        embed(model, normalizer, data), data.user_ids, n_enroll=cfg.n_enroll, impostors_per_user=cfg.impostors_per_user
    )
    return summarize(scores, threshold)


def train(
    train_data: KeystrokeData,
    val_data: KeystrokeData,
    cfg: TrainConfig | None = None,
    enc_cfg: EncoderConfig | None = None,
    log: Callable[[str], None] = print,
) -> tuple[RhythmEncoder, Normalizer, list[dict]]:
    if cfg is None:
        cfg = TrainConfig()
    torch.manual_seed(cfg.seed)

    normalizer = Normalizer.fit(train_data)
    ds = KeystrokeDataset(train_data, normalizer)
    sampler = PKBatchSampler(
        ds.labels, p=min(cfg.p, len(train_data.users)), k=cfg.k, batches_per_epoch=cfg.batches_per_epoch, seed=cfg.seed
    )
    model = RhythmEncoder(enc_cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    history = []
    best_eer = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        losses, active = [], []
        for batch in sampler:
            x = ds.x[batch].to(cfg.device)
            stats = batch_hard_triplet_loss(model(x, ds.lengths[batch]), ds.labels[batch].to(cfg.device), cfg.margin)
            opt.zero_grad()
            stats.loss.backward()
            opt.step()
            losses.append(stats.loss.item())
            active.append(stats.active_frac)
        sched.step()

        val = evaluate_split(model, normalizer, val_data, cfg)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "active": float(np.mean(active)), **val}
        history.append(row)
        log(
            f"epoch {epoch:3d}  loss {row['loss']:.4f}  active {row['active']:.2f}  "
            f"val EER {val['eer'] * 100:5.2f}%  per-user EER {val['per_user_eer'] * 100:5.2f}%  "
            f"({time.time() - t0:.1f}s)"
        )

        if val["eer"] < best_eer:
            best_eer = val["eer"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= cfg.patience:
            log(f"no val improvement for {cfg.patience} epochs, stopping")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, normalizer, history


# ---- checkpoints --------------------------------------------------------


def save_checkpoint(path: str | Path, model: RhythmEncoder, normalizer: Normalizer, threshold: float, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "keyauth-v1",
            "encoder_config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "normalizer": normalizer.to_dict(),
            "threshold": float(threshold),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[RhythmEncoder, Normalizer, dict]:
    """Returns (model in eval mode, normalizer, {"threshold", "metrics"})."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if ckpt.get("format") != "keyauth-v1":
        raise ValueError(f"{path} is not a keyauth checkpoint")
    model = RhythmEncoder(EncoderConfig.from_dict(ckpt["encoder_config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, Normalizer.from_dict(ckpt["normalizer"]), {"threshold": ckpt["threshold"], "metrics": ckpt["metrics"]}


# ---- CLI ----------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Train the keystroke-rhythm encoder")
    ap.add_argument("--data", choices=["cmu", "synthetic"], default="cmu")
    ap.add_argument("--cmu-path", default="data/DSL-StrongPasswordData.csv")
    ap.add_argument("--out", default="checkpoints/model.pt")
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--batches-per-epoch", type=int, default=TrainConfig.batches_per_epoch)
    ap.add_argument("--n-enroll", type=int, default=TrainConfig.n_enroll)
    ap.add_argument("--seed", type=int, default=TrainConfig.seed)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    if args.data == "cmu":
        data = load_cmu(args.cmu_path)
    else:
        data = make_synthetic_dataset(n_users=40, samples_per_user=60, n_sessions=4, seed=args.seed)
    train_data, val_data, test_data = split_users(data, seed=args.seed)
    print(f"users: {len(train_data.users)} train / {len(val_data.users)} val / {len(test_data.users)} test")

    cfg = TrainConfig(
        epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        n_enroll=args.n_enroll,
        seed=args.seed,
        device=args.device,
    )
    model, normalizer, history = train(train_data, val_data, cfg)

    val = evaluate_split(model, normalizer, val_data, cfg)
    threshold = val["eer_threshold"]  # chosen on val users only
    test = evaluate_split(model, normalizer, test_data, cfg, threshold=threshold)
    print(
        f"\nTEST (unseen users, {cfg.n_enroll} enrollment attempts):\n"
        f"  EER            {test['eer'] * 100:.2f}%\n"
        f"  per-user EER   {test['per_user_eer'] * 100:.2f}%\n"
        f"  at val threshold {threshold:.4f}: FAR {test['far'] * 100:.2f}%  FRR {test['frr'] * 100:.2f}%"
    )

    metrics = {"val": val, "test": test, "train_config": asdict(cfg), "epochs_run": len(history)}
    save_checkpoint(args.out, model, normalizer, threshold, metrics)
    Path(args.out).with_suffix(".json").write_text(json.dumps({"history": history, **metrics}, indent=2))
    print(f"saved {args.out}")
    return metrics


if __name__ == "__main__":
    main()
