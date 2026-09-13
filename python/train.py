"""Phase3: NSS training with rollout-horizon loss scheduling (仕様書§5.1).

Trains two variants for comparison (Ablation B baseline vs. the standard
schedule):
  - "n1":      N=1 fixed throughout (1-step loss only)
  - "rollout": horizon schedule 1 -> 5 -> 20 -> 50

Both use the Step1 output form (y_k = theta_k) and the unconstrained M-01
architecture (no M-04 direction constraint - see CLAUDE.md).

Run: python python/train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import data_utils
import experiment
from models.nss import NSSModel, Normalizer

HORIZON_SCHEDULE = [1, 5, 20, 50]
EPOCHS_PER_STAGE = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WINDOW_LENGTH = 50
WINDOW_STRIDE = 10


def make_dataset(data_dir, entries, x_normalizer, u_normalizer, y_normalizer):
    x0, u, y = data_utils.build_windows(data_dir, entries, WINDOW_LENGTH, WINDOW_STRIDE)
    x0_n = x_normalizer.normalize(torch.from_numpy(x0))
    u_n = u_normalizer.normalize(torch.from_numpy(u))
    y_n = y_normalizer.normalize(torch.from_numpy(y))
    return TensorDataset(x0_n, u_n, y_n)


def run_epoch(model, loader, horizon, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, n_batches = 0.0, 0
    for x0, u, y in loader:
        u_h = u[:, :horizon]
        y_h = y[:, :horizon]
        y_hat, _ = model.rollout(x0, u_h)
        loss = torch.mean((y_hat - y_h) ** 2)
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def one_step_val_error(model, val_loader, y_normalizer):
    """1-step prediction error on the validation set, in physical units (rad)."""
    model.eval()
    errors = []
    with torch.no_grad():
        for x0, u, y in val_loader:
            y_hat_n, _ = model.rollout(x0, u[:, :1])
            y_hat = y_normalizer.denormalize(y_hat_n[:, 0])
            y_true = y_normalizer.denormalize(y[:, 0])
            errors.append((y_hat - y_true).abs().numpy())
    return float(np.mean(np.concatenate(errors)))


def train_variant(name, schedule, train_ds, val_ds, logger):
    model = NSSModel(n_x=2, n_u=1, n_y=1, hidden=(64, 64), increment_scale=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    history = []
    for stage, horizon in enumerate(schedule):
        for epoch in range(EPOCHS_PER_STAGE):
            train_loss = run_epoch(model, train_loader, horizon, optimizer)
            val_loss = run_epoch(model, val_loader, horizon)
            history.append({"stage": stage, "horizon": horizon, "epoch": epoch,
                             "train_loss": train_loss, "val_loss": val_loss})
            logger.info("[%s] stage=%d horizon=%d epoch=%d train_loss=%.6f val_loss=%.6f",
                        name, stage, horizon, epoch, train_loss, val_loss)
    return model, history


def main():
    config, run_dir, logger = experiment.start_run("phase3_train")
    data_dir = config["paths"]["data_dir"]

    manifest = data_utils.load_manifest(data_dir)
    train_entries = data_utils.entries_by_split(manifest, "train")
    val_entries = data_utils.entries_by_split(manifest, "val")

    x0_raw, u_raw, y_raw = data_utils.build_windows(data_dir, train_entries, WINDOW_LENGTH, WINDOW_STRIDE)
    x_normalizer = Normalizer.from_data(x0_raw)
    u_normalizer = Normalizer.from_data(u_raw.reshape(-1, u_raw.shape[-1]))
    y_normalizer = Normalizer.from_data(y_raw.reshape(-1, y_raw.shape[-1]))
    logger.info("x_normalizer=%s u_normalizer=%s y_normalizer=%s",
                x_normalizer.to_dict(), u_normalizer.to_dict(), y_normalizer.to_dict())

    train_ds = make_dataset(data_dir, train_entries, x_normalizer, u_normalizer, y_normalizer)
    val_ds = make_dataset(data_dir, val_entries, x_normalizer, u_normalizer, y_normalizer)

    results = {}
    for name, schedule in [("n1", [1] * len(HORIZON_SCHEDULE)), ("rollout", HORIZON_SCHEDULE)]:
        model, history = train_variant(name, schedule, train_ds, val_ds, logger)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        one_step_err = one_step_val_error(model, val_loader, y_normalizer)
        rollout50_loss = run_epoch(model, val_loader, horizon=WINDOW_LENGTH)
        logger.info("[%s] final 1-step validation error: %.6f rad (normalized 50-step rollout MSE: %.6f)",
                    name, one_step_err, rollout50_loss)
        results[name] = {"one_step_err_rad": one_step_err, "rollout50_mse_norm": rollout50_loss}

        model_dir = Path(config["paths"]["models_dir"]) / "checkpoints"
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "x_normalizer": x_normalizer.to_dict(),
                "u_normalizer": u_normalizer.to_dict(),
                "y_normalizer": y_normalizer.to_dict(),
            },
            model_dir / f"nss_step1_{name}.pt",
        )

    logger.info("Comparison: %s", results)
    print("Comparison (1-step err rad / 50-step rollout MSE, normalized):", results)
    return results


if __name__ == "__main__":
    main()
