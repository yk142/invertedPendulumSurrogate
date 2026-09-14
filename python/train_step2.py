"""Step2 (仕様書§3.1): NSS trained with an explicitly-supervised [theta, theta_dot]
output (g_phi has n_y=2), instead of Step1's theta-only output.

This replaces the ad-hoc reuse of the model's unsupervised internal state as
a theta_dot estimate (issue #19) with a properly-trained one (issue #21's
follow-up), using the same rollout-schedule training as Step1.

Run: python python/train_step2.py
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
from train import BATCH_SIZE, HORIZON_SCHEDULE, LEARNING_RATE, WINDOW_LENGTH, WINDOW_STRIDE, run_epoch

EPOCHS_PER_STAGE = 20


def make_dataset(data_dir, entries, x_normalizer, u_normalizer, y_normalizer):
    x0, u, y = data_utils.build_windows(data_dir, entries, WINDOW_LENGTH, WINDOW_STRIDE,
                                          include_theta_dot=True)
    x0_n = x_normalizer.normalize(torch.from_numpy(x0))
    u_n = u_normalizer.normalize(torch.from_numpy(u))
    y_n = y_normalizer.normalize(torch.from_numpy(y))
    return TensorDataset(x0_n, u_n, y_n)


def one_step_val_error(model, val_loader, y_normalizer):
    """1-step prediction error on the validation set, per output channel
    (theta in rad, theta_dot in rad/s)."""
    model.eval()
    errors = []
    with torch.no_grad():
        for x0, u, y in val_loader:
            y_hat_n, _ = model.rollout(x0, u[:, :1])
            y_hat = y_normalizer.denormalize(y_hat_n[:, 0])
            y_true = y_normalizer.denormalize(y[:, 0])
            errors.append((y_hat - y_true).abs().numpy())
    errors = np.concatenate(errors, axis=0)
    return {"theta_rad": float(errors[:, 0].mean()), "theta_dot_rad_s": float(errors[:, 1].mean())}


def main():
    config, run_dir, logger = experiment.start_run("step2_train")
    data_dir = config["paths"]["data_dir"]

    manifest = data_utils.load_manifest(data_dir)
    train_entries = data_utils.entries_by_split(manifest, "train")
    val_entries = data_utils.entries_by_split(manifest, "val")

    x0_raw, u_raw, y_raw = data_utils.build_windows(data_dir, train_entries, WINDOW_LENGTH,
                                                      WINDOW_STRIDE, include_theta_dot=True)
    x_normalizer = Normalizer.from_data(x0_raw)
    u_normalizer = Normalizer.from_data(u_raw.reshape(-1, u_raw.shape[-1]))
    y_normalizer = Normalizer.from_data(y_raw.reshape(-1, y_raw.shape[-1]))
    logger.info("x_normalizer=%s u_normalizer=%s y_normalizer=%s",
                x_normalizer.to_dict(), u_normalizer.to_dict(), y_normalizer.to_dict())

    train_ds = make_dataset(data_dir, train_entries, x_normalizer, u_normalizer, y_normalizer)
    val_ds = make_dataset(data_dir, val_entries, x_normalizer, u_normalizer, y_normalizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = NSSModel(n_x=2, n_u=1, n_y=2, hidden=(64, 64), increment_scale=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for stage, horizon in enumerate(HORIZON_SCHEDULE):
        for epoch in range(EPOCHS_PER_STAGE):
            train_loss = run_epoch(model, train_loader, horizon, optimizer)
            val_loss = run_epoch(model, val_loader, horizon)
            logger.info("stage=%d horizon=%d epoch=%d train_loss=%.6f val_loss=%.6f",
                        stage, horizon, epoch, train_loss, val_loss)

    one_step_err = one_step_val_error(model, val_loader, y_normalizer)
    rollout50_loss = run_epoch(model, val_loader, horizon=WINDOW_LENGTH)
    logger.info("1-step val error: theta=%.6f rad, theta_dot=%.6f rad/s; "
                "50-step rollout MSE(norm)=%.6f", one_step_err["theta_rad"],
                one_step_err["theta_dot_rad_s"], rollout50_loss)

    model_dir = Path(config["paths"]["models_dir"]) / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_normalizer": x_normalizer.to_dict(),
            "u_normalizer": u_normalizer.to_dict(),
            "y_normalizer": y_normalizer.to_dict(),
        },
        model_dir / "nss_step2_thetadot.pt",
    )

    print("Step2 1-step validation error:", one_step_err,
          "50-step rollout MSE(norm):", rollout50_loss)
    return one_step_err


if __name__ == "__main__":
    main()
