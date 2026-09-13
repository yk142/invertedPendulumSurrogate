"""Ablation C (仕様書§6.3): tanh (bounded increment) vs ReLU (unbounded increment).

Trains a second NSS variant with hidden_activation="relu" and
bounded_increment=False (no M-01 tanh saturation on the state increment),
using the same rollout schedule as the standard model, then compares
long-horizon free-rollout stability both within and beyond the training
data's theta range - ReLU's linear extrapolation is expected to blow up
where tanh saturates.

Run: python python/ablation_c_activation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import data_utils
import experiment
from energy_check import compute_energy
from eval_openloop import DIVERGENCE_THRESHOLD_RAD, N_STEPS, TS_SURR, free_rollout
from models.nss import NSSModel, Normalizer
from train import (BATCH_SIZE, HORIZON_SCHEDULE, WINDOW_LENGTH, WINDOW_STRIDE,
                    make_dataset, one_step_val_error, run_epoch, train_variant)


def relu_unbounded_factory():
    return NSSModel(n_x=2, n_u=1, n_y=1, hidden=(64, 64), increment_scale=1.0,
                     hidden_activation="relu", bounded_increment=False)


def main():
    config, run_dir, logger = experiment.start_run("ablation_c_activation")
    data_dir = config["paths"]["data_dir"]

    manifest = data_utils.load_manifest(data_dir)
    train_entries = data_utils.entries_by_split(manifest, "train")
    val_entries = data_utils.entries_by_split(manifest, "val")

    x0_raw, u_raw, y_raw = data_utils.build_windows(data_dir, train_entries, WINDOW_LENGTH, WINDOW_STRIDE)
    x_normalizer = Normalizer.from_data(x0_raw)
    u_normalizer = Normalizer.from_data(u_raw.reshape(-1, u_raw.shape[-1]))
    y_normalizer = Normalizer.from_data(y_raw.reshape(-1, y_raw.shape[-1]))

    train_ds = make_dataset(data_dir, train_entries, x_normalizer, u_normalizer, y_normalizer)
    val_ds = make_dataset(data_dir, val_entries, x_normalizer, u_normalizer, y_normalizer)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model, _ = train_variant("ablationC_relu", HORIZON_SCHEDULE, train_ds, val_ds, logger,
                              model_factory=relu_unbounded_factory)
    one_step_err = one_step_val_error(model, val_loader, y_normalizer)
    rollout50_loss = run_epoch(model, val_loader, horizon=WINDOW_LENGTH)
    logger.info("[ablationC_relu] 1-step val error=%.6f rad, 50-step rollout MSE(norm)=%.6f",
                one_step_err, rollout50_loss)

    model_dir = Path(config["paths"]["models_dir"]) / "checkpoints"
    torch.save({"state_dict": model.state_dict(), "x_normalizer": x_normalizer.to_dict(),
                "u_normalizer": u_normalizer.to_dict(), "y_normalizer": y_normalizer.to_dict()},
               model_dir / "nss_step1_ablationC_relu.pt")

    # In-distribution vs. out-of-distribution free rollout (仕様書§4.3 coverage ~[-1.4, 1.4] rad).
    plant = config["plant"]
    cases = [("in-distribution (theta0=0.8)", 0.8), ("out-of-distribution (theta0=2.5)", 2.5)]
    fig, axes = plt.subplots(len(cases), 1, figsize=(9, 3 * len(cases)))
    results = {}
    for ax, (label, theta0) in zip(axes, cases):
        theta_hat, theta_dot_hat = free_rollout(model, x_normalizer, u_normalizer, theta0, 0.0, N_STEPS)
        max_abs = float(np.max(np.abs(theta_hat)))
        diverged = max_abs > DIVERGENCE_THRESHOLD_RAD
        results[label] = {"max_abs_theta": max_abs, "diverged": diverged}
        logger.info("[ablationC_relu] %s: max|theta_hat|=%.4f -> %s", label, max_abs,
                    "DIVERGED" if diverged else "bounded")
        t = np.arange(N_STEPS) * TS_SURR
        ax.plot(t, theta_hat)
        ax.set_title(f"{label}: max|theta|={max_abs:.2f} rad")
        ax.set_ylabel("theta_hat [rad]")
        ax.grid(True)
    axes[-1].set_xlabel("t [s]")
    fig.suptitle("Ablation C: ReLU + unbounded increment, free rollout (tau=0)")
    fig.tight_layout()

    out_path = Path("reports/phase6_ablationC_relu_rollout.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")
    print("Results:", results)
    return results


if __name__ == "__main__":
    main()
