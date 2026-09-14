"""Extrapolation / out-of-distribution monitor (仕様書§4.2, F-07).

Builds a k-nearest-neighbor distance reference from the training (theta,
theta_dot, tau) points and flags query points whose distance to the
training distribution exceeds a threshold calibrated on the validation set.
Intended to run alongside a closed-loop simulation so an operator can see
when the surrogate has left its trained region before divergence becomes
visible in the output.

Run: python python/extrapolation_monitor.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

import data_utils
import experiment
from eval_openloop import N_STEPS, TS_SURR, free_rollout
from model_utils import load_model


class ExtrapolationMonitor:
    def __init__(self, reference_points, k=5, threshold=None):
        self.mean = reference_points.mean(axis=0)
        self.std = reference_points.std(axis=0).clip(min=1e-6)
        self.tree = cKDTree(self._standardize(reference_points))
        self.k = k
        self.threshold = threshold

    def _standardize(self, points):
        return (points - self.mean) / self.std

    def distance(self, query_points):
        d, _ = self.tree.query(self._standardize(query_points), k=self.k)
        return d[:, -1] if d.ndim > 1 else d

    def calibrate_threshold(self, calibration_points, percentile=99):
        d = self.distance(calibration_points)
        self.threshold = float(np.percentile(d, percentile))
        return self.threshold

    def is_extrapolating(self, query_points):
        return self.distance(query_points) > self.threshold


def build_reference_points(data_dir, entries, subsample=5):
    theta_all, theta_dot_all, tau_all = [], [], []
    for entry in entries:
        d = data_utils.load_decimated_scenario(data_dir, entry["file"])
        theta_all.append(d["theta"][::subsample])
        theta_dot_all.append(d["theta_dot"][::subsample])
        tau_all.append(d["tau"][::subsample])
    return np.stack([np.concatenate(theta_all), np.concatenate(theta_dot_all),
                      np.concatenate(tau_all)], axis=1)


def main():
    config = experiment.load_config()
    data_dir = config["paths"]["data_dir"]
    model_dir = Path(config["paths"]["models_dir"]) / "checkpoints"

    manifest = data_utils.load_manifest(data_dir)
    train_points = build_reference_points(data_dir, data_utils.entries_by_split(manifest, "train"))
    val_points = build_reference_points(data_dir, data_utils.entries_by_split(manifest, "val"))

    monitor = ExtrapolationMonitor(train_points, k=5)
    threshold = monitor.calibrate_threshold(val_points, percentile=99)
    print(f"Extrapolation distance threshold (99th pct of val-set k=5 NN distance): {threshold:.4f}")

    model, x_normalizer, u_normalizer, _ = load_model(model_dir / "nss_step1_rollout.pt")
    theta_hat, theta_dot_hat = free_rollout(model, x_normalizer, u_normalizer, 0.8, 0.0, N_STEPS)
    tau_zero = np.zeros_like(theta_hat)
    query_points = np.stack([theta_hat, theta_dot_hat, tau_zero], axis=1)

    dist = monitor.distance(query_points)
    flagged = dist > threshold
    print(f"Free-rollout (theta0=0.8, tau=0, 100s): {flagged.sum()} / {len(flagged)} "
          f"steps flagged as extrapolating ({100 * flagged.mean():.2f}%)")
    if flagged.any():
        first_flag_t = np.argmax(flagged) * TS_SURR
        print(f"First flagged at t={first_flag_t:.3f}s")

    t = np.arange(N_STEPS) * TS_SURR
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t, theta_hat, "b-")
    axes[0].fill_between(t, theta_hat.min(), theta_hat.max(), where=flagged, color="red", alpha=0.2,
                          label="flagged extrapolating")
    axes[0].set_ylabel("theta_hat [rad]")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(t, dist, "k-")
    axes[1].axhline(threshold, color="red", linestyle="--", label="threshold")
    axes[1].set_ylabel("k-NN distance")
    axes[1].set_xlabel("t [s]")
    axes[1].legend()
    axes[1].grid(True)
    fig.suptitle("Extrapolation monitor on free rollout (theta0=0.8, tau=0)")
    fig.tight_layout()

    out_path = Path("reports/phase6_extrapolation_monitor.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
