"""Coverage visualization for a generated dataset (仕様書§4.2).

Loads every scenario .mat file listed in <data_dir>/manifest.json and plots
histograms of theta, theta_dot, tau to check whether the dataset covers the
intended operating range for the current data-generation step.

Run: python python/plot_coverage.py [--data-dir python/data] [--out reports/coverage.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import io_utils


def load_all(data_dir):
    manifest_path = Path(data_dir) / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    theta_all, theta_dot_all, tau_all = [], [], []
    for entry in manifest["scenarios"]:
        d = io_utils.load_dataset(Path(data_dir) / entry["file"])
        theta_all.append(d["theta"])
        theta_dot_all.append(d["theta_dot"])
        tau_all.append(d["tau"])

    return (
        np.concatenate(theta_all),
        np.concatenate(theta_dot_all),
        np.concatenate(tau_all),
        manifest,
    )


def plot_coverage(data_dir, out_path):
    theta, theta_dot, tau, manifest = load_all(data_dir)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist(theta, bins=50)
    axes[0].set_title("theta [rad]")
    axes[1].hist(theta_dot, bins=50)
    axes[1].set_title("theta_dot [rad/s]")
    axes[2].hist(tau, bins=50)
    axes[2].set_title("tau [N*m]")
    fig.suptitle(f"Coverage: {manifest['step']} (n_scenarios={manifest['n_scenarios']})")
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"theta:     min={theta.min():.3f} max={theta.max():.3f}")
    print(f"theta_dot: min={theta_dot.min():.3f} max={theta_dot.max():.3f}")
    print(f"tau:       min={tau.min():.3f} max={tau.max():.3f}")
    print(f"Saved coverage plot -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="python/data")
    parser.add_argument("--out", default="reports/coverage_step1.png")
    args = parser.parse_args()
    plot_coverage(args.data_dir, args.out)
