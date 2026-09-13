"""Visualize the Phase3 finding: n1-only vs rollout-schedule trained NSS
models, rolled out 50 steps on a validation window, vs ground truth.

Run: python python/plot_rollout_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import data_utils
import experiment
from models.nss import NSSModel, Normalizer

WINDOW_LENGTH = 50


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model = NSSModel(n_x=2, n_u=1, n_y=1, hidden=(64, 64), increment_scale=1.0)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    x_normalizer = Normalizer.from_dict(ckpt["x_normalizer"])
    u_normalizer = Normalizer.from_dict(ckpt["u_normalizer"])
    y_normalizer = Normalizer.from_dict(ckpt["y_normalizer"])
    return model, x_normalizer, u_normalizer, y_normalizer


def main():
    config = experiment.load_config()
    data_dir = config["paths"]["data_dir"]
    manifest = data_utils.load_manifest(data_dir)
    val_entries = data_utils.entries_by_split(manifest, "val")

    # Pick one validation window with a nontrivial trajectory.
    x0, u, y = data_utils.build_windows(data_dir, val_entries[:1], WINDOW_LENGTH, stride=1)
    idx = len(x0) // 2
    x0_sample = torch.from_numpy(x0[idx : idx + 1])
    u_sample = torch.from_numpy(u[idx : idx + 1])
    y_true = y[idx, :, 0]

    model_dir = Path(config["paths"]["models_dir"]) / "checkpoints"
    fig, ax = plt.subplots(figsize=(8, 5))
    t = range(1, WINDOW_LENGTH + 1)
    ax.plot(t, y_true, "k-", linewidth=2, label="ground truth (physical plant)")

    for name, style in [("n1", "r--"), ("rollout", "b--")]:
        model, x_normalizer, u_normalizer, y_normalizer = load_model(model_dir / f"nss_step1_{name}.pt")
        with torch.no_grad():
            x0_n = x_normalizer.normalize(x0_sample)
            u_n = u_normalizer.normalize(u_sample)
            y_hat_n, _ = model.rollout(x0_n, u_n)
            y_hat = y_normalizer.denormalize(y_hat_n)[0, :, 0].numpy()
        ax.plot(t, y_hat, style, linewidth=1.5, label=f"{name} prediction")

    ax.set_xlabel("step (200Hz, 50 steps = 0.25s)")
    ax.set_ylabel("theta [rad]")
    ax.set_title("50-step rollout: n1-only vs rollout-schedule NSS vs ground truth")
    ax.legend()
    ax.grid(True)

    out_path = Path("reports/phase3_rollout_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
