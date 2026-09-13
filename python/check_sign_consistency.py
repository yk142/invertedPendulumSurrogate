"""Phase4: input-output direction consistency diagnostic (M-04, 仕様書§3.3/§6.1).

Computes d(theta_next)/d(tau) = g_phi.weight @ d(x_next)/d(tau) at every
sampled point along the validation trajectories (teacher-forced: uses the
true (theta, theta_dot, tau) as the operating point, per M-04's "動作軌道上
の各時刻で" requirement) and reports the fraction where the sign disagrees
with the physical model (which is always positive: more torque -> more
angle). No constraint is added to the model - this is diagnostic only.

Run: python python/check_sign_consistency.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import data_utils
import experiment
from model_utils import load_model


def batched_dy_du(model, x_batch, u_batch):
    """d(y_next)/d(u) for each sample in the batch (independent samples, so
    summing over the batch before calling autograd.grad still yields the
    correct per-sample gradient)."""
    u_batch = u_batch.clone().requires_grad_(True)
    x_next = model.step(x_batch, u_batch)
    grads = []
    for j in range(x_next.shape[1]):
        g = torch.autograd.grad(x_next[:, j].sum(), u_batch, retain_graph=True)[0]
        grads.append(g[:, 0])
    dxnext_du = torch.stack(grads, dim=1)
    dy_du = dxnext_du @ model.g_phi.weight[0]
    return dy_du.detach()


def main():
    config = experiment.load_config()
    data_dir = config["paths"]["data_dir"]
    model_dir = Path(config["paths"]["models_dir"]) / "checkpoints"
    model, x_normalizer, u_normalizer, _ = load_model(model_dir / "nss_step1_rollout.pt")

    manifest = data_utils.load_manifest(data_dir)
    val_entries = data_utils.entries_by_split(manifest, "val")

    theta_all, theta_dot_all, tau_all, sign_all = [], [], [], []
    for entry in val_entries:
        d = data_utils.load_decimated_scenario(data_dir, entry["file"])
        theta, theta_dot, tau = d["theta"][:-1], d["theta_dot"][:-1], d["tau"][:-1]

        x = x_normalizer.normalize(
            torch.from_numpy(np.stack([theta, theta_dot], axis=1).astype(np.float32))
        )
        u = u_normalizer.normalize(torch.from_numpy(tau.astype(np.float32)).unsqueeze(-1))
        dy_du = batched_dy_du(model, x, u).numpy()

        theta_all.append(theta)
        theta_dot_all.append(theta_dot)
        tau_all.append(tau)
        sign_all.append(dy_du)

    theta_all = np.concatenate(theta_all)
    theta_dot_all = np.concatenate(theta_dot_all)
    sign_all = np.concatenate(sign_all)

    n_total = len(sign_all)
    n_reversed = int(np.sum(sign_all <= 0))
    print(f"Sign-consistency check (d(theta_next)/d(tau) > 0 expected): "
          f"{n_reversed} / {n_total} points reversed or zero "
          f"({100 * n_reversed / n_total:.2f}%)")
    if n_reversed > 0:
        print(f"Reversed-region theta range: "
              f"[{theta_all[sign_all <= 0].min():.3f}, {theta_all[sign_all <= 0].max():.3f}] rad")

    fig, ax = plt.subplots(figsize=(7, 6))
    ok = sign_all > 0
    ax.scatter(theta_all[ok], theta_dot_all[ok], s=4, c="tab:blue", label="sign OK (d theta_next/d tau > 0)")
    if n_reversed > 0:
        ax.scatter(theta_all[~ok], theta_dot_all[~ok], s=8, c="tab:red", label="sign REVERSED")
    ax.set_xlabel("theta [rad]")
    ax.set_ylabel("theta_dot [rad/s]")
    ax.set_title("M-04 sign-consistency map (validation trajectories)")
    ax.legend()
    ax.grid(True)

    out_path = Path("reports/phase4_sign_consistency.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")

    return {"n_total": n_total, "n_reversed": n_reversed}


if __name__ == "__main__":
    main()
