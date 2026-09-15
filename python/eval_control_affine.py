"""Phase4-equivalent diagnostics for the control-affine NSS (issue #27),
compared against the standard NSSModel baseline.

Run: python python/eval_control_affine.py
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
from energy_check import check_monotonic_nonincreasing, compute_energy
from model_utils import load_model
from models.nss import NSSControlAffine

DURATION_S = 100.0
TS_SURR = 1 / 200.0
N_STEPS = int(DURATION_S / TS_SURR)
DIVERGENCE_THRESHOLD_RAD = 10.0


def free_rollout(model, x_normalizer, u_normalizer, theta0, theta_dot0, n_steps):
    x = x_normalizer.normalize(torch.tensor([[theta0, theta_dot0]], dtype=torch.float32))
    u_zero = u_normalizer.normalize(torch.zeros((1, 1), dtype=torch.float32))
    theta_hat = np.zeros(n_steps)
    theta_dot_hat = np.zeros(n_steps)
    with torch.no_grad():
        for k in range(n_steps):
            x = model.step(x, u_zero)
            x_phys = x_normalizer.denormalize(x)
            theta_hat[k] = x_phys[0, 0].item()
            theta_dot_hat[k] = x_phys[0, 1].item()
    return theta_hat, theta_dot_hat


def sign_consistency(model, x_normalizer, u_normalizer, data_dir, entries):
    theta_all, sign_all = [], []
    for entry in entries:
        d = data_utils.load_decimated_scenario(data_dir, entry["file"])
        theta, theta_dot, tau = d["theta"][:-1], d["theta_dot"][:-1], d["tau"][:-1]
        x = x_normalizer.normalize(torch.from_numpy(np.stack([theta, theta_dot], axis=1).astype(np.float32)))
        u = u_normalizer.normalize(torch.from_numpy(tau.astype(np.float32)).unsqueeze(-1)).clone().requires_grad_(True)
        x_next = model.step(x, u)
        grads = []
        for j in range(x_next.shape[1]):
            g = torch.autograd.grad(x_next[:, j].sum(), u, retain_graph=True)[0]
            grads.append(g[:, 0])
        dxdu = torch.stack(grads, dim=1)
        dydu = (dxdu @ model.g_phi.weight[0]).detach().numpy()
        theta_all.append(theta)
        sign_all.append(dydu)
    return np.concatenate(theta_all), np.concatenate(sign_all)


def main():
    config = experiment.load_config()
    data_dir = config["paths"]["data_dir"]
    model_dir = Path(config["paths"]["models_dir"]) / "checkpoints"
    plant = config["plant"]

    model, x_n, u_n, _ = load_model(model_dir / "nss_control_affine.pt", model_cls=NSSControlAffine)
    print("Learned B:", model.B.detach().numpy().ravel())
    print("g_phi.weight @ B:", (model.g_phi.weight.detach() @ model.B.detach()).numpy().ravel())

    theta0, theta_dot0 = 0.8, 0.0
    theta_hat, theta_dot_hat = free_rollout(model, x_n, u_n, theta0, theta_dot0, N_STEPS)
    max_abs_theta = float(np.max(np.abs(theta_hat)))
    print(f"100s free rollout: max|theta_hat|={max_abs_theta:.4f} rad -> "
          f"{'FAIL (diverged)' if max_abs_theta > DIVERGENCE_THRESHOLD_RAD else 'PASS'}")
    print(f"final theta_hat={theta_hat[-1]:.4f} rad")

    energy = compute_energy(theta_hat, theta_dot_hat, plant["J"], plant["mgl"])
    energy_result = check_monotonic_nonincreasing(energy, tol=1e-4)
    print(f"Zero-input energy check: {energy_result['violations']}/{energy_result['n_steps']} "
          f"increased -> {'PASS' if energy_result['passed'] else 'FAIL'}")

    x_eq = x_n.normalize(torch.tensor([0.0, 0.0]))
    u_eq = u_n.normalize(torch.tensor([0.0]))
    dxdx, _ = model.jacobian_at(x_eq, u_eq)
    eig = np.linalg.eigvals(dxdx.numpy())
    spectral_radius = float(np.max(np.abs(eig)))
    print(f"Equilibrium spectral radius: {spectral_radius:.6f} -> "
          f"{'PASS' if spectral_radius < 1 else 'FAIL'}")

    manifest = data_utils.load_manifest(data_dir)
    val_entries = data_utils.entries_by_split(manifest, "val")
    theta_all, sign_all = sign_consistency(model, x_n, u_n, data_dir, val_entries)
    n_reversed = int(np.sum(sign_all <= 0))
    print(f"M-04 sign consistency: {n_reversed}/{len(sign_all)} reversed "
          f"({100 * n_reversed / len(sign_all):.2f}%)")

    fig, ax = plt.subplots(figsize=(9, 4))
    t = np.arange(N_STEPS) * TS_SURR
    ax.plot(t, theta_hat)
    ax.set_ylabel("theta_hat [rad]")
    ax.set_xlabel("t [s]")
    ax.set_title(f"Control-affine NSS free rollout (100s, tau=0, theta0={theta0} rad)")
    ax.grid(True)
    out_path = Path("reports/control_affine_openloop_rollout.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
