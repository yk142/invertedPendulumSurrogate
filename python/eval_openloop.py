"""Phase4: long-horizon free (tau=0) rollout stability check (仕様書§6.1, F-04).

Rolls the trained NSS surrogate forward 100s (at its 200Hz operating rate)
from initial conditions within the training range, with zero torque input,
and checks:
  - boundedness (no divergence)
  - zero-input energy monotonicity (uses the model's internal second state
    channel as a theta_dot proxy - see caveat in the module docstring below)

Run: python python/eval_openloop.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import experiment
from energy_check import check_monotonic_nonincreasing, compute_energy
from model_utils import load_model

DURATION_S = 100.0
TS_SURR = 1 / 200.0
N_STEPS = int(DURATION_S / TS_SURR)

# Divergence threshold: physical theta_max is pi/2 (~1.57 rad); anything well
# beyond the training range (仕様書§4.3 coverage ~[-1.4, 1.3] rad) is a failure.
DIVERGENCE_THRESHOLD_RAD = 10.0


def free_rollout(model, x_normalizer, u_normalizer, theta0, theta_dot0, n_steps):
    x0 = x_normalizer.normalize(torch.tensor([[theta0, theta_dot0]], dtype=torch.float32))
    u_zero = u_normalizer.normalize(torch.zeros((1, 1), dtype=torch.float32))

    theta_hat = np.zeros(n_steps)
    theta_dot_hat = np.zeros(n_steps)
    x = x0
    with torch.no_grad():
        for k in range(n_steps):
            x = model.step(x, u_zero)
            x_phys = x_normalizer.denormalize(x)
            theta_hat[k] = x_phys[0, 0].item()
            theta_dot_hat[k] = x_phys[0, 1].item()
    return theta_hat, theta_dot_hat


def main():
    config = experiment.load_config()
    model_dir = Path(config["paths"]["models_dir"]) / "checkpoints"
    plant = config["plant"]

    model, x_normalizer, u_normalizer, _ = load_model(model_dir / "nss_step1_rollout.pt")

    theta0, theta_dot0 = 0.8, 0.0  # matches matlab/verify_plant.m free-vibration test
    theta_hat, theta_dot_hat = free_rollout(model, x_normalizer, u_normalizer, theta0, theta_dot0, N_STEPS)

    max_abs_theta = float(np.max(np.abs(theta_hat)))
    diverged = max_abs_theta > DIVERGENCE_THRESHOLD_RAD
    print(f"100s free rollout: max|theta_hat| = {max_abs_theta:.4f} rad "
          f"(threshold {DIVERGENCE_THRESHOLD_RAD}) -> {'FAIL (diverged)' if diverged else 'PASS'}")

    energy = compute_energy(theta_hat, theta_dot_hat, plant["J"], plant["mgl"])
    energy_result = check_monotonic_nonincreasing(energy, tol=1e-4)
    print(f"Zero-input energy check: {energy_result['violations']} / {energy_result['n_steps']} "
          f"steps increased (max increase {energy_result['max_increase']:.3e} J) -> "
          f"{'PASS' if energy_result['passed'] else 'FAIL'}")
    print("Note: theta_dot_hat is the model's internal second state channel, "
          "not a guaranteed physical quantity - this is a diagnostic proxy.")

    t = np.arange(N_STEPS) * TS_SURR
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t, theta_hat)
    axes[0].set_ylabel("theta_hat [rad]")
    axes[0].set_title(f"NSS free rollout (100s, tau=0, theta0={theta0} rad)")
    axes[0].grid(True)
    axes[1].plot(t, energy)
    axes[1].set_ylabel("pseudo-energy [J]")
    axes[1].set_xlabel("t [s]")
    axes[1].grid(True)
    fig.tight_layout()

    out_path = Path("reports/phase4_openloop_rollout.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")

    return {"diverged": diverged, "max_abs_theta": max_abs_theta, "energy_check": energy_result}


if __name__ == "__main__":
    main()
