"""Phase4: equilibrium-point linearization / eigenvalue analysis (仕様書§6.1, F-05, M-02).

Computes the discrete-time Jacobian d(x_next)/d(x) of the trained NSS at the
equilibrium (theta=0, theta_dot=0, tau=0) and reports its eigenvalues.
Normalization is an affine (diagonal-scale) map, so eigenvalues computed in
normalized coordinates equal those in physical coordinates (similarity
transform).

Run: python python/eval_stability.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import experiment
from model_utils import load_model


def main():
    config = experiment.load_config()
    model_dir = Path(config["paths"]["models_dir"]) / "checkpoints"
    model, x_normalizer, u_normalizer, _ = load_model(model_dir / "nss_step1_rollout.pt")

    x_eq = x_normalizer.normalize(torch.tensor([0.0, 0.0]))
    u_eq = u_normalizer.normalize(torch.tensor([0.0]))

    dxnext_dx, dxnext_du = model.jacobian_at(x_eq, u_eq)
    eigvals = np.linalg.eigvals(dxnext_dx.numpy())
    spectral_radius = float(np.max(np.abs(eigvals)))

    print("Equilibrium Jacobian d(x_next)/d(x):")
    print(dxnext_dx.numpy())
    print("Eigenvalues:", eigvals)
    print(f"Spectral radius: {spectral_radius:.6f} -> "
          f"{'PASS (stable, <1)' if spectral_radius < 1 else 'FAIL (unstable, >=1)'}")
    print("d(x_next)/d(tau) at equilibrium:", dxnext_du.numpy())

    return {"eigenvalues": eigvals, "spectral_radius": spectral_radius}


if __name__ == "__main__":
    main()
