"""Export a trained NSSControlAffine checkpoint to .mat for MATLAB (issue #29,
counterpart to export_weights.py for the standard NSSModel).

Run: python python/export_weights_control_affine.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import scipy.io as sio

import experiment
from model_utils import load_model
from models.nss import NSSControlAffine


def main():
    config = experiment.load_config()
    model_dir = Path(config["paths"]["models_dir"]) / "checkpoints"
    model, x_n, u_n, y_n = load_model(model_dir / "nss_control_affine.pt", model_cls=NSSControlAffine)

    f_drift_weights, f_drift_biases = [], []
    for layer in model.f_drift:
        if hasattr(layer, "weight"):
            f_drift_weights.append(layer.weight.detach().numpy().astype(np.float64))
            f_drift_biases.append(layer.bias.detach().numpy().astype(np.float64).reshape(-1, 1))

    payload = {
        "f_drift_weights": np.empty((len(f_drift_weights),), dtype=object),
        "f_drift_biases": np.empty((len(f_drift_biases),), dtype=object),
        "B": model.B.detach().numpy().astype(np.float64),
        "g_phi_weight": model.g_phi.weight.detach().numpy().astype(np.float64),
        "g_phi_bias": model.g_phi.bias.detach().numpy().astype(np.float64).reshape(-1, 1),
        "increment_scale": float(model.increment_scale),
        "x_mean": x_n.mean.numpy().astype(np.float64).reshape(-1, 1),
        "x_std": x_n.std.numpy().astype(np.float64).reshape(-1, 1),
        "u_mean": u_n.mean.numpy().astype(np.float64).reshape(-1, 1),
        "u_std": u_n.std.numpy().astype(np.float64).reshape(-1, 1),
        "y_mean": y_n.mean.numpy().astype(np.float64).reshape(-1, 1),
        "y_std": y_n.std.numpy().astype(np.float64).reshape(-1, 1),
    }
    for i, w in enumerate(f_drift_weights):
        payload["f_drift_weights"][i] = w
    for i, b in enumerate(f_drift_biases):
        payload["f_drift_biases"][i] = b

    out_path = Path(config["paths"]["models_dir"]) / "surrogate_control_affine_weights.mat"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(out_path), payload)
    print(f"Exported -> {out_path} ({len(f_drift_weights)} f_drift linear layers, B shape {model.B.shape})")


if __name__ == "__main__":
    main()
