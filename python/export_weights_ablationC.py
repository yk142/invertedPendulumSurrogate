"""Export the Ablation C (ReLU hidden, unbounded increment) checkpoint to
.mat for MATLAB closed-loop verification (issue #33, follow-up to #15).

Run: python python/export_weights_ablationC.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import scipy.io as sio

import experiment
from model_utils import load_model


def main():
    config = experiment.load_config()
    model_dir = Path(config["paths"]["models_dir"]) / "checkpoints"
    # This checkpoint was trained with hidden_activation="relu",
    # bounded_increment=False (ablation_c_activation.py); load_model must be
    # told explicitly since activation functions carry no state_dict entries
    # and so can't be recovered from the checkpoint alone.
    model, x_normalizer, u_normalizer, y_normalizer = load_model(
        model_dir / "nss_step1_ablationC_relu.pt",
        model_kwargs={"hidden_activation": "relu", "bounded_increment": False},
    )

    f_theta_weights, f_theta_biases = [], []
    for layer in model.f_theta:
        if hasattr(layer, "weight"):
            f_theta_weights.append(layer.weight.detach().numpy().astype(np.float64))
            f_theta_biases.append(layer.bias.detach().numpy().astype(np.float64).reshape(-1, 1))

    payload = {
        "f_theta_weights": np.empty((len(f_theta_weights),), dtype=object),
        "f_theta_biases": np.empty((len(f_theta_biases),), dtype=object),
        "g_phi_weight": model.g_phi.weight.detach().numpy().astype(np.float64),
        "g_phi_bias": model.g_phi.bias.detach().numpy().astype(np.float64).reshape(-1, 1),
        "increment_scale": float(model.increment_scale),
        "bounded_increment": float(model.bounded_increment),  # 0 for this ablation
        "x_mean": x_normalizer.mean.numpy().astype(np.float64).reshape(-1, 1),
        "x_std": x_normalizer.std.numpy().astype(np.float64).reshape(-1, 1),
        "u_mean": u_normalizer.mean.numpy().astype(np.float64).reshape(-1, 1),
        "u_std": u_normalizer.std.numpy().astype(np.float64).reshape(-1, 1),
        "y_mean": y_normalizer.mean.numpy().astype(np.float64).reshape(-1, 1),
        "y_std": y_normalizer.std.numpy().astype(np.float64).reshape(-1, 1),
    }
    for i, w in enumerate(f_theta_weights):
        payload["f_theta_weights"][i] = w
    for i, b in enumerate(f_theta_biases):
        payload["f_theta_biases"][i] = b

    out_path = Path(config["paths"]["models_dir"]) / "surrogate_ablationC_weights.mat"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(out_path), payload)
    print(f"Exported -> {out_path} ({len(f_theta_weights)} f_theta linear layers, "
          f"bounded_increment={model.bounded_increment})")


if __name__ == "__main__":
    main()
