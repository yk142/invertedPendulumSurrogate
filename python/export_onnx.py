"""Phase5: export the trained NSS surrogate to ONNX for MATLAB import (仕様書§8).

Wraps the model's one-step transition in physical units (theta, theta_dot,
tau in, theta_next/theta_dot_next/y out) so MATLAB does not need to
replicate the normalization logic - it only needs to hold the 2-dim state
between calls and feed torque in at the surrogate's own rate (仕様書§6.4).

Run: python python/export_onnx.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn

import experiment
from model_utils import load_model


class NSSStepPhysical(nn.Module):
    """Physical-units one-step wrapper: (x_phys, tau_phys) -> (x_next_phys, y_phys)."""

    def __init__(self, model, x_normalizer, u_normalizer, y_normalizer):
        super().__init__()
        self.model = model
        self.register_buffer("x_mean", x_normalizer.mean)
        self.register_buffer("x_std", x_normalizer.std)
        self.register_buffer("u_mean", u_normalizer.mean)
        self.register_buffer("u_std", u_normalizer.std)
        self.register_buffer("y_mean", y_normalizer.mean)
        self.register_buffer("y_std", y_normalizer.std)

    def forward(self, x_phys, u_phys):
        x_n = (x_phys - self.x_mean) / self.x_std
        u_n = (u_phys - self.u_mean) / self.u_std
        x_next_n = self.model.step(x_n, u_n)
        y_n = self.model.output(x_next_n)
        x_next_phys = x_next_n * self.x_std + self.x_mean
        y_phys = y_n * self.y_std + self.y_mean
        return x_next_phys, y_phys


def main():
    config = experiment.load_config()
    model_dir = Path(config["paths"]["models_dir"]) / "checkpoints"
    model, x_normalizer, u_normalizer, y_normalizer = load_model(model_dir / "nss_step1_rollout.pt")

    wrapper = NSSStepPhysical(model, x_normalizer, u_normalizer, y_normalizer)
    wrapper.eval()

    x_example = torch.zeros(1, 2)
    u_example = torch.zeros(1, 1)

    out_path = Path(config["paths"]["models_dir"]) / "surrogate_step1.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (x_example, u_example),
        str(out_path),
        input_names=["x_phys", "u_phys"],
        output_names=["x_next_phys", "y_phys"],
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported -> {out_path}")

    # Sanity check: compare wrapper output vs. the underlying model directly.
    with torch.no_grad():
        x_test = torch.tensor([[0.5, -0.3]])
        u_test = torch.tensor([[0.1]])
        x_next, y = wrapper(x_test, u_test)
        print("wrapper(x=[0.5,-0.3], u=0.1) ->", x_next.numpy(), y.numpy())


if __name__ == "__main__":
    main()
