"""Issue #31: combine Step2 (explicitly-supervised [theta, theta_dot] output,
issue #23) with the control-affine architecture (M-04 structural fix,
issue #27). NSSControlAffine already accepts n_y as a parameter, so this
just reuses train_step2.py's data/eval plumbing with a different model_factory.

Run: python python/train_control_affine_step2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader

import data_utils
import experiment
from models.nss import NSSControlAffine, Normalizer
from train import BATCH_SIZE, HORIZON_SCHEDULE, WINDOW_LENGTH, WINDOW_STRIDE, run_epoch, train_variant
from train_step2 import make_dataset, one_step_val_error


def control_affine_step2_factory():
    return NSSControlAffine(n_x=2, n_u=1, n_y=2, hidden=(64, 64), increment_scale=1.0)


def main():
    config, run_dir, logger = experiment.start_run("control_affine_step2_train")
    data_dir = config["paths"]["data_dir"]

    manifest = data_utils.load_manifest(data_dir)
    train_entries = data_utils.entries_by_split(manifest, "train")
    val_entries = data_utils.entries_by_split(manifest, "val")

    x0_raw, u_raw, y_raw = data_utils.build_windows(data_dir, train_entries, WINDOW_LENGTH,
                                                      WINDOW_STRIDE, include_theta_dot=True)
    x_normalizer = Normalizer.from_data(x0_raw)
    u_normalizer = Normalizer.from_data(u_raw.reshape(-1, u_raw.shape[-1]))
    y_normalizer = Normalizer.from_data(y_raw.reshape(-1, y_raw.shape[-1]))
    logger.info("x_normalizer=%s u_normalizer=%s y_normalizer=%s",
                x_normalizer.to_dict(), u_normalizer.to_dict(), y_normalizer.to_dict())

    train_ds = make_dataset(data_dir, train_entries, x_normalizer, u_normalizer, y_normalizer)
    val_ds = make_dataset(data_dir, val_entries, x_normalizer, u_normalizer, y_normalizer)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # issue #31: the unbounded linear B*u term combined with 50-step BPTT caused
    # a training instability (loss spike, never recovered) at default lr/no clipping.
    # Lower lr + gradient-norm clipping stabilizes it.
    model, _ = train_variant("control_affine_step2", HORIZON_SCHEDULE, train_ds, val_ds, logger,
                              model_factory=control_affine_step2_factory, lr=3e-4, grad_clip=1.0)

    one_step_err = one_step_val_error(model, val_loader, y_normalizer)
    rollout50_loss = run_epoch(model, val_loader, horizon=WINDOW_LENGTH)
    logger.info("1-step val error: theta=%.6f rad, theta_dot=%.6f rad/s; "
                "50-step rollout MSE(norm)=%.6f", one_step_err["theta_rad"],
                one_step_err["theta_dot_rad_s"], rollout50_loss)
    logger.info("learned B = %s", model.B.detach().numpy().ravel())
    logger.info("g_phi.weight[0] @ B (theta row) = %s",
                (model.g_phi.weight.detach()[0:1] @ model.B.detach()).numpy().ravel())

    model_dir = Path(config["paths"]["models_dir"]) / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_normalizer": x_normalizer.to_dict(),
            "u_normalizer": u_normalizer.to_dict(),
            "y_normalizer": y_normalizer.to_dict(),
        },
        model_dir / "nss_control_affine_step2.pt",
    )

    print("1-step val error:", one_step_err, "50-step rollout MSE(norm):", rollout50_loss)
    print("Learned B:", model.B.detach().numpy().ravel())
    print("g_phi.weight[0] @ B (theta):", (model.g_phi.weight.detach()[0:1] @ model.B.detach()).numpy().ravel())
    return one_step_err, rollout50_loss


if __name__ == "__main__":
    main()
