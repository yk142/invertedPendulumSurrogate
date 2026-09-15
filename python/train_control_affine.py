"""Issue #27: train the control-affine NSS (M-03 structured NSS) and compare
against the standard (unconstrained) NSSModel on M-04 sign consistency and
the other Phase4 diagnostics.

Run: python python/train_control_affine.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

import data_utils
import experiment
from models.nss import Normalizer, NSSControlAffine
from train import (BATCH_SIZE, HORIZON_SCHEDULE, WINDOW_LENGTH, WINDOW_STRIDE,
                    make_dataset, one_step_val_error, run_epoch, train_variant)


def control_affine_factory():
    return NSSControlAffine(n_x=2, n_u=1, n_y=1, hidden=(64, 64), increment_scale=1.0)


def main():
    config, run_dir, logger = experiment.start_run("control_affine_train")
    data_dir = config["paths"]["data_dir"]

    manifest = data_utils.load_manifest(data_dir)
    train_entries = data_utils.entries_by_split(manifest, "train")
    val_entries = data_utils.entries_by_split(manifest, "val")

    x0_raw, u_raw, y_raw = data_utils.build_windows(data_dir, train_entries, WINDOW_LENGTH, WINDOW_STRIDE)
    x_normalizer = Normalizer.from_data(x0_raw)
    u_normalizer = Normalizer.from_data(u_raw.reshape(-1, u_raw.shape[-1]))
    y_normalizer = Normalizer.from_data(y_raw.reshape(-1, y_raw.shape[-1]))

    train_ds = make_dataset(data_dir, train_entries, x_normalizer, u_normalizer, y_normalizer)
    val_ds = make_dataset(data_dir, val_entries, x_normalizer, u_normalizer, y_normalizer)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model, _ = train_variant("control_affine", HORIZON_SCHEDULE, train_ds, val_ds, logger,
                              model_factory=control_affine_factory)

    one_step_err = one_step_val_error(model, val_loader, y_normalizer)
    rollout50_loss = run_epoch(model, val_loader, horizon=WINDOW_LENGTH)
    logger.info("[control_affine] 1-step val error=%.6f rad, 50-step rollout MSE(norm)=%.6f",
                one_step_err, rollout50_loss)
    logger.info("[control_affine] learned B = %s", model.B.detach().numpy().ravel())
    logger.info("[control_affine] g_phi.weight @ B = %s (sign determines global M-04 consistency)",
                (model.g_phi.weight.detach() @ model.B.detach()).numpy().ravel())

    model_dir = Path(config["paths"]["models_dir"]) / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_normalizer": x_normalizer.to_dict(),
            "u_normalizer": u_normalizer.to_dict(),
            "y_normalizer": y_normalizer.to_dict(),
        },
        model_dir / "nss_control_affine.pt",
    )

    print(f"1-step val error: {one_step_err:.6f} rad, 50-step rollout MSE(norm): {rollout50_loss:.6f}")
    print("Learned B:", model.B.detach().numpy().ravel())
    print("g_phi.weight @ B:", (model.g_phi.weight.detach() @ model.B.detach()).numpy().ravel())
    return one_step_err, rollout50_loss


if __name__ == "__main__":
    main()
