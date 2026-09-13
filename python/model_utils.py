"""Shared helper for loading a trained NSS checkpoint (Phase4+ eval scripts)."""

import torch

from models.nss import NSSModel, Normalizer


def load_model(checkpoint_path, hidden=(64, 64)):
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model = NSSModel(n_x=2, n_u=1, n_y=1, hidden=hidden, increment_scale=1.0)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    x_normalizer = Normalizer.from_dict(ckpt["x_normalizer"])
    u_normalizer = Normalizer.from_dict(ckpt["u_normalizer"])
    y_normalizer = Normalizer.from_dict(ckpt["y_normalizer"])
    return model, x_normalizer, u_normalizer, y_normalizer
