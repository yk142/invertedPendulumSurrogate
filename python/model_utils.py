"""Shared helper for loading a trained NSS checkpoint (Phase4+ eval scripts)."""

import torch

from models.nss import NSSModel, Normalizer


def load_model(checkpoint_path, hidden=(64, 64), model_cls=NSSModel, model_kwargs=None):
    """model_kwargs: extra constructor args beyond n_x/n_u/n_y/hidden/increment_scale
    (e.g. hidden_activation="relu", bounded_increment=False for the Ablation C
    checkpoint - these aren't recoverable from the state_dict alone since
    activation functions carry no parameters, so the caller must know and
    pass them explicitly)."""
    ckpt = torch.load(checkpoint_path, weights_only=False)
    n_y = len(ckpt["y_normalizer"]["mean"])  # 1 for Step1 (theta), 2 for Step2 ([theta, theta_dot])
    model = model_cls(n_x=2, n_u=1, n_y=n_y, hidden=hidden, increment_scale=1.0, **(model_kwargs or {}))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    x_normalizer = Normalizer.from_dict(ckpt["x_normalizer"])
    u_normalizer = Normalizer.from_dict(ckpt["u_normalizer"])
    y_normalizer = Normalizer.from_dict(ckpt["y_normalizer"])
    return model, x_normalizer, u_normalizer, y_normalizer
