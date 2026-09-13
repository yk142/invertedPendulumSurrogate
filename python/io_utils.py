"""MATLAB <-> Python .mat interchange utilities.

Counterpart to matlab/export_utils.m. Datasets are plain .mat files whose
top-level fields are t, theta, theta_dot, tau (all 1-D) plus a meta struct
(Ts, seed, scenario name, ...).
"""

from pathlib import Path

import numpy as np
import scipy.io as sio


def load_dataset(path):
    """Load a dataset .mat file produced by matlab/export_utils.m.

    Returns a dict with numpy arrays t, theta, theta_dot, tau and a dict meta.
    """
    raw = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    dataset = {
        "t": np.asarray(raw["t"], dtype=float).reshape(-1),
        "theta": np.asarray(raw["theta"], dtype=float).reshape(-1),
        "theta_dot": np.asarray(raw["theta_dot"], dtype=float).reshape(-1),
        "tau": np.asarray(raw["tau"], dtype=float).reshape(-1),
    }
    meta_raw = raw.get("meta")
    dataset["meta"] = _matstruct_to_dict(meta_raw)
    return dataset


def save_dataset(path, t, theta, theta_dot, tau, meta=None):
    """Save a dataset .mat file readable by matlab/export_utils.m's import_dataset."""
    payload = {
        "t": np.asarray(t, dtype=float).reshape(-1, 1),
        "theta": np.asarray(theta, dtype=float).reshape(-1, 1),
        "theta_dot": np.asarray(theta_dot, dtype=float).reshape(-1, 1),
        "tau": np.asarray(tau, dtype=float).reshape(-1, 1),
        "meta": meta or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(path), payload)


def _matstruct_to_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "_fieldnames"):
        return {name: getattr(obj, name) for name in obj._fieldnames}
    return {}
