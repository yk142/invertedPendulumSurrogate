"""Dataset loading, 8kHz->200Hz decimation, and windowing for NSS training.

Decimation follows the multi-rate ZOH convention in 02_仕様書.md §6.4: the
surrogate rate (200Hz) sample is the instantaneous value at the update
instant, not an average over the 40 control-rate samples it spans.
"""

import json
from pathlib import Path

import numpy as np

import io_utils

RATIO_8K_TO_200HZ = 40  # 8000 Hz / 200 Hz


def load_manifest(data_dir):
    with open(Path(data_dir) / "manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_decimated_scenario(data_dir, filename, ratio=RATIO_8K_TO_200HZ):
    d = io_utils.load_dataset(Path(data_dir) / filename)
    return {
        "t": d["t"][::ratio],
        "theta": d["theta"][::ratio],
        "theta_dot": d["theta_dot"][::ratio],
        "tau": d["tau"][::ratio],
    }


def build_windows(data_dir, entries, window_length=50, stride=10, ratio=RATIO_8K_TO_200HZ,
                   include_theta_dot=False):
    """Build fixed-length training windows from a list of manifest scenario entries.

    Returns x0 (N,2) [theta_0, theta_dot_0], u_win (N, window_length, 1) tau
    sequence, y_win (N, window_length, n_y) target sequence (y_{k+1..k+N}).
    n_y=1 (theta only, Step1) unless include_theta_dot=True, in which case
    n_y=2 ([theta, theta_dot], Step2 - 仕様書§3.1) with theta_dot explicitly
    supervised rather than left as an unconstrained internal state (issue #21).
    """
    x0_list, u_list, y_list = [], [], []
    for entry in entries:
        d = load_decimated_scenario(data_dir, entry["file"], ratio)
        theta, theta_dot, tau = d["theta"], d["theta_dot"], d["tau"]
        m = len(theta)
        for i in range(0, m - window_length - 1, stride):
            x0_list.append([theta[i], theta_dot[i]])
            u_list.append(tau[i : i + window_length])
            if include_theta_dot:
                y_list.append(np.stack(
                    [theta[i + 1 : i + window_length + 1], theta_dot[i + 1 : i + window_length + 1]],
                    axis=-1))
            else:
                y_list.append(theta[i + 1 : i + window_length + 1])

    x0 = np.asarray(x0_list, dtype=np.float32)
    u = np.asarray(u_list, dtype=np.float32)[..., None]
    y = np.asarray(y_list, dtype=np.float32)
    if not include_theta_dot:
        y = y[..., None]
    return x0, u, y


def entries_by_split(manifest, split):
    return [e for e in manifest["scenarios"] if e["split"] == split]
