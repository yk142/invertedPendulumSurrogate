"""Phase0 environment/framework verification script.

Checks:
  1. Required packages import and report their versions.
  2. experiment.start_run produces a run dir with config + log (seed/provenance framework).
  3. io_utils dataset round-trip (save then load) preserves values.

Run: python python/verify_env.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import experiment
import io_utils


def check_versions():
    import matplotlib
    import scipy
    import torch
    import yaml

    print("numpy      ", np.__version__)
    print("scipy      ", scipy.__version__)
    print("torch      ", torch.__version__)
    print("matplotlib ", matplotlib.__version__)
    print("pyyaml     ", yaml.__version__)


def check_experiment_framework():
    config, run_dir, logger = experiment.start_run("verify_env")
    logger.info("experiment framework OK")
    assert (run_dir / "config.json").exists()
    assert (run_dir / "run.log").exists()
    print("experiment framework OK ->", run_dir)


def check_io_roundtrip():
    t = np.linspace(0, 1, 100)
    theta = np.sin(t)
    theta_dot = np.cos(t)
    tau = np.zeros_like(t)
    meta = {"Ts": 0.01, "seed": 42, "scenario": "verify_env"}

    tmp_path = Path("python/data/_verify_roundtrip.mat")
    io_utils.save_dataset(tmp_path, t, theta, theta_dot, tau, meta)
    loaded = io_utils.load_dataset(tmp_path)

    assert np.allclose(loaded["t"], t)
    assert np.allclose(loaded["theta"], theta)
    assert np.allclose(loaded["theta_dot"], theta_dot)
    assert np.allclose(loaded["tau"], tau)
    tmp_path.unlink()
    print("io_utils round-trip OK")


if __name__ == "__main__":
    check_versions()
    check_experiment_framework()
    check_io_roundtrip()
    print("Phase0 verification complete.")
