"""Seed management and experiment run tracking (N-01 reproducibility, N-05 traceability).

Every experiment run gets its own timestamped directory under experiments/
containing a copy of the resolved config and a log file, so inputs
(dataset, hyperparameters) and outputs (model, metrics) stay linked.
"""

import json
import logging
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def new_run_dir(name, config, experiments_dir=None):
    """Create experiments/<experiments_dir>/<timestamp>_<name>/ and save the config used."""
    base = Path(experiments_dir or config.get("paths", {}).get("experiments_dir", "experiments"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return run_dir


def setup_logging(run_dir):
    logger = logging.getLogger(str(run_dir))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(Path(run_dir) / "run.log", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in (file_handler, stream_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def start_run(name, config_path="config.yaml"):
    """Convenience entry point: load config, fix seed, create run dir + logger."""
    config = load_config(config_path)
    set_seed(config["seed"])
    run_dir = new_run_dir(name, config)
    shutil.copyfile(config_path, run_dir / "config_source.yaml")
    logger = setup_logging(run_dir)
    logger.info("Run '%s' started, seed=%s, run_dir=%s", name, config["seed"], run_dir)
    return config, run_dir, logger
