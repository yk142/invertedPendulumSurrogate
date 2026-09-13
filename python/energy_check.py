"""Zero-input energy monotonicity check (仕様書§6.1), reusable by eval_openloop.py.

energy = 0.5*J*theta_dot^2 + mgl*(1-cos(theta)), matching the physical-plant
definition used in matlab/verify_plant.m.
"""

import numpy as np


def compute_energy(theta, theta_dot, J, mgl):
    theta = np.asarray(theta)
    theta_dot = np.asarray(theta_dot)
    return 0.5 * J * theta_dot**2 + mgl * (1 - np.cos(theta))


def check_monotonic_nonincreasing(energy, tol=1e-6):
    d_energy = np.diff(energy)
    violations = int(np.sum(d_energy > tol))
    max_increase = float(np.max(d_energy)) if len(d_energy) else 0.0
    return {
        "n_steps": len(d_energy),
        "violations": violations,
        "max_increase": max_increase,
        "passed": violations == 0,
    }
