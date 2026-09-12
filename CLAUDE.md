# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This repository currently contains only planning documents (in Japanese) — no code has been
written yet. There is no MATLAB, Python, build, lint, or test tooling in place. When
implementation begins, follow the directory layout and phase plan below, and update this file
with real commands (test runners, MATLAB entry points, Python env setup) once they exist.

## What this project is

A principle-verification (PoC) project for surrogate-modeling a 6-axis robot manipulator
(torque input → angle output) with a Neural State Space (NSS) model, to diagnose why NSS
surrogates diverge when embedded in closed-loop control. Rather than debugging the full 6-axis
system directly, the project uses a **single inverted pendulum** (1-DOF, nonlinear,
torque-in/angle-out) as a reduced test case to isolate which of five candidate root causes is
responsible for closed-loop divergence:

1. Training data design (coverage gaps, covariate shift)
2. NSS dynamical instability (lack of passivity / energy conservation)
3. Training method (1-step-only prediction loss, vulnerable to error accumulation on rollout)
4. Discretization / sampling-period mismatch between training and control loop
5. Controller gain design mismatched to surrogate characteristics

This is Stage 2 of a 4-stage program: Stage1 (mass-spring-damper, linear) → **Stage2 (this repo:
nonlinear pendulum)** → Stage3 (2-link planar manipulator, coupled inertia/Coriolis) → Stage4
(full 6-axis robot). Code and metrics should be written to generalize to higher state
dimensions (non-functional requirement N-03) since Stage3/4 will reuse this framework.

## Source documents (read these before implementing)

- `01_要件定義書.md` — Requirements: scope, functional requirements F-01..F-10, non-functional
  requirements N-01..N-05, and the acceptance criteria that define when this PoC phase is done.
- `02_仕様書.md` — **Single source of truth for equations, parameters, and pass/fail
  thresholds.** Contains the pendulum equation of motion, state-space form, NSS architecture
  spec, training loss (multi-step rollout loss, not 1-step), dataset spec, open-loop and
  closed-loop validation criteria, and the 5 ablation experiments (A–E) used to reproduce
  divergence causes.
- `03_実装計画書.md` — Implementation plan: intended directory layout, 8 phases (Phase 0–7)
  with deliverables/exit criteria per phase, and tooling choices.

## Intended architecture (per 03_実装計画書.md)

Hybrid MATLAB/Python split: **MATLAB/Simulink** owns the physical plant model, closed-loop
simulation, and visualization; **Python (PyTorch)** owns NSS training. Data crosses the
boundary as `.mat` files (MATLAB → `scipy.io.loadmat`), and trained models cross back as
ONNX/TorchScript imported into MATLAB via Deep Learning Toolbox's `importONNXNetwork`.

Planned directory structure:

```
matlab/
├── plant_pendulum.m / .slx      # physical plant (equation of motion in 02_仕様書.md §2)
├── controller_pid.m / .slx      # PID or state-feedback controller
├── gen_training_data.m          # excitation signal generation (step/PRBS/chirp/PTP/free-vibration)
├── closed_loop_compare.m        # physical-vs-surrogate closed-loop comparison
└── export_utils.m               # .mat I/O utilities
python/
├── data/                        # converted training data
├── models/nss.py                # NSS model: x_{k+1}=f_θ(x_k,u_k), y_k=g_φ(x_k)
├── train.py                     # training with rollout-horizon loss scheduling
├── eval_openloop.py             # long-horizon free rollout stability check
├── eval_stability.py            # equilibrium linearization / eigenvalue analysis
└── export_onnx.py               # ONNX export for MATLAB import
experiments/ablation_A_E/        # results of the 5 divergence-cause ablation experiments
reports/report_template.md       # auto-generated report template (dimension-agnostic)
docs/                             # copies of the 3 planning documents
```

## Key technical constraints to preserve when implementing

- **Sampling period consistency**: physical model integration step, training data Ts, NSS
  discrete time step, and closed-loop control period must all match (仕様書 §6.4). Ablation
  experiment D specifically tests what happens when they don't — don't accidentally "fix" this
  by resampling without recording it.
- **Training loss must use multi-step rollout, not 1-step-only prediction** (仕様書 §5.1) — a
  scheduled horizon (1→5→20→50 steps), since 1-step-only loss is Ablation B (a known way to
  induce divergence, not the default training method).
- **Prefer bounded activations (tanh/SiLU) over ReLU** in the state-transition network — 仕様書
  §3.2 notes ReLU extrapolates linearly and is more prone to divergence outside the training
  distribution (this is also Ablation C).
- Every trained NSS model must have its equilibrium-point Jacobian eigenvalues computed and
  reported (requirement M-02) — this is a required diagnostic, not optional analysis.
- Fix random seeds for reproducibility (N-01); keep dataset/hyperparameter/model/metric
  provenance linked (N-05) — experiment tracking is a stated requirement, not an afterthought.
- Actuator saturation (`τ_max`) must be clipped identically for both the physical-model loop and
  the surrogate loop when comparing closed-loop responses (仕様書 §7) — an unfair comparison
  invalidates Phase 5/6 conclusions.

## Working with MATLAB in this repo

MATLAB tools are available via MCP (`mcp__matlab__*`) against a locally running, user-visible
MATLAB instance — graphical output (plots, Simulink) appears in the MATLAB desktop, not in this
terminal. Use `check_matlab_code` for static analysis and `run_matlab_test_file` for
`matlab.unittest`-based tests once test files exist. Do not run MATLAB code that alters files or
environment settings without confirming with the user first.
