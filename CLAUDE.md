# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This repository started as planning documents only; implementation now proceeds phase-by-phase
per `03_実装計画書.md`. Phase 0-3 are done (env/interchange, physical plant, Step1 PTP
feedforward dataset, Step1 NSS training). Keep this section and the commands below current as
later phases land.

## Common commands

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Phase0: environment + MATLAB<->Python interchange + experiment framework sanity check
python python/verify_env.py

# Phase1: physical plant check (step response, free-vibration energy monotonicity) - run via
# MATLAB MCP (mcp__matlab__run_matlab_file) or MATLAB desktop, not plain `matlab` CLI
#   matlab/verify_plant.m

# Phase2 Step1: generate the PTP-feedforward training dataset (60 scenarios, seed 42) -
# via MATLAB MCP (mcp__matlab__evaluate_matlab_code), addpath('matlab') first:
#   gen_training_data(60, 42)
python python/plot_coverage.py --data-dir python/data --out reports/coverage_step1.png

# Phase3: train the NSS surrogate (n1 baseline + rollout-schedule variant), ~4 min on CPU
python python/train.py
```

Generated datasets (`python/data/*.mat`), model checkpoints (`python/models/checkpoints/*.pt`),
and experiment run directories (`experiments/*/`) are gitignored — regenerate them with the
commands above rather than expecting them to be checked in. `python/data/manifest.json` and
`reports/*.png` (small, and useful as recorded evidence) are the exception and are committed.

## Development workflow (required)

For every unit of work (a plan phase, a sub-task, a fix):

1. **Open a GitHub issue first** describing the task and which phase/requirement it maps to
   (`gh issue create`).
2. **Create a branch off `main`** for that issue before writing any code (e.g.
   `phase0/python-env-setup`, `issue-12-nss-model`).
3. Do the work, commit on that branch, push, and merge back to `main` (PR or direct merge, per
   user preference at the time) — reference the issue number in the commit/PR.
4. Don't commit directly to `main` for substantive work; `main` receives merges only.

This applies to all future phases (Phase 0 onward) — don't batch multiple phases into one branch
or one issue.

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
experiments/ablation_A_F/        # results of the 6 divergence-cause ablation experiments (A-E plus F)
reports/report_template.md       # auto-generated report template (dimension-agnostic)
docs/                             # copies of the 3 planning documents
```

## Key technical constraints to preserve when implementing

- **Multi-rate control/surrogate design (仕様書 §6.4)**: the physical plant and controller run
  at 8kHz (Ts_ctrl = 125μs) — this is the baseline everything else is defined relative to. The
  NSS surrogate may run at the same 8kHz or be downsampled to 200Hz (40:1 decimation). When
  downsampled, torque input to the surrogate is sampled with zero-order hold at the 200Hz
  update instant (not averaged), and the surrogate's output is held (ZOH) for the controller
  until the next 200Hz update — this adds up to 5ms of feedback latency that must be accounted
  for in gain tuning. Ablation experiment D is about *unintended* rate mismatches (e.g.
  inconsistent ZOH convention between training and runtime), not about the sanctioned 8kHz/200Hz
  split itself.
- **Training data generation is staged, not built for full coverage upfront** (仕様書 §4.1):
  Step1 = PTP feedforward torque waveforms only. Only if Phase5 closed-loop testing shows
  divergence/insufficient accuracy do you add Step2 (actual closed-loop torque waveforms
  recorded from running PTP with the controller) and, if still insufficient, Step3 (PRBS/chirp
  broadband signals). Don't jump straight to broadband excitation — that skips the diagnostic
  value of seeing which step fixes it.
- **Surrogate output form is also staged** (仕様書 §3.1): start with y_k = θ (angle) directly.
  Only if accuracy/stability is insufficient, switch to outputting θ_dot (and/or θ_ddot) and
  numerically integrate to reconstruct θ for the controller feedback — and when doing so,
  evaluate integration drift explicitly (checked during the long-horizon rollout test, §6.1).
- **Training loss must use multi-step rollout, not 1-step-only prediction** (仕様書 §5.1) — a
  scheduled horizon (1→5→20→50 steps), since 1-step-only loss is Ablation B (a known way to
  induce divergence, not the default training method).
- **Prefer bounded activations (tanh/SiLU) over ReLU** in the state-transition network — 仕様書
  §3.2 notes ReLU extrapolates linearly and is more prone to divergence outside the training
  distribution (this is also Ablation C).
- **Do NOT add an input-output monotonicity/sign constraint to the NSS model yet** (M-04,
  仕様書 §3.3). The current leading divergence hypothesis is that with no such constraint, the
  surrogate can predict a state change with the same sign as the controller's corrective torque,
  turning error correction into a positive-feedback runaway. The fix (a direction-constrained
  architecture) is deliberately deferred — first ship the unconstrained baseline (M-01's scale
  constraint only), instrument it with the M-04 sign-consistency diagnostic (log the sign of
  τ_fb vs. Δŷ at every step), and only design the constrained architecture if Ablation
  experiment F (which *is* the unconstrained baseline, not a separate degraded variant)
  confirms the hypothesis.
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
