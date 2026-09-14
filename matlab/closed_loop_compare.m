function results = closed_loop_compare(Kp, Kd, theta_target, T_total, weights_path, runtime_ratio, theta_dot_source)
%CLOSED_LOOP_COMPARE Phase5: physical-plant vs. NSS-surrogate closed loop (仕様書§6.2, §7).
%   results = closed_loop_compare(Kp, Kd, theta_target, T_total, weights_path, runtime_ratio)
%
%   Same PD controller (angle error + finite-difference angle-rate error,
%   both derived from the theta measurement only - so the comparison stays
%   fair, since Step1's surrogate output is theta-only, 仕様書§3.1) drives:
%     - the physical plant at Ts_ctrl (8kHz)
%     - the NSS surrogate, updated only every Ts_surr/Ts_ctrl steps (200Hz,
%       ZOH input/output per 仕様書§6.4), introducing up to one surrogate
%       period of feedback latency that the physical loop does not have.
%
%   Saturation (saturate_torque) is applied identically to both loops
%   (仕様書§7).
%
%   runtime_ratio (optional): control-steps-per-surrogate-update actually
%   used at runtime. Defaults to the trained ratio (Ts_surr/Ts_ctrl = 40).
%   Passing a different value reproduces Ablation D (仕様書§6.3): the model
%   internally still assumes each of its own steps spans Ts_surr=5ms (that
%   is baked into its trained weights), but the wall-clock time between
%   surrogate calls is actually runtime_ratio*Ts_ctrl - an unintended
%   sampling-period mismatch between training and runtime.

if nargin < 1 || isempty(Kp), Kp = 0.35; end
if nargin < 2 || isempty(Kd), Kd = 0.09; end
if nargin < 3 || isempty(theta_target), theta_target = 0.8; end
if nargin < 4 || isempty(T_total), T_total = 8; end
if nargin < 5 || isempty(weights_path)
    weights_path = fullfile(fileparts(mfilename('fullpath')), '..', 'python', 'models', 'surrogate_step1_weights.mat');
end

addpath(fileparts(mfilename('fullpath')));
params = pendulum_params();
nss = nss_surrogate();
w = nss.load(weights_path);

trained_ratio = round(params.Ts_surr / params.Ts_ctrl); % 40 (8kHz / 200Hz)
if nargin < 6 || isempty(runtime_ratio)
    runtime_ratio = trained_ratio;
end
ratio = runtime_ratio;
N = round(T_total / params.Ts_ctrl);
if nargin < 7 || isempty(theta_dot_source)
    theta_dot_source = 'finite_difference'; % or 'internal_state' (issue #19) or 'step2_output' (issue #23, requires a Step2 weights file)
end

[theta_ref, theta_dot_ref, ~, t] = ptp_trajectory(0, theta_target, 1.5, 4.0, params.Ts_ctrl, T_total);

% --- Physical closed loop ---
x_phys = [0; 0];
theta_phys = zeros(N + 1, 1);
tau_phys = zeros(N, 1);
theta_prev = 0;
for k = 1:N
    e = theta_ref(k) - x_phys(1);
    theta_dot_est = (x_phys(1) - theta_prev) / params.Ts_ctrl;
    theta_prev = x_phys(1);
    tau_cmd = saturate_torque(Kp * e + Kd * (theta_dot_ref(k) - theta_dot_est), params);
    x_phys = pendulum_step(x_phys, tau_cmd, params);
    theta_phys(k + 1) = x_phys(1);
    tau_phys(k) = tau_cmd;
end

% --- Surrogate closed loop (ZOH multi-rate, 仕様書§6.4) ---
x_surr = [0; 0];
y_surr = 0;
y_prev = 0;
theta_dot_surr_est = 0;
theta_surr = zeros(N + 1, 1);
tau_surr = zeros(N, 1);
for k = 1:N
    e = theta_ref(k) - y_surr;
    tau_cmd = saturate_torque(Kp * e + Kd * (theta_dot_ref(k) - theta_dot_surr_est), params);
    tau_surr(k) = tau_cmd;

    if mod(k - 1, ratio) == 0
        [x_surr, y_new] = nss.step(w, x_surr, tau_cmd);
        y_theta = y_new(1);
        if strcmp(theta_dot_source, 'internal_state')
            theta_dot_surr_est = x_surr(2); % model's own unsupervised latent (issue #21 caveat)
        elseif strcmp(theta_dot_source, 'step2_output')
            theta_dot_surr_est = y_new(2); % explicitly-supervised theta_dot output (仕様書§3.1 Step2)
        else
            theta_dot_surr_est = (y_theta - y_prev) / (ratio * params.Ts_ctrl);
        end
        y_prev = y_theta;
        y_surr = y_theta;
    end
    theta_surr(k + 1) = y_surr;
end

results.t = t;
results.theta_ref = theta_ref;
results.theta_phys = theta_phys;
results.theta_surr = theta_surr;
results.tau_phys = tau_phys;
results.tau_surr = tau_surr;
results.diverged_surr = max(abs(theta_surr)) > 10;
results.final_error_phys = abs(theta_ref(end) - theta_phys(end));
results.final_error_surr = abs(theta_ref(end) - theta_surr(end));
results.max_abs_theta_surr = max(abs(theta_surr));

fprintf('Physical: final error = %.4f rad, max|theta| = %.4f rad\n', ...
    results.final_error_phys, max(abs(theta_phys)));
fprintf('Surrogate: final error = %.4f rad, max|theta| = %.4f rad -> %s\n', ...
    results.final_error_surr, results.max_abs_theta_surr, ...
    ternary(results.diverged_surr, 'DIVERGED', 'bounded'));

figure('Name', 'Closed-loop PTP comparison');
subplot(2,1,1);
plot(t, theta_ref, 'k--', t, theta_phys, 'b-', t, theta_surr, 'r-');
legend('reference', 'physical closed loop', 'surrogate closed loop', 'Location', 'best');
ylabel('\theta [rad]'); grid on;
title(sprintf('PTP closed-loop comparison (Kp=%.3f, Kd=%.3f, target=%.2f rad)', Kp, Kd, theta_target));
subplot(2,1,2);
plot(t(1:end-1), tau_phys, 'b-', t(1:end-1), tau_surr, 'r-');
legend('physical', 'surrogate', 'Location', 'best');
ylabel('\tau [N\cdotm]'); xlabel('t [s]'); grid on;

end

function out = ternary(cond, a, b)
if cond
    out = a;
else
    out = b;
end
end
