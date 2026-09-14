function manifest = gen_training_data(n_scenarios, seed, out_dir, n_free_vibration, n_free_vibration_highvel)
%GEN_TRAINING_DATA Phase2 Step1: PTP feedforward torque + free-vibration auxiliary data.
%   manifest = gen_training_data(n_scenarios, seed, out_dir, n_free_vibration)
%
%   Two scenario types are generated (仕様書§4.1):
%     1. PTP feedforward: sample a random PTP move, build a trapezoidal-
%        velocity reference trajectory, compute the inverse-dynamics
%        feedforward torque, saturate it, and run it through the real
%        plant (simulate_pendulum) to record the actual (theta, theta_dot,
%        tau) response. This is the Step1 standard configuration.
%     2. Free vibration (auxiliary, all stages): random initial condition,
%        tau=0 throughout. Without this, the dataset never contains true
%        zero-torque samples away from theta=0 (PTP dwell segments still
%        apply gravity-compensation torque), which was found in Phase4 to
%        leave the trained surrogate without a correctly-damped zero-input
%        response away from theta=0 (see issue #10).
%
%   All scenarios are split train/val/test (70/15/15) together, matching
%   仕様書§4.3. Returns the manifest struct and also writes it as JSON to
%   <out_dir>/manifest.json.

if nargin < 1 || isempty(n_scenarios)
    n_scenarios = 60;
end
if nargin < 2 || isempty(seed)
    seed = 42;
end
if nargin < 3 || isempty(out_dir)
    out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'python', 'data');
end
if nargin < 4 || isempty(n_free_vibration)
    n_free_vibration = 20;
end
if nargin < 5 || isempty(n_free_vibration_highvel)
    n_free_vibration_highvel = 0;
end

addpath(fileparts(mfilename('fullpath')));
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

rng(seed);
params = pendulum_params();
utils = export_utils();

margin = 0.05 * (params.theta_max - params.theta_min); % keep clear of hard range edges
lo = params.theta_min + margin;
hi = params.theta_max - margin;

n_total = n_scenarios + n_free_vibration + n_free_vibration_highvel;
entries = struct('file', {}, 'theta_start', {}, 'theta_end', {}, 'duration', {});

for i = 1:n_scenarios
    theta_start = lo + (hi - lo) * rand();
    theta_end = lo + (hi - lo) * rand();
    while abs(theta_end - theta_start) < 0.2 % avoid degenerate near-zero moves
        theta_end = lo + (hi - lo) * rand();
    end

    vmax = 1.0 + 1.0 * rand();   % 1.0-2.0 rad/s
    amax = 3.0 + 3.0 * rand();   % 3.0-6.0 rad/s^2
    T_total = 10 + 4 * rand();   % 10-14 s (仕様書§4.3: 10-60s range, lower end used for Step1 runtime)

    [theta_ref, theta_dot_ref, theta_ddot_ref, ~] = ...
        ptp_trajectory(theta_start, theta_end, vmax, amax, params.Ts_ctrl, T_total);

    tau_ff = params.J * theta_ddot_ref + params.b * theta_dot_ref + params.mgl * sin(theta_ref);
    tau_ff = saturate_torque(tau_ff, params);

    % tau_ff has one more sample than the number of steps needed by
    % simulate_pendulum (which takes N torque samples and returns N+1
    % states); drop the last sample so state length matches trajectory length.
    tau_seq = tau_ff(1:end-1);
    [t, theta, theta_dot] = simulate_pendulum([theta_start; 0], tau_seq, params);

    meta = struct('Ts', params.Ts_ctrl, 'seed', seed, 'scenario_id', i, ...
        'scenario_type', 'ptp_feedforward', 'theta_start', theta_start, ...
        'theta_end', theta_end, 'vmax', vmax, 'amax', amax);

    filename = sprintf('scenario_%03d.mat', i);
    filepath = fullfile(out_dir, filename);
    tau_logged = [tau_seq; tau_seq(end)]; % pad to match t/theta length
    utils.export_dataset(filepath, t, theta, theta_dot, tau_logged, meta);

    entries(i).file = filename;
    entries(i).theta_start = theta_start;
    entries(i).theta_end = theta_end;
    entries(i).duration = T_total;
end

for j = 1:n_free_vibration
    i = n_scenarios + j;
    theta0 = lo + (hi - lo) * rand();
    theta_dot0 = -2.0 + 4.0 * rand(); % +-2 rad/s, matches PTP-observed theta_dot range
    T_total = 10; % s, several damping time constants (see matlab/verify_plant.m)

    N = round(T_total / params.Ts_ctrl);
    tau_seq = zeros(N, 1);
    [t, theta, theta_dot] = simulate_pendulum([theta0; theta_dot0], tau_seq, params);

    meta = struct('Ts', params.Ts_ctrl, 'seed', seed, 'scenario_id', i, ...
        'scenario_type', 'free_vibration_zero_input', 'theta_start', theta0, ...
        'theta_end', NaN, 'theta_dot_start', theta_dot0);

    filename = sprintf('scenario_%03d.mat', i);
    filepath = fullfile(out_dir, filename);
    tau_logged = [tau_seq; tau_seq(end)];
    utils.export_dataset(filepath, t, theta, theta_dot, tau_logged, meta);

    entries(i).file = filename;
    entries(i).theta_start = theta0;
    entries(i).theta_end = NaN;
    entries(i).duration = T_total;
end

% High-velocity-biased free vibration (Issue #25): |theta_dot0| drawn from
% [1.2, 2.0] rad/s instead of the uniform [-2,2] above, to thicken coverage
% of the fast-swing region where M-04 sign reversal was found concentrated.
for j = 1:n_free_vibration_highvel
    i = n_scenarios + n_free_vibration + j;
    theta0 = lo + (hi - lo) * rand();
    speed = 1.2 + 0.8 * rand();
    theta_dot0 = speed * sign(rand() - 0.5);
    T_total = 10;

    N = round(T_total / params.Ts_ctrl);
    tau_seq = zeros(N, 1);
    [t, theta, theta_dot] = simulate_pendulum([theta0; theta_dot0], tau_seq, params);

    meta = struct('Ts', params.Ts_ctrl, 'seed', seed, 'scenario_id', i, ...
        'scenario_type', 'free_vibration_zero_input_highvel', 'theta_start', theta0, ...
        'theta_end', NaN, 'theta_dot_start', theta_dot0);

    filename = sprintf('scenario_%03d.mat', i);
    filepath = fullfile(out_dir, filename);
    tau_logged = [tau_seq; tau_seq(end)];
    utils.export_dataset(filepath, t, theta, theta_dot, tau_logged, meta);

    entries(i).file = filename;
    entries(i).theta_start = theta0;
    entries(i).theta_end = NaN;
    entries(i).duration = T_total;
end

% Shuffle scenario order and split 70/15/15 across both scenario types together.
order = randperm(n_total);
n_train = round(0.70 * n_total);
n_val = round(0.15 * n_total);

splits = repmat({'test'}, 1, n_total);
splits(order(1:n_train)) = {'train'};
splits(order(n_train+1 : n_train+n_val)) = {'val'};

for i = 1:n_total
    entries(i).split = splits{i};
end

manifest.seed = seed;
manifest.n_scenarios = n_total;
manifest.n_ptp_feedforward = n_scenarios;
manifest.n_free_vibration = n_free_vibration;
manifest.n_free_vibration_highvel = n_free_vibration_highvel;
manifest.step = 'Step1_ptp_feedforward_plus_free_vibration_aux';
manifest.Ts = params.Ts_ctrl;
manifest.scenarios = entries;

manifest_path = fullfile(out_dir, 'manifest.json');
fid = fopen(manifest_path, 'w');
fwrite(fid, jsonencode(manifest, 'PrettyPrint', true));
fclose(fid);

fprintf('Generated %d scenarios (%d PTP + %d free-vibration + %d high-vel free-vibration) -> %s (train=%d, val=%d, test=%d)\n', ...
    n_total, n_scenarios, n_free_vibration, n_free_vibration_highvel, out_dir, n_train, n_val, n_total - n_train - n_val);

end
