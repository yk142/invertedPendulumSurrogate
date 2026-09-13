%VERIFY_PLANT Phase1 completion check: step response, free vibration, and
%   energy monotonicity under zero input (02_仕様書.md §6.1, plan Phase1
%   exit criterion).

addpath(fileparts(mfilename('fullpath')));
params = pendulum_params();

%% Step response (tau = 50% of tau_max, from rest at theta=0)
T_step = 5; % s
N_step = round(T_step / params.Ts_ctrl);
tau_step = params.tau_max * 0.5 * ones(N_step, 1);
[t_step, theta_step, theta_dot_step] = simulate_pendulum([0; 0], tau_step, params);

figure('Name', 'Step response');
subplot(2,1,1); plot(t_step, theta_step); ylabel('\theta [rad]'); grid on;
title('Step response, \tau = 50% \tau_{max}');
subplot(2,1,2); plot(t_step, theta_dot_step); ylabel('\theta_{dot} [rad/s]'); xlabel('t [s]'); grid on;

%% Free vibration (zero input, nonzero initial condition) - passivity check
T_free = 20; % s
N_free = round(T_free / params.Ts_ctrl);
tau_free = zeros(N_free, 1);
x0_free = [0.8; 0]; % rad, within [-pi/2, pi/2]
[t_free, theta_free, theta_dot_free] = simulate_pendulum(x0_free, tau_free, params);

energy = 0.5 * params.J * theta_dot_free.^2 + params.mgl * (1 - cos(theta_free));

figure('Name', 'Free vibration + energy');
subplot(2,1,1); plot(t_free, theta_free); ylabel('\theta [rad]'); grid on;
title('Free vibration, \tau = 0, \theta_0 = 0.8 rad');
subplot(2,1,2); plot(t_free, energy); ylabel('Energy [J]'); xlabel('t [s]'); grid on;

% Energy must be monotonically non-increasing (allow tiny numerical tolerance)
d_energy = diff(energy);
tol = 1e-9;
violations = sum(d_energy > tol);
max_increase = max(d_energy);

fprintf('Free-vibration energy check: %d / %d steps increased (tol=%.1e)\n', ...
    violations, numel(d_energy), tol);
fprintf('Max single-step energy increase: %.3e J\n', max_increase);
if violations == 0
    fprintf('PASS: energy is monotonically non-increasing (dissipative, no passivity violation).\n');
else
    fprintf('FAIL: energy increased at %d steps - check integrator/parameters.\n', violations);
end
