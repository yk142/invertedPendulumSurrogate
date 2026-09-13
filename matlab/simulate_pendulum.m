function [t, theta, theta_dot] = simulate_pendulum(x0, tau_seq, params)
%SIMULATE_PENDULUM Fixed-step (RK4) propagation at the base control period Ts_ctrl.
%   [t, theta, theta_dot] = simulate_pendulum(x0, tau_seq, params)
%   x0: initial state [theta0; theta_dot0]
%   tau_seq: N-by-1 torque command, held constant (zero-order hold) over
%            each Ts_ctrl step. Saturation must already be applied by the
%            caller (or via saturate_torque) if desired.
%
%   Uses a fixed-step RK4 integrator (rather than ode45) so that the
%   piecewise-constant torque input aligns exactly with the control period,
%   matching how the closed-loop simulation will drive the plant
%   (仕様書§2.4, §6.4).

N = numel(tau_seq);
Ts = params.Ts_ctrl;

theta = zeros(N + 1, 1);
theta_dot = zeros(N + 1, 1);
theta(1) = x0(1);
theta_dot(1) = x0(2);

x = x0(:);
for k = 1:N
    tau = tau_seq(k);
    k1 = plant_pendulum(x, tau, params);
    k2 = plant_pendulum(x + Ts/2 * k1, tau, params);
    k3 = plant_pendulum(x + Ts/2 * k2, tau, params);
    k4 = plant_pendulum(x + Ts * k3, tau, params);
    x = x + Ts/6 * (k1 + 2*k2 + 2*k3 + k4);

    theta(k + 1) = x(1);
    theta_dot(k + 1) = x(2);
end

t = (0:N)' * Ts;

end
