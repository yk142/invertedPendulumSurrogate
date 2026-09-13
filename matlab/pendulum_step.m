function x_next = pendulum_step(x, tau, params)
%PENDULUM_STEP Single fixed-step RK4 update of the physical plant, at Ts_ctrl.
%   Single-step counterpart to simulate_pendulum.m, for use inside a
%   step-by-step closed-loop control loop.

Ts = params.Ts_ctrl;
k1 = plant_pendulum(x, tau, params);
k2 = plant_pendulum(x + Ts/2 * k1, tau, params);
k3 = plant_pendulum(x + Ts/2 * k2, tau, params);
k4 = plant_pendulum(x + Ts * k3, tau, params);
x_next = x + Ts/6 * (k1 + 2*k2 + 2*k3 + k4);

end
