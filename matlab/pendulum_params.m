function params = pendulum_params()
%PENDULUM_PARAMS Standard parameter set for the single-pendulum plant.
%   Values must be kept in sync with config.yaml (plant: section) at the
%   repository root, since MATLAB has no built-in YAML reader in this
%   environment. See 02_仕様書.md §2.3.

params.J = 0.02;                 % kg*m^2, inertia
params.b = 0.05;                 % N*m*s/rad, viscous friction
params.mgl = 0.15;               % kg*m, equivalent m*g*l term
params.tau_max = 3.0;            % N*m, actuator saturation
params.theta_min = -pi/2;        % rad
params.theta_max = pi/2;         % rad
params.Ts_ctrl = 1 / 8000;       % s, base control/plant period (8kHz)
params.Ts_surr = 1 / 200;        % s, optional surrogate downsample period (200Hz)

end
