function dxdt = plant_pendulum(x, tau, params)
%PLANT_PENDULUM Continuous-time equation of motion for the single pendulum.
%   dxdt = plant_pendulum(x, tau, params), x = [theta; theta_dot]
%   J*theta_ddot + b*theta_dot + mgl*sin(theta) = tau   (02_仕様書.md §2.1-2.2)

theta = x(1);
theta_dot = x(2);

theta_ddot = (tau - params.b * theta_dot - params.mgl * sin(theta)) / params.J;
dxdt = [theta_dot; theta_ddot];

end
