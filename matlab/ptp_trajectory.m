function [theta_ref, theta_dot_ref, theta_ddot_ref, t] = ptp_trajectory(theta_start, theta_end, vmax, amax, Ts, T_total)
%PTP_TRAJECTORY Trapezoidal-velocity point-to-point trajectory.
%   [theta_ref, theta_dot_ref, theta_ddot_ref, t] = ...
%       ptp_trajectory(theta_start, theta_end, vmax, amax, Ts, T_total)
%
%   Generates a trapezoidal (or triangular, if the distance is too short
%   to reach vmax) velocity profile from theta_start to theta_end, then
%   holds at rest at theta_end for the remainder of T_total.
%   See 02_仕様書.md §4.1 (Step1: PTP軌道フィードフォワードトルク).

D = theta_end - theta_start;
dir = sign(D);
dist = abs(D);

t_acc = vmax / amax;
d_acc = 0.5 * amax * t_acc^2;

if 2 * d_acc >= dist
    % Triangular profile: never reaches vmax
    t_acc = sqrt(dist / amax);
    vmax_actual = amax * t_acc;
    t_flat = 0;
else
    vmax_actual = vmax;
    d_flat = dist - 2 * d_acc;
    t_flat = d_flat / vmax;
end
t_motion = 2 * t_acc + t_flat;

if t_motion > T_total
    error('ptp_trajectory:durationTooShort', ...
        'T_total (%.3f s) is shorter than the motion time (%.3f s).', T_total, t_motion);
end

t = (0:Ts:T_total)';
n = numel(t);
theta_ref = zeros(n, 1);
theta_dot_ref = zeros(n, 1);
theta_ddot_ref = zeros(n, 1);

for i = 1:n
    tt = t(i);
    if tt < t_acc
        % Acceleration phase
        a = amax;
        v = amax * tt;
        s = 0.5 * amax * tt^2;
    elseif tt < t_acc + t_flat
        % Constant-velocity phase
        a = 0;
        v = vmax_actual;
        s = d_acc + vmax_actual * (tt - t_acc);
    elseif tt < t_motion
        % Deceleration phase
        td = tt - (t_acc + t_flat);
        a = -amax;
        v = vmax_actual - amax * td;
        s = d_acc + vmax_actual * t_flat + vmax_actual * td - 0.5 * amax * td^2;
    else
        % Settled at target
        a = 0;
        v = 0;
        s = dist;
    end
    theta_ddot_ref(i) = dir * a;
    theta_dot_ref(i) = dir * v;
    theta_ref(i) = theta_start + dir * s;
end

end
