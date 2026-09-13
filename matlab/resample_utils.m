function utils = resample_utils()
%RESAMPLE_UTILS Multi-rate ZOH interface between the 8kHz control loop and
%   a surrogate running at a slower rate (e.g. 200Hz, 40:1). See
%   02_仕様書.md §6.4.
%
%   utils.decimate_input(tau_ctrl, ratio)   -> torque sampled at surrogate
%       update instants (instantaneous value, NOT averaged - this is the
%       default ZOH convention).
%   utils.hold_output(y_surr, ratio, n_ctrl_steps) -> surrogate output
%       held constant (zero-order hold) at control rate until the next
%       surrogate update.

utils.decimate_input = @decimate_input_impl;
utils.hold_output = @hold_output_impl;

end

function tau_surr = decimate_input_impl(tau_ctrl, ratio)
tau_surr = tau_ctrl(1:ratio:end);
end

function y_ctrl = hold_output_impl(y_surr, ratio, n_ctrl_steps)
y_ctrl = repelem(y_surr, ratio);
if numel(y_ctrl) < n_ctrl_steps
    y_ctrl(end+1:n_ctrl_steps) = y_ctrl(end);
else
    y_ctrl = y_ctrl(1:n_ctrl_steps);
end
end
