function utils = nss_surrogate_control_affine()
%NSS_SURROGATE_CONTROL_AFFINE Native MATLAB implementation of the
%   control-affine NSS (issue #27/#29): x_next = x + scale*tanh(f_drift(x)) + B*u.
%   Counterpart to nss_surrogate.m for the standard (blended) architecture.
%
%   w = utils.load(weights_path)          - load exported weights (.mat)
%   [x_next, y] = utils.step(w, x, tau)   - one surrogate step, physical units

utils.load = @load_impl;
utils.step = @step_impl;

end

function w = load_impl(weights_path)
w = load(weights_path);
end

function [x_next, y] = step_impl(w, x_phys, tau_phys)
x_n = (x_phys - w.x_mean) ./ w.x_std;
u_n = (tau_phys - w.u_mean) ./ w.u_std;

h = x_n; % f_drift depends on state only, NOT concatenated with u
n_layers = numel(w.f_drift_weights);
for i = 1:n_layers
    h = w.f_drift_weights{i} * h + w.f_drift_biases{i};
    if i < n_layers
        h = tanh(h);
    end
end
drift = w.increment_scale * tanh(h);
input_effect = w.B * u_n; % state-independent constant-matrix input term
x_next_n = x_n + drift + input_effect;

y_n = w.g_phi_weight * x_next_n + w.g_phi_bias;

x_next = x_next_n .* w.x_std + w.x_mean;
y = y_n .* w.y_std + w.y_mean;

end
