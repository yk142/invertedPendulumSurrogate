function utils = nss_surrogate()
%NSS_SURROGATE Native MATLAB implementation of the trained NSS surrogate
%   (physical units in/out), used in place of importONNXNetwork since the
%   ONNX Converter support package is not installed in this environment
%   (仕様書§8 lists this as an accepted alternative interface).
%
%   w = utils.load(weights_path)          - load exported weights (.mat)
%   [x_next, y] = utils.step(w, x, tau)   - one surrogate step, physical units
%       x: 2x1 [theta; theta_dot] (latent state, maintained by the caller)
%       tau: 1x1 torque
%       x_next: 2x1 next latent state
%       y: 1x1 theta output (Step1 output form, 仕様書§3.1)

utils.load = @load_impl;
utils.step = @step_impl;

end

function w = load_impl(weights_path)
w = load(weights_path);
end

function [x_next, y] = step_impl(w, x_phys, tau_phys)
x_n = (x_phys - w.x_mean) ./ w.x_std;
u_n = (tau_phys - w.u_mean) ./ w.u_std;

h = [x_n; u_n];
n_layers = numel(w.f_theta_weights);
for i = 1:n_layers
    h = w.f_theta_weights{i} * h + w.f_theta_biases{i};
    if i < n_layers
        h = tanh(h);
    end
end
delta = w.increment_scale * tanh(h);
x_next_n = x_n + delta;

y_n = w.g_phi_weight * x_next_n + w.g_phi_bias;

x_next = x_next_n .* w.x_std + w.x_mean;
y = y_n .* w.y_std + w.y_mean;

end
