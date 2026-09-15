function utils = nss_surrogate_ablationC()
%NSS_SURROGATE_ABLATIONC Native MATLAB implementation of Ablation C
%   (tanh->ReLU hidden activations, M-01 tanh bounding removed - issue
%   #15/#33). Same shared-nonlinearity architecture as nss_surrogate.m
%   otherwise (u concatenated into f_theta, unlike the control-affine
%   variant).
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

h = [x_n; u_n];
n_layers = numel(w.f_theta_weights);
for i = 1:n_layers
    h = w.f_theta_weights{i} * h + w.f_theta_biases{i};
    if i < n_layers
        h = max(h, 0); % ReLU
    end
end
if w.bounded_increment ~= 0
    delta = w.increment_scale * tanh(h);
else
    delta = h; % unbounded, M-01 removed
end
x_next_n = x_n + delta;

y_n = w.g_phi_weight * x_next_n + w.g_phi_bias;

x_next = x_next_n .* w.x_std + w.x_mean;
y = y_n .* w.y_std + w.y_mean;

end
