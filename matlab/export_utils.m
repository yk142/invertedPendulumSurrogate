function utils = export_utils()
%EXPORT_UTILS Factory returning MATLAB<->Python .mat interchange functions.
%   utils = export_utils() returns a struct of function handles:
%     utils.export_dataset(filename, t, theta, theta_dot, tau, meta)
%     utils.import_dataset(filename) -> struct with t, theta, theta_dot, tau, meta
%
%   Counterpart to python/io_utils.py.

utils.export_dataset = @export_dataset_impl;
utils.import_dataset = @import_dataset_impl;

end

function export_dataset_impl(filename, t, theta, theta_dot, tau, meta)
if nargin < 6
    meta = struct();
end
data.t = t(:);
data.theta = theta(:);
data.theta_dot = theta_dot(:);
data.tau = tau(:);
data.meta = meta;
save(filename, '-struct', 'data');
end

function s = import_dataset_impl(filename)
s = load(filename);
end
