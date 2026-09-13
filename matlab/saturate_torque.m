function tau_sat = saturate_torque(tau, params)
%SATURATE_TORQUE Clip torque command to +-tau_max (02_仕様書.md §7).
%   Must be applied identically to the physical-model loop and the
%   surrogate loop when comparing closed-loop responses.

tau_sat = min(max(tau, -params.tau_max), params.tau_max);

end
