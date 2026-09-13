"""Neural State Space surrogate model (02_仕様書.md §3.1-3.2).

x_{k+1} = x_k + increment_scale * tanh(f_theta([x_k, u_k]))   (M-01 scale constraint)
y_k     = g_phi(x_k)

Step1 configuration: n_x=2 (matches physical state dims), y is angle only.
No input-output direction/monotonicity constraint is applied (M-04 is a
diagnostic added in Phase4, not a constraint on this model - see
CLAUDE.md "Key technical constraints").
"""

import torch
import torch.nn as nn


class Normalizer:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["std"])

    @classmethod
    def from_data(cls, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return cls(x.mean(dim=0), x.std(dim=0))


class NSSModel(nn.Module):
    def __init__(self, n_x=2, n_u=1, n_y=1, hidden=(64, 64), increment_scale=1.0):
        super().__init__()
        layers = []
        in_dim = n_x + n_u
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers += [nn.Linear(in_dim, n_x)]
        self.f_theta = nn.Sequential(*layers)
        self.g_phi = nn.Linear(n_x, n_y)
        self.increment_scale = increment_scale
        self.n_x = n_x

    def step(self, x, u):
        """One state-transition step, in normalized coordinates."""
        raw = self.f_theta(torch.cat([x, u], dim=-1))
        delta = self.increment_scale * torch.tanh(raw)
        return x + delta

    def output(self, x):
        return self.g_phi(x)

    def rollout(self, x0, u_seq):
        """x0: (B, n_x). u_seq: (B, T, n_u). Returns y_hat: (B, T, n_y) for
        y_hat_{k+1..k+T}, and the final state x_T."""
        x = x0
        ys = []
        for k in range(u_seq.shape[1]):
            x = self.step(x, u_seq[:, k])
            ys.append(self.output(x))
        return torch.stack(ys, dim=1), x

    def jacobian_at(self, x_eq, u_eq):
        """d(x_next)/d(x) and d(x_next)/d(u) at (x_eq, u_eq), for M-02/M-04 diagnostics."""
        step_of_x = lambda xx: self.step(xx.unsqueeze(0), u_eq.unsqueeze(0)).squeeze(0)
        step_of_u = lambda uu: self.step(x_eq.unsqueeze(0), uu.unsqueeze(0)).squeeze(0)
        dxnext_dx = torch.autograd.functional.jacobian(step_of_x, x_eq)
        dxnext_du = torch.autograd.functional.jacobian(step_of_u, u_eq)
        return dxnext_dx, dxnext_du
