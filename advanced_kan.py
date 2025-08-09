"""Advanced KAN head implementations."""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class OptimizedKANLayer(nn.Module):
    """KAN layer with vectorised B-spline basis functions.

    The implementation avoids explicit Python loops over the spline bases and
    supports learnable knot placement as well as adaptive basis function
    selection.  It is designed to operate efficiently on batched inputs of shape
    ``(batch, tokens, in_dim)``.
    """

    def __init__(
        self, in_dim: int, out_dim: int, num_splines: int = 8, degree: int = 3
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_splines = num_splines
        self.degree = degree

        self.control_points = nn.Parameter(torch.empty(out_dim, in_dim, num_splines))
        nn.init.xavier_uniform_(self.control_points)

        raw_knots = torch.linspace(-1, 1, num_splines + degree + 1)
        self.raw_knots = nn.Parameter(raw_knots.expand(out_dim, in_dim, -1))

        self.activation_weights = nn.Parameter(torch.ones(out_dim, in_dim, 4) / 4)
        self.selector = nn.Linear(in_dim, num_splines)
        self.regularization: Tensor | None = None

    def _sorted_knots(self) -> torch.Tensor:
        return torch.sort(self.raw_knots, dim=-1)[0]

    def b_spline_basis(self, x: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
        """Vectorised Cox–de Boor recursion."""

        p = self.degree
        m = knots.shape[-1]
        n = m - p - 1
        x = x.unsqueeze(-1)  # (..., 1)
        basis = ((x >= knots[..., :-1]) & (x < knots[..., 1:])).float()
        for d in range(1, p + 1):
            left_num = x - knots[..., : m - d - 1]
            left_den = knots[..., d:m - 1] - knots[..., : m - d - 1]
            right_num = knots[..., d + 1 :] - x
            right_den = knots[..., d + 1 :] - knots[..., 1 : m - d]
            left = left_num / (left_den + 1e-8) * basis[..., : m - d - 1]
            right = right_num / (right_den + 1e-8) * basis[..., 1 : m - d]
            basis = left + right
        return basis[..., :n]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        knots = self._sorted_knots()  # (out_dim, in_dim, m)
        cp = self.control_points

        x_in = x.unsqueeze(2).unsqueeze(-1)  # (b, t, 1, in_dim, 1)
        knots = knots.unsqueeze(0).unsqueeze(0)  # (1, 1, out_dim, in_dim, m)
        basis = self.b_spline_basis(x_in, knots)  # (b, t, out_dim, in_dim, n)

        gate = torch.sigmoid(self.selector(x))  # (b, t, num_splines)
        gate = gate.unsqueeze(2).unsqueeze(3)  # (b, t, 1, 1, n)
        basis = basis * gate

        cp = cp.unsqueeze(0).unsqueeze(0)  # (1,1,out_dim,in_dim,n)
        out = (basis * cp).sum(-1).sum(-1)  # (b, t, out_dim)

        acts = torch.stack(
            [torch.relu(x), torch.tanh(x), torch.sigmoid(x), x], dim=-1
        )
        act_w = self.activation_weights.unsqueeze(0).unsqueeze(0)
        act_out = (acts.unsqueeze(2) * act_w).sum(-1).sum(-1)

        self.regularization = cp.abs().mean()
        return out + 0.1 * act_out


class ParallelKANHead(nn.Module):
    """Parallel KAN heads for multi-task learning."""

    def __init__(self, in_dim: int, task_dims: Dict[str, int], num_layers: int = 2) -> None:
        super().__init__()
        self.task_names = list(task_dims.keys())
        self.task_networks = nn.ModuleDict()
        for task, out_dim in task_dims.items():
            layers = []
            cur = in_dim
            for _ in range(num_layers - 1):
                layers.append(OptimizedKANLayer(cur, in_dim))
                layers.append(nn.LayerNorm(in_dim))
                layers.append(nn.Dropout(0.1))
                cur = in_dim
            layers.append(OptimizedKANLayer(in_dim, out_dim))
            self.task_networks[task] = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        if x.device.type == "cpu":
            with ThreadPoolExecutor(max_workers=len(self.task_names)) as executor:
                futures = {t: executor.submit(self.task_networks[t], x) for t in self.task_names}
                outputs = {t: f.result() for t, f in futures.items()}
        else:
            streams = [torch.cuda.Stream() for _ in self.task_names]
            for task, stream in zip(self.task_names, streams):
                with torch.cuda.stream(stream):
                    outputs[task] = self.task_networks[task](x)
            for stream in streams:
                stream.synchronize()
        return outputs
