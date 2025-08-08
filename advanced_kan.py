"""Advanced KAN head implementations."""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import torch
import torch.nn as nn


class OptimizedKANLayer(nn.Module):
    """Optimized KAN layer using B-spline basis functions."""

    def __init__(self, in_dim: int, out_dim: int, num_splines: int = 8, degree: int = 3) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_splines = num_splines
        self.degree = degree
        self.control_points = nn.Parameter(torch.randn(out_dim, in_dim, num_splines) * 0.1)
        knots = torch.linspace(-1, 1, num_splines + degree + 1)
        self.knots = nn.Parameter(knots.expand(out_dim, in_dim, -1))
        self.activation_weights = nn.Parameter(torch.ones(out_dim, in_dim, 4) / 4)

    def _de_boor(self, x: torch.Tensor, knots: torch.Tensor, i: int, degree: int) -> torch.Tensor:
        if degree == 0:
            return ((x >= knots[..., i]) & (x < knots[..., i + 1])).float()
        c1 = (x - knots[..., i]) / (knots[..., i + degree] - knots[..., i] + 1e-8)
        c2 = (knots[..., i + degree + 1] - x) / (knots[..., i + degree + 1] - knots[..., i + 1] + 1e-8)
        return c1 * self._de_boor(x, knots, i, degree - 1) + c2 * self._de_boor(x, knots, i + 1, degree - 1)

    def b_spline_basis(self, x: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
        n = knots.shape[-1] - degree - 1
        basis = torch.zeros(*x.shape, n, device=x.device)
        for i in range(n):
            basis[..., i] = self._de_boor(x, knots, i, degree)
        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        x = x.unsqueeze(1).expand(-1, self.out_dim, -1, -1)
        basis = self.b_spline_basis(x, self.knots.unsqueeze(0), self.degree)
        out = torch.einsum("bond,ods->bons", basis, self.control_points).sum(dim=-1)
        acts = torch.stack(
            [torch.relu(x), torch.tanh(x), torch.sigmoid(x), x], dim=-1
        )
        act_out = torch.einsum("bond,od4->bon", acts, self.activation_weights).sum(dim=-1)
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
