from typing import Callable, Optional

from reinmax import reinmax
from torch import Tensor, nn
import torch


class RobustClassicTunerSampler(nn.Module):
    def __init__(
        self,
        num_feats: int,
        capacity_range: int,
        hidden_length: int = 1,
        hidden_width: int = 32,
        dropout_percentage: float = 0,
        categorical_mode: str = "gumbel",
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.in_norm = norm_layer(num_feats)
        self.in_layer = nn.Linear(num_feats, hidden_width)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_percentage)
        hidden = []
        for _ in range(hidden_length):
            hidden.append(nn.Linear(hidden_width, hidden_width))
        self.hidden = nn.Sequential(*hidden)

        self.t_decision = nn.Linear(hidden_width, capacity_range)
        self.bits_decision = nn.Linear(hidden_width, 1)
        self.policy_decision = nn.Linear(hidden_width, 2)
        self.lagrangian_mu = nn.Linear(num_feats, 2)
        self.lagrangian_sigma = nn.Linear(num_feats, 2)
        # self.lagrangians = nn.Linear(num_feats, 2)

        self.capacity_range = capacity_range
        self.num_feats = num_feats
        self.categorical_mode = categorical_mode

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)

    def _forward_impl(self, x: Tensor, temp=1e-3, hard=False) -> Tensor:
        normed_x = self.in_norm(x)
        out = self.in_layer(normed_x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.hidden(out)
        out = self.relu(out)

        bits = self.bits_decision(out)
        t = self.t_decision(out)
        policy = self.policy_decision(out)
        if self.categorical_mode == "reinmax":
            t, _ = reinmax(t, tau=temp)
            policy, _ = reinmax(policy, tau=temp)
        else:  # categorical_mode == 'gumbel'
            t = nn.functional.gumbel_softmax(t, tau=temp, hard=hard)
            policy = nn.functional.gumbel_softmax(policy, tau=temp, hard=hard)

        epsilon = torch.normal(0, 1, size=(x.shape[0], 2)).to(x.device)
        mu = self.lagrangian_mu(normed_x)
        sigma = self.lagrangian_sigma(normed_x)
        lagrangians = mu + (epsilon * sigma)

        out = torch.concat([lagrangians, bits, t, policy], dim=-1)

        return out

    def forward(self, x: Tensor, temp=1e-3, hard=False) -> Tensor:
        out = self._forward_impl(x, temp=temp, hard=hard)

        return out
