from typing import Callable, Optional, Tuple

from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F


class ClassicLCM(nn.Module):
    def __init__(
        self,
        num_feats: int,
        capacity_range: int,
        embedding_size: int = 8,
        hidden_length: int = 1,
        hidden_width: int = 32,
        dropout_percentage: float = 0,
        policy_embedding_size: int = 1,
        decision_dim: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        disable_one_hot_encoding: bool = False,
    ) -> None:
        super().__init__()
        width = (num_feats - 2) + embedding_size + policy_embedding_size
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.t_embedding = nn.Linear(capacity_range, embedding_size)
        self.policy_embedding = nn.Linear(2, policy_embedding_size)
        self.in_norm = norm_layer(width)
        self.in_layer = nn.Linear(width, hidden_width)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_percentage)
        hidden = []
        for _ in range(hidden_length):
            hidden.append(nn.Linear(hidden_width, hidden_width))
            hidden.append(nn.ReLU(inplace=True))
        self.hidden = nn.Sequential(*hidden)
        self.out_layer = nn.Linear(hidden_width, decision_dim)
        split_head_width = int(decision_dim / 4)
        self.z0 = nn.Linear(split_head_width, 1)
        self.z1 = nn.Linear(split_head_width, 1)
        self.q = nn.Linear(split_head_width, 1)
        self.w = nn.Linear(split_head_width, 1)

        self.capacity_range = capacity_range
        self.num_feats = num_feats
        self.decision_dim = decision_dim
        self.disable_one_hot_encoding = disable_one_hot_encoding

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)

    def _split_input(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        t_boundary = self.num_feats - 2
        policy_boundary = t_boundary + self.capacity_range
        feats = x[:, :t_boundary]

        if self.disable_one_hot_encoding:
            policy = x[:, policy_boundary : policy_boundary + 2]
            size_ratio = x[:, t_boundary:policy_boundary]
        else:
            size_ratio = x[:, -2]
            size_ratio = size_ratio.to(torch.long)
            size_ratio = F.one_hot(size_ratio, num_classes=self.capacity_range)
            policy = x[:, -1]
            policy = policy.to(torch.long)
            policy = F.one_hot(policy, num_classes=2)

        return (feats, size_ratio, policy)

    def _forward_impl(self, x: Tensor) -> Tensor:
        feats, size_ratio, policy = self._split_input(x)
        size_ratio = size_ratio.to(torch.float)
        size_ratio = self.t_embedding(size_ratio)

        policy = policy.to(torch.float)
        policy = self.policy_embedding(policy)

        inputs = torch.cat([feats, size_ratio, policy], dim=-1)

        out = self.in_norm(inputs)
        out = self.in_layer(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.hidden(out)
        out = self.out_layer(out)
        head_dim = int(self.decision_dim / 4)
        z0 = self.z0(out[:, 0:head_dim])
        z1 = self.z1(out[:, head_dim : 2 * head_dim])
        q = self.q(out[:, 2 * head_dim : 3 * head_dim])
        w = self.w(out[:, 3 * head_dim : 4 * head_dim])
        out = torch.cat([z0, z1, q, w], dim=-1)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._forward_impl(x)

        return out
