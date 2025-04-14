from typing import Callable, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.functional import one_hot


class KapEmbedding(nn.Module):
    """
    Special embedding that creates separate embeddings for each K_i on each
    level. Number of k's will dictate the number of linear layers.
    """

    def __init__(self, input_size: int, embedding_size: int, num_k: int) -> None:
        super().__init__()
        embeddings = nn.ModuleList()
        for _ in range(num_k):
            embeddings.append(nn.Linear(input_size, embedding_size))
        self.num_k = num_k
        self.embeddings = embeddings

    def _forward_impl(self, x: Tensor) -> Tensor:
        out = []
        for idx in range(self.num_k):
            out.append(self.embeddings[idx](x[:, idx, :]))

        return torch.stack(out, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)

        return out


class KapLCM(nn.Module):
    def __init__(
        self,
        num_feats: int,
        capacity_range: int,
        embedding_size: int = 8,
        hidden_length: int = 1,
        hidden_width: int = 32,
        dropout_percentage: float = 0,
        max_levels: int = 20,
        decision_dim: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        disable_one_hot_encoding: bool = False,
    ) -> None:
        super().__init__()
        width = (max_levels + 1) * embedding_size + num_feats - (max_levels + 1)
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.t_embedding = nn.Linear(capacity_range, embedding_size)
        self.k_embedding = nn.Linear(capacity_range, embedding_size)

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
        self.max_levels = max_levels
        self.decision_dim = decision_dim
        self.disable_one_hot_encoding = disable_one_hot_encoding

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)

    def _split_input(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        categorical_bound = self.num_feats - (self.max_levels + 1)
        feats = x[:, :categorical_bound]
        capacities = x[:, categorical_bound:]

        if self.disable_one_hot_encoding:
            capacities = torch.unflatten(capacities, 1, (-1, self.capacity_range))
        else:
            capacities = capacities.to(torch.long)
            capacities = one_hot(capacities, num_classes=self.capacity_range)

        size_ratio = capacities[:, 0, :]
        k_cap = capacities[:, 1:, :]

        return (feats, size_ratio, k_cap)

    def _forward_impl(self, x: Tensor) -> Tensor:
        feats, size_ratio, k_cap = self._split_input(x)

        size_ratio = size_ratio.to(torch.float)
        size_ratio = self.t_embedding(size_ratio)

        k_cap = k_cap.to(torch.float)
        k_cap = self.k_embedding(k_cap)
        k_cap = torch.flatten(k_cap, start_dim=1)

        inputs = torch.cat([feats, size_ratio, k_cap], dim=-1)

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

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)

        return out
