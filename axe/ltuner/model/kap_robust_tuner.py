from typing import Callable, Optional

from torch import Tensor, nn
import torch
from reinmax import reinmax


class ResidualBlock(nn.Module):
    def __init__(
        self, num_neurons: int, num_layers: int, activation: nn.Module
    ) -> None:
        super().__init__()
        hidden = []
        for _ in range(num_layers):
            hidden.append(nn.Linear(num_neurons, num_neurons))
            hidden.append(activation)
        self.hidden = nn.Sequential(*hidden)

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.hidden(inputs)
        return out + inputs


class ExpActivation(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return torch.log(1 + torch.exp(inputs))


class KapDecision(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        num_kap: int,
        categorical_mode: str = "gumbel",
    ) -> None:
        super().__init__()
        self.decision_layers = nn.ModuleList(
            [nn.Linear(input_size, num_classes) for _ in range(num_kap)]
        )
        self.categorical_mode = categorical_mode

    def _forward_impl(self, x: Tensor, temp=1e-3, hard=False) -> Tensor:
        out = []
        for layer in self.decision_layers:
            k = layer(x)
            if self.categorical_mode == "reinmax":
                k, _ = reinmax(k, tau=temp)
            else:  # categorical_mode == 'gumbel'
                k = nn.functional.gumbel_softmax(k, tau=temp, hard=hard)
            out.append(k)
        out = torch.stack(out, dim=1)

        return out

    def forward(self, x: Tensor, temp=1e-3, hard=False) -> Tensor:
        out = self._forward_impl(x, temp=temp, hard=hard)

        return out


class KapLSMRobustTuner(nn.Module):
    def __init__(
        self,
        num_feats: int,
        capacity_range: int,
        hidden_length: int = 1,
        hidden_width: int = 32,
        dropout_percentage: float = 0,
        num_kap: int = 10,
        categorical_mode: str = "gumbel",
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.in_norm = norm_layer(num_feats)
        self.in_layer = nn.Linear(num_feats, hidden_width)
        self.hidden = ResidualBlock(hidden_width, hidden_length, nn.ReLU())
        self.dropout = nn.Dropout(p=dropout_percentage)
        self.k_path = ResidualBlock(hidden_width, 1, nn.ReLU())
        self.k_decision = KapDecision(hidden_width, capacity_range, num_kap)
        self.t_decision = nn.Sequential(
            ResidualBlock(hidden_width, 1, nn.ReLU()),
            nn.Linear(hidden_width, capacity_range)
        )
        self.bits_decision = nn.Sequential(
            ResidualBlock(hidden_width, 1, nn.ReLU()),
            nn.Linear(hidden_width, 1),
            ExpActivation()
        )
        self.lagrangian_mu = nn.Linear(num_feats, 2)
        self.lagrangian_sigma = nn.Linear(num_feats, 2)

        self.capacity_range = capacity_range
        self.num_feats = num_feats
        self.num_kap = num_kap
        self.categorical_mode = categorical_mode

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)

    def calc_max_level(
        self,
        x: Tensor,  # input tensor
        bpe: Tensor,
        size_ratio: Tensor,
    ) -> Tensor:
        # KLSM: ["z0", "z1", "q", "w", "B", "s", "E", "H", "N"]
        # IDX:  [  0 ,   1 ,  2 ,  3 ,  4 ,  5 ,  6 ,  7 ,  8]
        size_ratio = torch.squeeze(torch.argmax(size_ratio, dim=-1)) + 2
        bits = torch.squeeze(bpe)
        max_bits = x[:, 7]  # H
        num_elem = x[:, 8]  # N
        entry_size = x[:, 6]  # E
        min_bits = torch.zeros(bits.shape).to(bits.device)
        bits = torch.clamp(bits, min=min_bits, max=(max_bits - 0.1))
        mbuff = (max_bits - bits) * num_elem
        level = torch.log(((num_elem * entry_size) / mbuff) + 1)
        level = level / torch.log(size_ratio)
        level = torch.ceil(level)
        level = torch.clamp(level, min=1)

        return level

    def get_mask_and_default(self, max_levels: Tensor):
        mask = nn.functional.one_hot(max_levels, num_classes=self.num_kap)
        cum_sum = torch.cumsum(mask, dim=1)
        mask = 1 - cum_sum + mask  # Sets all values AFTER max_level to 1
        default_values = torch.zeros(self.capacity_range)
        default_values[0] = 1
        default_values = default_values.to(torch.long)

        return mask, default_values

    def forward(self, x: Tensor, temp=1e-3, hard=False) -> Tensor:
        normed_x = self.in_norm(x)
        out = self.in_layer(normed_x)
        out = self.dropout(out)
        out = self.hidden(out)

        bits = self.bits_decision(out)
        t = self.t_decision(out)
        if self.categorical_mode == "reinmax":
            t, _ = reinmax(t, tau=temp)
        else:  # categorical_mode == 'gumbel'
            t = nn.functional.gumbel_softmax(t, tau=temp, hard=hard)
        k = self.k_path(out)
        k = self.k_decision(k, temp=temp, hard=hard)

        max_levels = self.calc_max_level(x, bits, t) - 1
        max_levels = max_levels.to(torch.long)
        mask, default = self.get_mask_and_default(max_levels)
        k = mask.unsqueeze(-1) * k
        k[mask == 0] += default.to(k.device)

        k = torch.flatten(k, start_dim=1)

        epsilon = torch.normal(0, 1, size=(x.shape[0], 2)).to(x.device)
        mu = self.lagrangian_mu(normed_x)
        sigma = self.lagrangian_sigma(normed_x)
        lagrangians = mu + (epsilon * sigma)

        out = torch.concat([lagrangians, bits, t, k], dim=-1)

        return out
