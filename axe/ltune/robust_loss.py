from typing import Any
import os

import torch
from torch import Tensor
import toml

from axe.lcm.model.builder import LearnedCostModelBuilder
from axe.lsm.types import Policy, LSMBounds


class LearnedRobustLoss(torch.nn.Module):
    def __init__(self, config: dict[str, Any], model_path: str):
        super().__init__()
        bounds: LSMBounds = LSMBounds(**config["lsm"]["bounds"])

        model_data_dir = os.path.join(config["io"]["data_dir"], model_path)
        lcm_cfg = toml.load(os.path.join(model_data_dir, "axe.toml"))
        lcm_model: Policy = getattr(Policy, lcm_cfg["lsm"]["design"])
        lcm_bounds: LSMBounds = LSMBounds(**lcm_cfg["lsm"]["bounds"])
        self.model: torch.nn.Module = LearnedCostModelBuilder(
            size_ratio_range=lcm_bounds.size_ratio_range,
            max_levels=lcm_bounds.max_considered_levels,
            **lcm_cfg["lcm"]["model"],
        ).build_model(lcm_model)

        data = torch.load(os.path.join(model_data_dir, "best.model"))
        status = self.model.load_state_dict(data)

        assert bounds.size_ratio_range == lcm_bounds.size_ratio_range
        assert bounds.max_considered_levels == lcm_bounds.max_considered_levels
        assert len(status.missing_keys) == 0
        assert len(status.unexpected_keys) == 0
        self.capacity_range = bounds.size_ratio_range[1] - bounds.size_ratio_range[0]
        self.num_levels = bounds.max_considered_levels
        self.bounds: LSMBounds = bounds
        self.penalty_factor = config["ltune"]["penalty_factor"]
        # TODO: We will need a way for this to be user definable or something
        # that isn't just straight hardcoded into this loss function
        # This is the index for H--the total available memory that
        # could be split between buffer and bloom filters
        self.mem_budget_idx = 7
        self.model.eval()

    def kl_div_conj(self, input):
        ret = torch.exp(input) - 1

        return ret

    def calc_mem_penalty(self, label, bpe):
        mem_budget = label[:, self.mem_budget_idx].view(-1, 1)
        penalty = torch.zeros(bpe.size()).to(bpe.device)
        idx = bpe >= mem_budget
        penalty[idx] = self.penalty_factor * (bpe[idx] - mem_budget[idx])
        idx = bpe < 0
        penalty[idx] = self.penalty_factor * (0 - bpe[idx])

        bpe[bpe > mem_budget] = mem_budget[bpe > mem_budget]
        bpe[bpe < 0] = 0

        return bpe, penalty

    def split_tuner_out(self, tuner_out):
        # First 3 items are going to be rho and lagragians, then BPE, then
        # categorical features (size ratio and capacities)
        eta = tuner_out[:, 0]
        lamb = tuner_out[:, 1]
        bpe = tuner_out[:, 2]
        bpe = bpe.view(-1, 1)
        categorical_feats = tuner_out[:, 1:]

        return eta, lamb, bpe, categorical_feats

    def l1_penalty_klsm(self, k_decision: Tensor):
        batch, _ = k_decision.shape
        base = torch.zeros((batch, self.num_levels))
        base = torch.nn.functional.one_hot(
            base.to(torch.long), num_classes=self.capacity_range
        )
        base = base.flatten(start_dim=1)

        if k_decision.get_device() >= 0:  # Tensor on GPU
            base = base.to(k_decision.device)

        penalty = k_decision - base
        penalty = penalty.square()
        penalty = penalty.sum(dim=-1)
        penalty = penalty.mean()

        return penalty

    def _forward_impl(self, pred, label):
        # For learned cost model loss, the prediction is the DB configuration
        # and label is the workload and system params, rho at the end
        eta, lamb, bpe, categorical_feats = self.split_tuner_out(pred)
        rho = label[:, 9]
        label = label[:9]
        bpe, penalty = self.calc_mem_penalty(label, bpe)

        inputs = torch.concat([label, bpe, categorical_feats], dim=-1)
        out = self.model(inputs)
        out = (out - eta) / lamb
        out = self.kl_div_conj(out)
        out = out.sum(dim=-1)
        out = eta + (lamb * rho) + (lamb * out)
        out = out + penalty
        out = out.mean()

        return out

    def forward(self, pred, label):
        assert self.model.training is False
        out = self._forward_impl(pred, label)

        return out
