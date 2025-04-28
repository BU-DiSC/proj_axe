import os
from typing import Any

import toml
import torch
from torch import Tensor

from axe.lcm.data.schema import LCMDataSchema
from axe.lcm.model.builder import LearnedCostModelBuilder
from axe.lsm.types import LSMBounds, Policy


class LearnedCostModelLoss(torch.nn.Module):
    def __init__(self, config: dict[str, Any], model_path: str):
        super().__init__()
        self.penalty_factor = config["ltuner"]["penalty_factor"]
        self.bounds: LSMBounds = LSMBounds(**config["lsm"]["bounds"])
        self.policy: Policy = getattr(Policy, config["lsm"]["policy"])
        min_size_ratio, max_size_ratio = self.bounds.size_ratio_range
        self.kap_categories = max_size_ratio - min_size_ratio

        lcm_cfg = toml.load(os.path.join(model_path, "axe.toml"))
        self.lcm_bounds: LSMBounds = LSMBounds(**lcm_cfg["lsm"]["bounds"])
        self.lcm_policy: Policy = getattr(Policy, lcm_cfg["lsm"]["policy"])
        self.lcm_schema = LCMDataSchema(self.lcm_policy, self.lcm_bounds)
        self.mem_budget_idx = self.lcm_schema.feat_cols().index("mem_budget")

        # Before building we assert LCM -> LTuner designs are matching
        assert self.bounds == self.lcm_bounds
        assert self.policy == self.lcm_policy

        self.lcm = LearnedCostModelBuilder(
            schema=self.lcm_schema,
            **lcm_cfg["lcm"]["model"],
        ).build(disable_one_hot_encoding=True)
        data = torch.load(
            os.path.join(model_path, "best_model.model"),
            weights_only=True,
        )
        status = self.lcm.load_state_dict(data["model_state_dict"])
        # Check model loaded in correctly then set to evaluation mode
        assert len(status.missing_keys) == 0
        assert len(status.unexpected_keys) == 0
        self.lcm.eval()

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
        bpe = tuner_out[:, 0]
        bpe = bpe.view(-1, 1)
        categorical_feats = tuner_out[:, 1:]

        return bpe, categorical_feats

    def forward(self, pred: Tensor, label: Tensor):
        assert self.lcm.training is False
        # For learned cost model loss, the prediction is the DB configuration
        # and label is the workload and system params
        bpe, categorical_feats = self.split_tuner_out(pred)
        bpe, penalty = self.calc_mem_penalty(label, bpe)

        inputs = torch.concat([label, bpe, categorical_feats], dim=-1)
        out = self.lcm(inputs)
        out = out.square()
        out = out.sum(dim=-1)
        out = out + penalty
        out = out.mean()

        return out
