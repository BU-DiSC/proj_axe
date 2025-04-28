import os
from typing import Any

import toml
import torch
from torch import Tensor

from axe.lcm.data.schema import LCMDataSchema
from axe.lcm.model.builder import LearnedCostModelBuilder
from axe.lsm.types import LSMBounds, Policy
from axe.ltuner.data.schema import LTunerDataSchema


class LearnedRobustLoss(torch.nn.Module):
    def __init__(self, config: dict[str, Any], model_path: str):
        super().__init__()
        self.penalty_factor = config["ltuner"]["penalty_factor"]
        self.bounds: LSMBounds = LSMBounds(**config["lsm"]["bounds"])
        self.policy: Policy = getattr(Policy, config["lsm"]["policy"])
        min_size_ratio, max_size_ratio = self.bounds.size_ratio_range
        self.kap_categories = max_size_ratio - min_size_ratio
        self.tuner_schema = LTunerDataSchema(
            policy=self.policy,
            bounds=self.bounds,
            robust=True,
        )
        self.rho_idx = self.tuner_schema.feat_cols().index("rho")

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

    def kl_div_conj(self, x: Tensor):
        # ret = torch.exp(x.clamp(max=50)) - 1
        ret = torch.exp(x.clamp(max=20)) - 1

        return ret

    def split_label(self, label):
        rho = label[:, self.rho_idx]
        workload = label[:, 0:4]  # TODO Andy stop hardcoding stupid stuff
        env_label = label[:, : self.rho_idx]

        return rho, workload, env_label

    def split_pred(self, pred: Tensor):
        # First 3 items are going to be lagragians then BPE.
        # After we have categorical features (size ratio and capacities)
        # [lagrag0, lagrag1, bpe, size_ratio, k1, ..., kL]
        eta = pred[:, 0]
        lamb = pred[:, 1]
        bpe = pred[:, 2]
        categorical_feats = pred[:, 3:]

        return eta, lamb, bpe, categorical_feats

    def calc_mem_penalty(self, label: Tensor, bpe: Tensor):
        mem_budget = label[:, self.mem_budget_idx]
        penalty = torch.zeros(bpe.size()).to(bpe.device)
        over_idx = bpe >= mem_budget
        penalty[over_idx] = self.penalty_factor * (bpe[over_idx] - mem_budget[over_idx])
        under_idx = bpe < 0
        penalty[under_idx] = self.penalty_factor * (0 - bpe[under_idx])
        bpe[bpe >= mem_budget] = mem_budget[bpe >= mem_budget]
        bpe[bpe < 0] = 0

        return bpe, penalty

    def forward(self, pred: Tensor, label: Tensor):
        # For learned cost model loss, the prediction is the DB configuration
        # and label is the workload and system params, rho at the end
        assert self.lcm.training is False
        eta, lamb, bpe, categorical_feats = self.split_pred(pred)
        rho, workload, env_label = self.split_label(label)
        bpe, penalty = self.calc_mem_penalty(env_label, bpe)

        lcm_input = torch.concat(
            [env_label, bpe.unsqueeze(1), categorical_feats], dim=-1
        )
        out = self.lcm(lcm_input)
        out = out / (workload.clamp(min=1e-4))
        out = (out - eta.unsqueeze(1)) / lamb.unsqueeze(1).clamp(min=1)
        out = self.kl_div_conj(out)
        out = workload * out
        out = out.sum(dim=-1)
        out = eta + (lamb * rho) + (lamb * out)
        out = out.square()
        out = out + penalty
        out = out.mean()

        return out
