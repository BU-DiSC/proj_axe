import torch
from typing import Tuple
from torch import nn
from axe.lsm.types import Policy

from axe.ltune.model import ClassicTuner, QLSMTuner, KapLSMTuner, YZLSMTuner
from axe.ltune.model.kap_robust_tuner import KapLSMRobustTuner
from axe.ltune.data.input_features import kINPUT_FEATS, kINPUT_FEATS_ROBUST


class LTuneModelBuilder:
    def __init__(
        self,
        hidden_length: int = 1,
        hidden_width: int = 64,
        norm_layer: str = "Batch",
        dropout: float = 0.0,
        categorical_mode: str = "gumbel",
        size_ratio_range: Tuple[int, int] = (2, 31),
        max_levels: int = 16,
    ) -> None:
        self.hidden_length = hidden_length
        self.hidden_width = hidden_width
        self.dropout = dropout
        self.categorical_mode = categorical_mode
        self.max_levels = max_levels
        self.size_ratio_min, self.size_ratio_max = size_ratio_range
        self.capacity_range = self.size_ratio_max - self.size_ratio_min

        self.norm_layer = nn.BatchNorm1d
        if norm_layer == "Layer":
            self.norm_layer = nn.LayerNorm

        self._models = {
            Policy.Classic: ClassicTuner,
            Policy.QFixed: QLSMTuner,
            Policy.KHybrid: KapLSMTuner,
            Policy.YZHybrid: YZLSMTuner,
        }

    def get_choices(self):
        return self._models.keys()

    def build_robust_model(self) -> torch.nn.Module:
        feat_list = kINPUT_FEATS_ROBUST
        kwargs = {
            "num_feats": len(feat_list),
            "capacity_range": self.capacity_range,
            "hidden_length": self.hidden_length,
            "hidden_width": self.hidden_width,
            "dropout_percentage": self.dropout,
            "norm_layer": self.norm_layer,
            "num_kap": self.max_levels,
            "categorical_mode": self.categorical_mode
        }
        model = KapLSMRobustTuner(**kwargs)

        return model

    def build_model(self, policy: Policy) -> torch.nn.Module:
        feat_list = kINPUT_FEATS

        kwargs = {
            "num_feats": len(feat_list),
            "capacity_range": self.capacity_range,
            "hidden_length": self.hidden_length,
            "hidden_width": self.hidden_width,
            "dropout_percentage": self.dropout,
            "norm_layer": self.norm_layer,
        }

        model_class = self._models.get(policy, None)
        if model_class is None:
            raise NotImplementedError("Tuner for LSM Design not implemented.")

        if model_class is KapLSMTuner:
            kwargs["num_kap"] = self.max_levels
            kwargs["categorical_mode"] = self.categorical_mode

        model = model_class(**kwargs)

        return model
