import torch
from torch import nn
from axe.lsm.types import Policy

from axe.ltuner.data.schema import LTunerDataSchema
from axe.ltuner.model import ClassicTuner, QLSMTuner, KapLSMTuner, YZLSMTuner
from axe.ltuner.model.kap_robust_tuner import KapLSMRobustTuner
from axe.ltuner.model.rc_latent import RCLatent
from axe.ltuner.model.robust_classic_tuner import RobustClassicTuner
from axe.ltuner.model.robust_classic_sampling import RobustClassicTunerSampler


class LTuneModelBuilder:
    def __init__(
        self,
        schema: LTunerDataSchema,
        hidden_length: int = 1,
        hidden_width: int = 64,
        norm_layer: str = "Batch",
        dropout: float = 0.0,
        categorical_mode: str = "gumbel",
    ) -> None:
        self.hidden_length = hidden_length
        self.hidden_width = hidden_width
        self.dropout = dropout
        self.categorical_mode = categorical_mode
        self.max_levels = schema.bounds.max_considered_levels
        size_ratio_min, size_ratio_max = schema.bounds.size_ratio_range
        self.capacity_range = size_ratio_max - size_ratio_min
        self.schema: LTunerDataSchema = schema

        self.norm_layer = nn.BatchNorm1d
        if norm_layer == "Layer":
            self.norm_layer = nn.LayerNorm
        self._models = {
            Policy.Classic: ClassicTuner,
            Policy.QHybrid: QLSMTuner,
            Policy.Fluid: YZLSMTuner,
            Policy.Kapacity: KapLSMTuner,
        }

    def get_choices(self):
        return self._models.keys()

    def build(self, robust: bool = False, override: str = "None") -> torch.nn.Module:
        feat_list = self.schema.feat_cols()
        kwargs = {
            "num_feats": len(feat_list),
            "capacity_range": self.capacity_range,
            "hidden_length": self.hidden_length,
            "hidden_width": self.hidden_width,
            "dropout_percentage": self.dropout,
            "norm_layer": self.norm_layer,
        }
        model_class = self._models.get(self.schema.policy, None)
        if override == "RobustClassicTunerSampler":
            return RobustClassicTunerSampler(**kwargs)
        elif override == "RobustClassicTuner":
            return RobustClassicTuner(**kwargs)
        elif override == "RCLatent":
            return RCLatent(**kwargs)

        if model_class is None:
            raise NotImplementedError("Tuner for LSM Design not implemented.")
        if model_class is KapLSMTuner:
            kwargs["num_kap"] = self.max_levels
            kwargs["categorical_mode"] = self.categorical_mode
            if robust:
                # return KapLSMRobustTunerLatent(**kwargs)
                return KapLSMRobustTuner(**kwargs)

        model = model_class(**kwargs)

        return model
