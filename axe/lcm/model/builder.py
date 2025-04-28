from torch import nn
import torch

from axe.lcm.data.schema import LCMDataSchema
from axe.lcm.model import KapLCM, QHybridLCM, ClassicLCM, FluidLCM
from axe.lsm.types import Policy


class LearnedCostModelBuilder:
    def __init__(
        self,
        schema: LCMDataSchema,
        hidden_length: int = 1,
        hidden_width: int = 64,
        embedding_size: int = 8,
        norm_layer: str = "Batch",
        policy_embedding_size: int = 2,  # only used for classic model
        decision_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        self.embedding_size = embedding_size
        self.policy_embedding_size = policy_embedding_size
        self.hidden_length = hidden_length
        self.hidden_width = hidden_width
        self.decision_dim = decision_dim
        self.dropout = dropout
        self.max_levels = schema.bounds.max_considered_levels
        self.capacity_range = (
            schema.bounds.size_ratio_range[1] - schema.bounds.size_ratio_range[0]
        )
        self.schema = schema

        self.norm_layer = nn.BatchNorm1d
        if norm_layer == "Layer":
            self.norm_layer = nn.LayerNorm

        self._models = {
            Policy.Classic: ClassicLCM,
            Policy.QHybrid: QHybridLCM,
            Policy.Fluid: FluidLCM,
            Policy.Kapacity: KapLCM,
        }

    def get_choices(self):
        return self._models.keys()

    def build(self, **kwargs) -> torch.nn.Module:
        feat_columns = self.schema.feat_cols()
        num_feats = len(feat_columns)
        args = {
            "num_feats": num_feats,
            "capacity_range": self.capacity_range,
            "embedding_size": self.embedding_size,
            "hidden_length": self.hidden_length,
            "hidden_width": self.hidden_width,
            "dropout_percentage": self.dropout,
            "decision_dim": self.decision_dim,
            "norm_layer": self.norm_layer,
        }
        args.update(kwargs)

        model_class = self._models.get(self.schema.policy, None)
        if model_class is None:
            raise NotImplementedError(f"Policy {self.schema.policy} not Implemented")

        if model_class is ClassicLCM:
            args["policy_embedding_size"] = self.policy_embedding_size

        if model_class is KapLCM:
            args["max_levels"] = self.max_levels

        model = model_class(**args)

        return model
