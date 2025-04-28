from typing import Any, Callable, Optional

import torch


class LossBuilder:
    def __init__(self, loss_kwargs: dict[str, Any]) -> None:
        self.loss_kwargs = loss_kwargs

    def build(self, choice: str) -> Optional[torch.nn.Module]:
        losses: dict[str, Callable] = {
            "MSLE": MSLELoss,
            "NMSE": NMSELoss,
            "RMSLE": RMSLELoss,
            "RMSE": RMSELoss,
            "MSE": self._build_mse,
            "Huber": self._build_huber,
        }
        loss = losses.get(choice, None)
        if loss is None:
            return None

        return loss()

    def _build_huber(self) -> torch.nn.Module:
        return torch.nn.HuberLoss(**self.loss_kwargs["Huber"])

    def _build_mse(self) -> torch.nn.Module:
        return torch.nn.MSELoss(**self.loss_kwargs["MSE"])


class MSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, label):
        return self.mse(torch.log(pred + 1), torch.log(label + 1))


class RMSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, label):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(label + 1)))


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, label):
        return torch.sqrt(self.mse(pred, label))


class NMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, label):
        return self.mse(pred, label) / torch.square(torch.sum(label))


class NMSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.msle = MSLELoss()

    def forward(self, pred, label):
        return self.msle(pred, label) / torch.sum(torch.square(torch.log(pred + 1)))
