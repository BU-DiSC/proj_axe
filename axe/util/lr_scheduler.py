from typing import Optional, Any

import torch.optim as Opt


class LRSchedulerBuilder:
    def __init__(self, config: dict[str, Any]):
        self.scheduler_kwargs = config

    def _build_cosine_anneal(
        self,
        optimizer: Opt.Optimizer,
    ) -> Opt.lr_scheduler.CosineAnnealingLR:
        return Opt.lr_scheduler.CosineAnnealingLR(
            optimizer, **self.scheduler_kwargs["CosineAnnealingLR"]
        )

    def _build_exponential(
        self,
        optimizer: Opt.Optimizer,
    ) -> Opt.lr_scheduler.ExponentialLR:
        return Opt.lr_scheduler.ExponentialLR(
            optimizer,
            **self.scheduler_kwargs["Exponential"],
        )

    def build(
        self, optimizer: Opt.Optimizer, scheduler_choice: str = "Constant"
    ) -> Optional[Opt.lr_scheduler._LRScheduler]:
        schedules = {
            "CosineAnnealing": self._build_cosine_anneal,
            "Exponential": self._build_exponential,
            "Constant": "Constant",
            "None": "Constant",
        }
        schedule_builder = schedules.get(scheduler_choice, "Constant")
        if schedule_builder == "Constant":
            return None
        scheduler = schedule_builder(optimizer)

        return scheduler
