from typing import Any, Optional, Tuple

from torch import Tensor
import scipy.optimize as SciOpt
import torch

from endure.lcm.util import eval_lcm_impl
from endure.lsm.cost import EndureCost
from endure.lsm.types import LSMBounds, LSMDesign, System, Policy
from endure.ltune.data.generator import LTuneDataGenerator
from endure.ltune.loss import LearnedCostModelLoss
import endure.lsm.solver as Solver


class LTuneEvalUtil:
    def __init__(
        self,
        config: dict[str, Any],
        model: torch.nn.Module,
        design_type: Policy,
    ) -> None:
        self.bounds = LSMBounds()
        self.gen = LTuneDataGenerator(self.bounds)
        self.loss = LearnedCostModelLoss(
            config, config["job"]["LTuneTrain"]["loss_fn_path"]
        )
        self.max_t = self.bounds.size_ratio_range[1]
        self.min_t = self.bounds.size_ratio_range[0]
        self.model = model
        self.cf = EndureCost(self.bounds.max_considered_levels)
        self.config = config
        self.design_type = design_type

    def calc_size_ratio_range(self) -> int:
        return self.max_t - self.min_t

    def eval_lcm(
        self,
        design: LSMDesign,
        system: System,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        return eval_lcm_impl(
            design, system, z0, z1, q, w, self.loss.model, self.min_t, self.max_t
        )

    def eval_lcm_direct(
        self,
        model_out: Tensor,
        system: System,
        z0: float,
        z1: float,
        q: float,
        w: float,
    ) -> float:
        feat = torch.Tensor(
            [z0, z1, q, w, system.B, system.s, system.E, system.H, system.N]
        )
        feat = feat.view(1, -1)
        inputs = torch.concat([feat, model_out], dim=-1)
        with torch.no_grad():
            pred = self.loss.model(inputs)
            pred = pred.sum().item()

        return pred

    def get_ltune_out(
        self,
        system: System,
        z0: float,
        z1: float,
        q: float,
        w: float,
        temp=1e-2,
        hard=True,
    ) -> Tensor:
        x = torch.Tensor(
            [z0, z1, q, w, system.B, system.s, system.E, system.H, system.N]
        )
        x = x.view(1, -1)
        with torch.no_grad():
            out = self.model(x, temp=temp, hard=hard)

        return out

    def get_solver_nominal_design(
        self,
        system: System,
        z0: float,
        z1: float,
        q: float,
        w: float,
        **kwargs,
    ) -> Tuple[LSMDesign, SciOpt.OptimizeResult]:
        if self.design_type == Policy.QFixed:
            solver = Solver.QLSMSolver(self.bounds)
        elif self.design_type == Policy.KHybrid:
            solver = Solver.KLSMSolver(self.bounds)
        elif self.design_type == Policy.YZHybrid:
            solver = Solver.YZLSMSolver(self.bounds)
        else:  # design_type == Policy.Classic
            solver = Solver.ClassicSolver(self.bounds)

        design, sol = solver.get_nominal_design(system, z0, z1, q, w, **kwargs)

        return design, sol

    def convert_ltune_output(self, output: Tensor):
        if self.design_type == Policy.QFixed:
            design = self._qlsm_convert(output)
        elif self.design_type == Policy.KHybrid:
            design = self._klsm_convert(output)
        elif self.design_type == Policy.YZHybrid:
            design = self._yzlsm_convert(output)
        else:  # self.design_type == Policy.Classic
            design = self._classic_convert(output)

        return design

    def _klsm_convert(self, output: Tensor) -> LSMDesign:
        out = output.flatten()
        cap_range = self.calc_size_ratio_range()
        h = out[0].item()
        caps = out[1:].reshape(-1, cap_range)
        t = torch.argmax(caps[0]).item() + 2
        k = [torch.argmax(x).item() + 1 for x in caps[1:]]

        return LSMDesign(h=h, T=t, K=k, policy=Policy.KHybrid)

    def _qlsm_convert(self, output: Tensor) -> LSMDesign:
        out = output.flatten()
        cap_range = self.calc_size_ratio_range()
        h = out[0].item()
        caps = out[1:].reshape(-1, cap_range)
        t = torch.argmax(caps[0]).item() + 2
        q = torch.argmax(caps[1]).item() + 1

        return LSMDesign(h=h, T=t, Q=q, policy=Policy.QFixed)

    def _yzlsm_convert(self, output: Tensor) -> LSMDesign:
        out = output.flatten()
        cap_range = self.calc_size_ratio_range()
        h = out[0].item()
        caps = out[1:].reshape(-1, cap_range)
        t = torch.argmax(caps[0]).item() + 2
        y = torch.argmax(caps[1]).item() + 1
        z = torch.argmax(caps[2]).item() + 1

        return LSMDesign(h=h, T=t, Y=y, Z=z, policy=Policy.YZHybrid)

    def _classic_convert(self, output: Tensor) -> LSMDesign:
        out = output.flatten()
        cap_range = self.calc_size_ratio_range()
        h = out[0].item()
        t = torch.argmax(out[1 : cap_range + 1]).item() + 2
        policy_val = torch.argmax(out[cap_range + 1 :]).item()
        if policy_val:
            policy = Policy.Leveling
        else:
            policy = Policy.Tiering

        return LSMDesign(h=h, T=t, policy=policy)

    def gen_sample_eval(
        self,
        system: Optional[System] = None,
        solver_nominal_kwargs: dict = {},
    ) -> dict[str, Any]:
        if system is None:
            system = self.gen._sample_system()
        z0, z1, q, w = self.gen._sample_workload(4)

        stune_design, _ = self.get_solver_nominal_design(
            system, z0, z1, q, w, **solver_nominal_kwargs
        )
        stune_design.T = int(stune_design.T)
        if stune_design.policy == Policy.QFixed:
            stune_design.Q = int(stune_design.Q)
        elif stune_design.policy == Policy.KHybrid:
            stune_design.K = [int(k) for k in stune_design.K]
        elif stune_design.policy == Policy.YZHybrid:
            stune_design.Y = int(stune_design.Y)
            stune_design.Z = int(stune_design.Z)
        stune_loss = self.eval_lcm(stune_design, system, z0, z1, q, w)
        stune_cost = self.cf.calc_cost(stune_design, system, z0, z1, q, w)
        stune_level = self.cf.L(stune_design, system, ceil=True)

        out = self.get_ltune_out(system, z0, z1, q, w)
        ltune_design = self.convert_ltune_output(out)
        ltune_loss = self.eval_lcm(ltune_design, system, z0, z1, q, w)
        ltune_loss_direct = self.eval_lcm_direct(out, system, z0, z1, q, w)
        ltune_cost = self.cf.calc_cost(ltune_design, system, z0, z1, q, w)
        ltune_level = self.cf.L(ltune_design, system, ceil=True)

        row = {
            "z0": z0,
            "z1": z1,
            "q": q,
            "w": w,
            "B": system.B,
            "s": system.s,
            "E": system.E,
            "H": system.H,
            "N": system.N,
            "stune_policy": stune_design.policy.value,
            "stune_h": stune_design.h,
            "stune_T": stune_design.T,
            "stune_level": stune_level,
            "stune_cost": stune_cost,
            "stune_loss": stune_loss,
            "ltune_policy": ltune_design.policy.value,
            "ltune_h": ltune_design.h,
            "ltune_T": ltune_design.T,
            "ltune_level": ltune_level,
            "ltune_cost": ltune_cost,
            "ltune_loss": ltune_loss,
            "ltune_loss_direct": ltune_loss_direct,
        }
        if stune_design.policy == Policy.QFixed:
            row["stune_Q"] = stune_design.Q
            row["ltune_Q"] = ltune_design.Q
        elif stune_design.policy == Policy.YZHybrid:
            row["stune_Y"] = stune_design.Y
            row["ltune_Y"] = ltune_design.Y
            row["stune_Z"] = stune_design.Z
            row["ltune_Z"] = ltune_design.Z
        elif stune_design.policy == Policy.KHybrid:
            row["stune_K"] = stune_design.K
            row["ltune_K"] = ltune_design.K

        return row

