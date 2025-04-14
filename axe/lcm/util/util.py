from torch import Tensor
import torch
import torch.nn.functional as F

from axe.lsm.types import LSMDesign, Policy, System, Workload


def one_hot_lcm(
    data: Tensor, num_features: int, categorical_features: int, categories: int
) -> Tensor:
    capacities = data[num_features - categorical_features :]
    capacities = capacities.to(torch.long)
    capacities = F.one_hot(capacities, num_classes=categories)
    capacities = torch.flatten(capacities)
    out = [data[: num_features - categorical_features], capacities]
    out = torch.cat(out)

    return out


def one_hot_lcm_classic_distinct(
    data: Tensor, categories: int, size_ratio: Tensor, policy: Tensor
) -> Tensor:
    size_ratio = F.one_hot(size_ratio, num_classes=categories)
    policy = F.one_hot(policy, num_classes=2)
    out = [data[:-2], size_ratio, policy]
    out = torch.cat(out)

    return out


def one_hot_lcm_classic(data: Tensor, categories: int) -> Tensor:
    policy = data[-2].to(torch.long)
    size_ratio = data[-1].to(torch.long)
    policy = F.one_hot(policy, num_classes=2)
    size_ratio = F.one_hot(size_ratio, num_classes=categories)
    out = [data[:-2], size_ratio, policy]
    out = torch.cat(out)

    return out


def create_input_from_types(
    design: LSMDesign,
    system: System,
    workload: Workload,
    min_t: int,
    max_t: int,
) -> Tensor:
    categories = max_t - min_t
    wl = [workload.z0, workload.z1, workload.q, workload.w]
    sys = [
        system.entries_per_page,
        system.selectivity,
        system.entry_size,
        system.mem_budget,
        system.num_entries,
    ]
    size_ratio_idx = design.size_ratio - min_t
    if design.policy in (Policy.Tiering, Policy.Leveling, Policy.Classic):
        inputs = wl + sys + [design.bits_per_elem, size_ratio_idx, design.policy.value]
        data = torch.Tensor(inputs)
        size_ratio = torch.tensor(size_ratio_idx).to(torch.long)
        policy = torch.tensor(design.policy.value).to(torch.long)
        out = one_hot_lcm_classic_distinct(data, categories, size_ratio, policy)
    elif design.policy == Policy.Kapacity:
        ks = [k - 1 if k > 0 else 0 for k in design.kapacity]
        inputs = wl + sys + [design.bits_per_elem, size_ratio_idx] + ks
        data = torch.Tensor(inputs)
        num_feats = 1 + len(design.kapacity)
        out = one_hot_lcm(data, len(inputs), num_feats, categories)
    elif design.policy == Policy.Fluid:  # design.policy == Policy.QFixed
        inputs = wl + sys + [design.bits_per_elem, size_ratio_idx, design.kapacity[0] - 1, design.kapacity[1] - 1]
        data = torch.Tensor(inputs)
        out = one_hot_lcm(data, len(inputs), 3, categories)
    else:  # design.policy == Policy.QFixed
        inputs = wl + sys + [design.bits_per_elem, size_ratio_idx, design.kapacity[0] - 1]
        data = torch.Tensor(inputs)
        out = one_hot_lcm(data, len(inputs), 2, categories)

    return out


def eval_lcm_impl(
    design: LSMDesign,
    system: System,
    workload: Workload,
    model: torch.nn.Module,
    min_t: int,
    max_t: int,
) -> float:
    x = create_input_from_types(design, system, workload, min_t, max_t)
    x = x.to(torch.float).view(1, -1)
    model.eval()
    with torch.no_grad():
        pred = model(x)
        pred = pred.sum().item()

    return pred
