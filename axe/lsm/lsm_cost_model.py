import numpy as np
from numba import jit


@jit(nopython=True)
def calc_mbuff(bpe: float, max_bits: float, num_elem: int) -> float:
    return (max_bits - bpe) * num_elem


@jit(nopython=True)
def calc_level(
    bpe: float,
    size_ratio: float,
    entry_size: int,
    max_bits: float,
    num_elem: int,
    ceil: bool = False,
) -> float:
    level = np.log(((num_elem * entry_size) / calc_mbuff(bpe, max_bits, num_elem)) + 1)
    level /= np.log(size_ratio)
    if ceil:
        level = np.ceil(level)

    return level


@jit(nopython=True)
def calc_level_fp(
    level: int,
    bpe: float,
    size_ratio: float,
    entry_size: int,
    max_bits: float,
    num_elem: int,
) -> float:
    max_level = calc_level(bpe, size_ratio, entry_size, max_bits, num_elem, True)
    alpha = np.exp(-bpe * (np.log(2) ** 2))
    top = size_ratio ** (size_ratio / (size_ratio - 1))
    bot = size_ratio ** (max_level + 1 - level)

    return alpha * (top / bot)


@jit(nopython=True)
def calc_full_tree(
    tot_levels: int,
    bpe: float,
    size_ratio: float,
    entry_size: int,
    max_bits: float,
    num_elem: int,
) -> float:
    nfull = 0
    mbuff = calc_mbuff(bpe, max_bits, num_elem)
    for level in range(1, tot_levels + 1):
        nfull += (size_ratio - 1) * (size_ratio ** (level - 1)) * mbuff / entry_size

    return nfull


@jit(nopython=True)
def calc_run_prob(
    level: int, size_ratio: float, entry_size: int, mbuff: float, nfull: float
) -> float:
    return (size_ratio - 1) * mbuff * (size_ratio ** (level - 1)) / (nfull * entry_size)


@jit(nopython=True)
def empty_op(
    h: float, T: float, K: np.ndarray, num_elem: int, entry_size: int, max_bits: float
) -> float:
    z0 = 0
    max_level = int(calc_level(h, T, entry_size, max_bits, num_elem, ceil=True))
    for i in range(1, max_level + 1):
        z0 += K[i - 1] * calc_level_fp(i, h, T, entry_size, max_bits, num_elem)

    return z0


@jit(nopython=True)
def non_empty_op(
    h: float, T: float, K: np.ndarray, entry_size: int, max_bits: float, num_elem: int
) -> float:
    mbuff = calc_mbuff(h, max_bits, num_elem)
    max_level = int(calc_level(h, T, entry_size, max_bits, num_elem, ceil=True))
    nfull = calc_full_tree(max_level, h, T, entry_size, max_bits, num_elem)

    z1 = 0
    for level in range(1, max_level + 1):
        upper_fp = 0
        run_prob = calc_run_prob(level, T, entry_size, mbuff, nfull)
        level_fp = calc_level_fp(level, h, T, entry_size, max_bits, num_elem)
        for idx in range(1, level):
            upper_fp += K[idx - 1] * calc_level_fp(
                idx, h, T, entry_size, max_bits, num_elem
            )
        current_fp = ((K[level - 1] - 1) / 2) * level_fp
        z1 += run_prob * (1 + upper_fp + current_fp)

    return z1


@jit(nopython=True)
def range_op(
    h: float,
    T: float,
    K: np.ndarray,
    entry_per_page: int,  # B
    selectivity: float,  # s
    entry_size: int,  # E
    max_bits: float,  # H
    num_elem: int,  # N
) -> float:
    max_level = int(calc_level(h, T, entry_size, max_bits, num_elem, ceil=True))
    fuzz_level = calc_level(h, T, entry_size, max_bits, num_elem, ceil=False)
    q = sum(K[: (max_level - 1)])
    residual = 1 - (max_level - fuzz_level)
    q += K[max_level - 1] * residual
    q = q + (selectivity * num_elem / entry_per_page)

    return q


@jit(nopython=True)
def write_op(
    h: float,
    T: float,
    K: np.ndarray,
    entry_per_page: int,
    entry_size: int,
    max_bits: float,
    num_elem: int,
    phi: float,
) -> float:
    w = 0
    max_level = int(calc_level(h, T, entry_size, max_bits, num_elem, ceil=True))
    fuzz_level = calc_level(h, T, entry_size, max_bits, num_elem, ceil=False)
    for level in range(0, max_level - 1):
        w += (T - 1 + K[level]) / (2 * K[level])
    residual = 1 - (max_level - fuzz_level)
    w += residual * (T - 1 + K[max_level - 1]) / (2 * K[max_level - 1])
    w *= (1 + phi) / entry_per_page

    return w


@jit(nopython=True)
def calc_cost(
    h: float,
    T: float,
    K: np.ndarray,
    z0: float,
    z1: float,
    q: float,
    w: float,
    entry_per_page: int,  # B
    selectivity: float,  # s
    entry_size: int,  # E
    max_bits: float,  # H
    num_elem: int,  # N
    phi: float,
) -> float:
    cost = 0
    if np.isnan(h) or np.isnan(T) or np.isnan(K).any():
        return np.finfo(np.float64).max.item()

    cost += z0 * empty_op(h, T, K, num_elem, entry_size, max_bits)
    cost += z1 * non_empty_op(h, T, K, entry_size, max_bits, num_elem)
    cost += q * range_op(
        h, T, K, entry_per_page, selectivity, entry_size, max_bits, num_elem
    )
    cost += w * write_op(h, T, K, entry_per_page, entry_size, max_bits, num_elem, phi)

    return cost


@jit(nopython=True)
def calc_individual_cost(
    h: float,
    T: float,
    K: np.ndarray,
    z0: float,
    z1: float,
    q: float,
    w: float,
    entry_per_page: int,  # B
    selectivity: float,  # s
    entry_size: int,  # E
    max_bits: float,  # H
    num_elem: int,  # N
    phi: float,
) -> tuple[float, float, float, float]:
    if np.isnan(h) or np.isnan(T) or np.isnan(K).any():
        return (
            np.finfo(np.float64).max.item(),
            np.finfo(np.float64).max.item(),
            np.finfo(np.float64).max.item(),
            np.finfo(np.float64).max.item(),
        )

    c_z0 = z0 * empty_op(h, T, K, num_elem, entry_size, max_bits)
    c_z1 = z1 * non_empty_op(h, T, K, entry_size, max_bits, num_elem)
    c_q = q * range_op(
        h, T, K, entry_per_page, selectivity, entry_size, max_bits, num_elem
    )
    c_w = w * write_op(h, T, K, entry_per_page, entry_size, max_bits, num_elem, phi)

    return (c_z0, c_z1, c_q, c_w)
