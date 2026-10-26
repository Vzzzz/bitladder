"""Greedy mixed-precision assignment from a one-shot sensitivity report (analyze.py).

Rule in the spirit of Rusci et al. 2020 (memory-driven): start with every layer at the
highest probed bit width, then repeatedly lower the bits of the layer that saves the most
bytes per point of accuracy drop, as long as the summed (one-shot) drop stays within
``budget``. The sum of one-shot drops is only a proxy: the result is a *starting profile*
for QAT, not a guarantee.

    from bitladder.measure.mixed import greedy_profile
    prof = greedy_profile(report, budget=1.0)        # {'stem': 8, 'blocks.0.pw': 2, ...}
    cfg = QuantConfig(scheme='uniform', wbits=8, abits=8, overrides=prof)

``report`` is the dict written by ``analysis_report`` (keys ``weights``, ``sensitivity``);
only probes named ``wK`` (uniform K bits) and ``w1.58`` (ternary) are considered.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

TERNARY = "w1.58"


def _probe_bits(label: str) -> Optional[float]:
    if label == TERNARY:
        return 1.58
    if label.startswith("w") and label[1:].isdigit():
        return float(label[1:])
    return None


def layer_bytes(params: int, bits: float) -> float:
    return params * bits / 8.0


def greedy_profile(report: Dict, budget: float, allow_ternary: bool = True,
                   min_bits: float = 1.58, fixed: Optional[Dict[str, int]] = None
                   ) -> Dict[str, Union[int, str]]:
    """Return {layer: bits | 'ternary'} minimising bytes under a summed-drop budget (in
    metric units, e.g. 0.01 = one accuracy point)."""
    sens = report["sensitivity"]
    params = {w["name"]: int(w["params"]) for w in report["weights"]}
    probes: List[Tuple[int, float]] = []  # (column, bits)
    for j, label in enumerate(sens["probes"]):
        b = _probe_bits(label)
        if b is None or b < min_bits or (b == 1.58 and not allow_ternary):
            continue
        probes.append((j, b))
    probes.sort(key=lambda t: -t[1])  # high bits first
    if not probes:
        raise ValueError("report has no wK / w1.58 probes")
    top_bits = probes[0][1]
    drop = {name: {b: max(0.0, sens["drop"][i][j]) for j, b in probes} for i, name in enumerate(sens["layers"])}
    current: Dict[str, float] = {name: top_bits for name in sens["layers"]}
    fixed = fixed or {}
    for name, b in fixed.items():
        current[name] = float(b)
    used = sum(drop[n][current[n]] if current[n] in drop[n] else 0.0 for n in current)
    while True:
        best = None
        for name in current:
            if name in fixed:
                continue
            lower = [b for _, b in probes if b < current[name]]
            if not lower:
                continue
            nb = lower[0]  # next step down
            extra = drop[name][nb] - drop[name].get(current[name], 0.0)
            if used + extra > budget:
                continue
            saved = layer_bytes(params[name], current[name]) - layer_bytes(params[name], nb)
            score = saved / (extra + 1e-9)
            if best is None or score > best[0]:
                best = (score, name, nb, extra)
        if best is None:
            break
        _, name, nb, extra = best
        current[name] = nb
        used += extra
    out: Dict[str, Union[int, str]] = {}
    for name, b in current.items():
        out[name] = "ternary" if b == 1.58 else int(b)
    return out


def profile_bytes(report: Dict, profile: Dict[str, Union[int, str]]) -> float:
    params = {w["name"]: int(w["params"]) for w in report["weights"]}
    return sum(layer_bytes(params[n], 1.58 if b == "ternary" else float(b)) for n, b in profile.items())
