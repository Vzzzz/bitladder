"""Shared argparse pieces for experiment scripts."""

from __future__ import annotations

import argparse

from .quant.layers import QuantConfig, parse_overrides


def add_quant_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("quantization")
    g.add_argument("--scheme", default="fp32", choices=["fp32", "fp16", "uniform", "ternary", "twn", "ttq", "binary"])
    g.add_argument("--wbits", type=int, default=8, help="bits for the uniform scheme")
    g.add_argument("--abits", type=int, default=None, help="PACT activation bits; omit for float activations")
    g.add_argument("--first-last-bits", type=int, default=8, help="uniform bits for first/last layer (32 = float)")
    g.add_argument("--no-first-last", action="store_true", help="quantize first/last layer like the rest")
    g.add_argument("--per-tensor", action="store_true", help="per-tensor scales instead of per-channel")
    g.add_argument("--lsq", action="store_true", help="learned step size for the uniform scheme")
    g.add_argument("--overrides", default="", help="per-layer precision: 'blocks.*.pw=2,stem=8,fc=ternary'")
    g.add_argument("--mixed-from", default=None, help="analysis JSON (analyze.py) for a greedy mixed-precision profile")
    g.add_argument("--mixed-budget", type=float, default=0.01,
                   help="summed one-shot accuracy drop allowed (0.01 = 1 pt)")


def quant_config(args: argparse.Namespace) -> QuantConfig:
    overrides = parse_overrides(args.overrides) if args.overrides else None
    if getattr(args, "mixed_from", None):
        import json

        from .measure.mixed import greedy_profile

        report = json.load(open(args.mixed_from, encoding="utf-8"))
        prof = greedy_profile(report, args.mixed_budget)
        overrides = {**prof, **(overrides or {})}  # explicit overrides win
    return QuantConfig(
        scheme=args.scheme,
        wbits=args.wbits,
        abits=args.abits,
        per_channel=not args.per_tensor,
        lsq=args.lsq,
        first_last_bits=None if args.no_first_last else args.first_last_bits,
        overrides=overrides,
    )
