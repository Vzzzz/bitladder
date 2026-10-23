"""Shared argparse pieces for experiment scripts."""

from __future__ import annotations

import argparse

from .quant.layers import QuantConfig


def add_quant_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("quantization")
    g.add_argument("--scheme", default="fp32", choices=["fp32", "fp16", "uniform", "ternary", "binary"])
    g.add_argument("--wbits", type=int, default=8, help="bits for the uniform scheme")
    g.add_argument("--abits", type=int, default=None, help="PACT activation bits; omit for float activations")
    g.add_argument("--first-last-bits", type=int, default=8, help="uniform bits for first/last layer (32 = float)")
    g.add_argument("--no-first-last", action="store_true", help="quantize first/last layer like the rest")
    g.add_argument("--per-tensor", action="store_true", help="per-tensor scales instead of per-channel")
    g.add_argument("--lsq", action="store_true", help="learned step size for the uniform scheme")


def quant_config(args: argparse.Namespace) -> QuantConfig:
    return QuantConfig(
        scheme=args.scheme,
        wbits=args.wbits,
        abits=args.abits,
        per_channel=not args.per_tensor,
        lsq=args.lsq,
        first_last_bits=None if args.no_first_last else args.first_last_bits,
    )
