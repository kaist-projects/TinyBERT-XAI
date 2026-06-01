"""Shared --config glue: load a YAML RunSpec and overlay explicit CLI flags.

Precedence is ``Config()`` / run defaults < YAML (``--config``) < explicit CLI
flags. Override flags register with ``argparse.SUPPRESS`` so only the ones the
user actually passes land in the namespace; anything absent keeps the YAML/base
value. Boolean flags use ``BooleanOptionalAction`` (``--eval/--no-eval``,
``--logit/--no-logit``, ...) so the CLI can both set and unset a YAML baseline.

Imported by the teacher/student train + eval scripts, which insert the repo root
onto sys.path first, so ``from tinybert_xai import ...`` resolves.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from tinybert_xai import ALL_DATASETS, RunSpec, load_run_spec

_UNSET = object()
_SIGNALS = ("logit", "hidden", "attention")
# CLI flag attr -> Config field (identical names; argparse maps - to _).
_CONFIG_OVERRIDES = ("ce_weight", "logit_weight", "hidden_weight", "attn_weight", "logit_temperature")


def add_config_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="YAML run config (CLI flags override it)")


def add_dataset_override(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        choices=[spec.name for spec in ALL_DATASETS],
        default=argparse.SUPPRESS,
        help="override run.dataset",
    )


def add_signal_overrides(parser: argparse.ArgumentParser) -> None:
    for signal in _SIGNALS:
        parser.add_argument(
            f"--{signal}",
            action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
            help=f"override condition signal: {signal}",
        )


def add_eval_override(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="evaluate on dev/test after training (overrides run.eval)",
    )


def add_weight_overrides(parser: argparse.ArgumentParser) -> None:
    for name in ("ce", "logit", "hidden", "attn"):
        parser.add_argument(
            f"--{name}-weight", type=float, default=argparse.SUPPRESS, help=f"override {name} loss weight"
        )
    parser.add_argument(
        "--logit-temperature", type=float, default=argparse.SUPPRESS, help="override logit KD temperature"
    )


def resolve_run_spec(args: argparse.Namespace) -> RunSpec:
    """Overlay any explicitly-passed CLI flags onto the YAML/default RunSpec."""
    base = load_run_spec(args.config) if getattr(args, "config", None) else RunSpec()

    config = replace(base.config, **_present(args, _CONFIG_OVERRIDES))
    run_overrides = _present(args, ("dataset", "eval", *_SIGNALS))
    return replace(base, config=config, **run_overrides)


def _present(args: argparse.Namespace, names: tuple[str, ...]) -> dict:
    """Return {name: value} for flags actually present in the namespace."""
    return {name: value for name in names if (value := getattr(args, name, _UNSET)) is not _UNSET}
