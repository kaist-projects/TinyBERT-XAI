"""YAML run specification: one file fully describes a training run.

A ``RunSpec`` bundles the training ``Config`` (hyperparameters) with the
run-level selections that are not hyperparameters: which dataset and which
distillation condition. Training always evaluates on dev/test at the end, so
evaluation is not a config option. ``load_run_spec`` reads a YAML file into a
``RunSpec``; omitted keys fall back to ``Config()`` / run defaults, and unknown
keys raise ``ValueError`` to catch typos.

The committed root ``config.yaml`` (loaded by the scripts when no ``--config`` is
given) reproduces the design-doc-locked recipe, so it matches the built-in
``RunSpec()`` defaults byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.config import Config

DEFAULT_DATASET = "tweet_eval-sentiment"
_SIGNALS = ("logit", "hidden", "attention")

# Allowed YAML keys per section. Maps each config-facing key to its flat Config
# field; run/distillation handled separately. Unknown keys anywhere -> ValueError.
_MODEL_KEYS = {
    "teacher": "teacher_checkpoint",
    "student": "student_checkpoint",
    "tokenizer": "tokenizer_checkpoint",
}
_TRAINING_KEYS = {
    "seed": "seed",
    "device": "device",
    "precision": "precision",
    "max_seq_length": "max_seq_length",
    "learning_rate": "learning_rate",
    "train_batch_size": "train_batch_size",
    "eval_batch_size": "eval_batch_size",
    "num_epochs": "num_epochs",
    "patience": "patience",
}
_LOSS_WEIGHT_KEYS = {"ce": "ce_weight", "logit": "logit_weight", "hidden": "hidden_weight", "attn": "attn_weight"}
_RUN_KEYS = {"dataset", "conditions"}
_DISTILL_KEYS = {"logit_temperature", "loss_weights"}
_TOP_KEYS = {"run", "model", "training", "distillation"}


@dataclass(frozen=True)
class RunSpec:
    config: Config = field(default_factory=Config)
    dataset: str = DEFAULT_DATASET
    logit: bool = False
    hidden: bool = False
    attention: bool = False


def load_run_spec(path: str | Path) -> RunSpec:
    """Parse a YAML file into a validated RunSpec."""
    with open(path) as f:
        mapping = yaml.safe_load(f) or {}
    if not isinstance(mapping, dict):
        raise ValueError(f"Config root must be a mapping, got {type(mapping).__name__}")
    return run_spec_from_mapping(mapping)


def run_spec_from_mapping(mapping: dict) -> RunSpec:
    """Build a RunSpec from an already-parsed mapping (pure; unit-testable)."""
    _reject_unknown_keys("(root)", mapping, _TOP_KEYS)
    run = _section(mapping, "run", _RUN_KEYS)
    distillation = _section(mapping, "distillation", _DISTILL_KEYS)

    config = Config(**_config_kwargs(mapping, distillation))
    logit, hidden, attention = _condition_flags(run.get("conditions", {}))
    return RunSpec(
        config=config,
        dataset=run.get("dataset", DEFAULT_DATASET),
        logit=logit,
        hidden=hidden,
        attention=attention,
    )


def _config_kwargs(mapping: dict, distillation: dict) -> dict:
    """Flatten the model/training/distillation sections into Config(**kwargs)."""
    model = _section(mapping, "model", set(_MODEL_KEYS))
    training = _section(mapping, "training", set(_TRAINING_KEYS))
    loss_weights = distillation.get("loss_weights") or {}
    _reject_unknown_keys("distillation.loss_weights", loss_weights, set(_LOSS_WEIGHT_KEYS))

    kwargs: dict = {}
    for src, dst in _MODEL_KEYS.items():
        if src in model:
            kwargs[dst] = model[src]
    for src, dst in _TRAINING_KEYS.items():
        if src in training:
            kwargs[dst] = training[src]
    for src, dst in _LOSS_WEIGHT_KEYS.items():
        if src in loss_weights:
            kwargs[dst] = loss_weights[src]
    if "logit_temperature" in distillation:
        kwargs["logit_temperature"] = distillation["logit_temperature"]
    return kwargs


def _condition_flags(conditions: dict) -> tuple[bool, bool, bool]:
    if not isinstance(conditions, dict):
        raise ValueError(f"run.conditions must be a mapping of signal->bool, got {type(conditions).__name__}")
    _reject_unknown_keys("run.conditions", conditions, set(_SIGNALS))
    return tuple(bool(conditions.get(signal, False)) for signal in _SIGNALS)  # type: ignore[return-value]


def _section(mapping: dict, name: str, allowed: set[str]) -> dict:
    section = mapping.get(name) or {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section {name!r} must be a mapping, got {type(section).__name__}")
    _reject_unknown_keys(name, section, allowed)
    return section


def _reject_unknown_keys(where: str, mapping: dict, allowed: set[str]) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"Unknown key(s) in {where}: {sorted(unknown)}; allowed: {sorted(allowed)}")
