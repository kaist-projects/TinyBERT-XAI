"""Typed student distillation condition definitions.

A condition is fully determined by three boolean signal flags
(logit / hidden / attention). The canonical ``kd_*`` name and the full 2**3
factorial set are *derived* from those flags -- there are no named constants
and no name->signal lookup table. Flags are the sole source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

SIGNAL_NAMES = ("logit", "hidden", "attention")
# Tokens used to build the canonical condition name (attention is abbreviated).
_NAME_TOKENS = ("logit", "hidden", "attn")


@dataclass(frozen=True)
class ConditionSpec:
    logit: bool
    hidden: bool
    attention: bool

    @property
    def name(self) -> str:
        active = [token for token, on in zip(_NAME_TOKENS, self._flags) if on]
        if not active:
            return "ce_only"
        if len(active) == len(_NAME_TOKENS):
            return "kd_full"
        return "kd_" + "_".join(active)

    @property
    def uses_teacher(self) -> bool:
        return any(self._flags)

    @property
    def _flags(self) -> tuple[bool, bool, bool]:
        return (self.logit, self.hidden, self.attention)


def condition_from_flags(logit: bool, hidden: bool, attention: bool) -> ConditionSpec:
    """Build a condition from the three signal flags."""
    return ConditionSpec(logit=logit, hidden=hidden, attention=attention)


def all_conditions() -> tuple[ConditionSpec, ...]:
    """Generate the full 2**3 factorial of conditions in canonical order.

    Order is by number of active signals, then by signal index, reproducing
    ``ce_only, kd_logit, kd_hidden, kd_attn, kd_logit_hidden, kd_logit_attn,
    kd_hidden_attn, kd_full``.
    """
    specs = [ConditionSpec(*flags) for flags in product((False, True), repeat=len(SIGNAL_NAMES))]
    return tuple(sorted(specs, key=_canonical_order_key))


def _canonical_order_key(spec: ConditionSpec) -> tuple[int, tuple[int, ...]]:
    active_indices = tuple(index for index, on in enumerate(spec._flags) if on)
    return (len(active_indices), active_indices)
