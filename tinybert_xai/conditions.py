"""Typed student distillation condition definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    logit: bool
    hidden: bool
    attention: bool

    @property
    def uses_teacher(self) -> bool:
        return self.logit or self.hidden or self.attention


CE_ONLY = ConditionSpec("ce_only", False, False, False)
KD_LOGIT = ConditionSpec("kd_logit", True, False, False)
KD_HIDDEN = ConditionSpec("kd_hidden", False, True, False)
KD_ATTN = ConditionSpec("kd_attn", False, False, True)
KD_LOGIT_HIDDEN = ConditionSpec("kd_logit_hidden", True, True, False)
KD_LOGIT_ATTN = ConditionSpec("kd_logit_attn", True, False, True)
KD_HIDDEN_ATTN = ConditionSpec("kd_hidden_attn", False, True, True)
KD_FULL = ConditionSpec("kd_full", True, True, True)


ALL_CONDITIONS = (
    CE_ONLY,
    KD_LOGIT,
    KD_HIDDEN,
    KD_ATTN,
    KD_LOGIT_HIDDEN,
    KD_LOGIT_ATTN,
    KD_HIDDEN_ATTN,
    KD_FULL,
)
