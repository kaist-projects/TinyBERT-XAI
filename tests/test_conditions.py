from tinybert_xai.distill.conditions import all_conditions, condition_from_flags

CANONICAL_NAMES = [
    "ce_only",
    "kd_logit",
    "kd_hidden",
    "kd_attn",
    "kd_logit_hidden",
    "kd_logit_attn",
    "kd_hidden_attn",
    "kd_full",
]


def test_name_derived_from_flags():
    assert condition_from_flags(False, False, False).name == "ce_only"
    assert condition_from_flags(True, False, False).name == "kd_logit"
    assert condition_from_flags(True, False, True).name == "kd_logit_attn"
    assert condition_from_flags(True, True, True).name == "kd_full"


def test_uses_teacher_tracks_any_signal():
    assert not condition_from_flags(False, False, False).uses_teacher
    assert condition_from_flags(False, False, True).uses_teacher


def test_all_conditions_are_the_eight_in_canonical_order():
    assert [c.name for c in all_conditions()] == CANONICAL_NAMES
