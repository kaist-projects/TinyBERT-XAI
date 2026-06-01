import argparse
from pathlib import Path

import pytest

from tinybert_xai import Config, RunSpec, load_run_spec, run_spec_from_mapping

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_empty_mapping_reproduces_locked_defaults():
    spec = run_spec_from_mapping({})

    assert spec.config == Config()
    assert spec.dataset == "tweet_eval-sentiment"
    assert (spec.logit, spec.hidden, spec.attention, spec.eval) == (False, False, False, False)


def test_full_mapping_maps_every_field():
    spec = run_spec_from_mapping(
        {
            "run": {"dataset": "imdb", "condition": ["logit", "attention"], "eval": True},
            "model": {"teacher_checkpoint": "bert-x", "student_checkpoint": "tiny-y", "tokenizer_checkpoint": "tok-z"},
            "training": {
                "seed": 7,
                "device": "cpu",
                "precision": "fp32",
                "max_seq_length": 64,
                "learning_rate": 1.0e-4,
                "train_batch_size": 8,
                "eval_batch_size": 16,
                "num_epochs": 5,
                "patience": 1,
            },
            "distillation": {
                "logit_temperature": 2.0,
                "loss_weights": {"ce": 1.5, "logit": 2.5, "hidden": 3.5, "attn": 4.5},
            },
        }
    )

    assert (spec.dataset, spec.eval) == ("imdb", True)
    assert (spec.logit, spec.hidden, spec.attention) == (True, False, True)
    c = spec.config
    assert (c.teacher_checkpoint, c.student_checkpoint, c.tokenizer_checkpoint) == ("bert-x", "tiny-y", "tok-z")
    assert (c.seed, c.device, c.precision, c.max_seq_length) == (7, "cpu", "fp32", 64)
    assert (c.learning_rate, c.train_batch_size, c.eval_batch_size, c.num_epochs, c.patience) == (1.0e-4, 8, 16, 5, 1)
    assert c.logit_temperature == 2.0
    assert (c.ce_weight, c.logit_weight, c.hidden_weight, c.attn_weight) == (1.5, 2.5, 3.5, 4.5)


def test_partial_mapping_falls_back_to_defaults():
    spec = run_spec_from_mapping({"training": {"learning_rate": 9.9e-5}})

    assert spec.config.learning_rate == 9.9e-5
    assert spec.config.seed == Config().seed  # untouched fields keep defaults
    assert spec.dataset == "tweet_eval-sentiment"


@pytest.mark.parametrize(
    "mapping",
    [
        {"bogus": 1},  # unknown top-level
        {"training": {"learning_rat": 1.0}},  # typo'd nested key
        {"distillation": {"loss_weights": {"attention": 1.0}}},  # wrong weight key (attn, not attention)
    ],
)
def test_unknown_key_raises(mapping):
    with pytest.raises(ValueError):
        run_spec_from_mapping(mapping)


def test_unknown_condition_signal_raises():
    with pytest.raises(ValueError):
        run_spec_from_mapping({"run": {"condition": ["logit", "bogus"]}})


def test_default_yaml_equals_locked_config():
    spec = load_run_spec(REPO_ROOT / "configs" / "default.yaml")

    assert spec.config == Config()  # committed recipe must match the locked defaults


# --- CLI override resolution (precedence: defaults < YAML < explicit flag) ---


def _resolver():
    # Imported lazily: scripts/ is added to sys.path by the scripts themselves.
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import _config_cli

    return _config_cli


def test_cli_flag_overrides_yaml(tmp_path):
    cc = _resolver()
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("run:\n  dataset: imdb\n  eval: true\ndistillation:\n  loss_weights:\n    logit: 2.0\n")

    parser = argparse.ArgumentParser()
    cc.add_config_flag(parser)
    cc.add_dataset_override(parser)
    cc.add_eval_override(parser)
    cc.add_signal_overrides(parser)
    cc.add_weight_overrides(parser)

    # CLI overrides dataset + eval + one weight; condition signal added on top.
    args = parser.parse_args(["--config", str(cfg_file), "--dataset", "anli", "--no-eval", "--logit-weight", "9.0"])
    spec = cc.resolve_run_spec(args)

    assert spec.dataset == "anli"  # CLI beats YAML
    assert spec.eval is False  # --no-eval beats YAML eval: true
    assert spec.config.logit_weight == 9.0  # CLI beats YAML's 2.0


def test_absent_flag_keeps_yaml_value(tmp_path):
    cc = _resolver()
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("run:\n  dataset: imdb\n  eval: true\n")

    parser = argparse.ArgumentParser()
    cc.add_config_flag(parser)
    cc.add_dataset_override(parser)
    cc.add_eval_override(parser)

    args = parser.parse_args(["--config", str(cfg_file)])  # no overrides
    spec = cc.resolve_run_spec(args)

    assert spec.dataset == "imdb"  # YAML retained
    assert spec.eval is True
