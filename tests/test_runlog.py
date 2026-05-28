import json

from tinybert_xai.runlog import RunMetadata, make_run_id, write_run_metadata


def test_schema_v2_rounding_and_teacher_fields(tmp_path):
    meta = RunMetadata(
        schema_version="2",
        run={
            "run_id": "teacher-test",
            "stage": "teacher",
            "condition": None,
        },
        dataset={
            "name": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "num_labels": 3,
            "label_names": ["negative", "neutral", "positive"],
            "splits": {"train": 45615, "validation": 2000, "test": 12284},
            "max_seq_length": 128,
            "truncation": True,
            "padding": "max_length",
        },
        model={
            "checkpoint": "bert-base-uncased",
            "tokenizer": "bert-base-uncased",
            "parameter_count": 109484547,
        },
        optimization={
            "optimizer": "AdamW",
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "scheduler": None,
            "grad_clip": None,
            "precision": "fp32",
            "train_batch_size": 16,
            "eval_batch_size": 32,
            "num_epochs": 3,
        },
        checkpoint_selection={
            "monitor": "dev_macro_f1",
            "mode": "max",
            "patience": 2,
            "best_epoch": 1,
            "early_stopped": False,
            "checkpoint": "checkpoints/teachers/tweet_eval-sentiment/best.pt",
        },
        reproducibility={
            "seed": 42,
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "shuffle_seed_scheme": "seed + epoch",
        },
        environment={
            "device": "cuda:0",
            "gpu_model": "NVIDIA GeForce RTX 3090",
            "gpu_memory_total_mb": 24118.25,
            "torch_cuda": "12.4",
        },
        training={
            "epochs_completed": 1,
            "train_time_seconds": 857.0005699680041,
            "history": [
                {
                    "epoch": 0,
                    "global_step": 2851,
                    "epoch_time_seconds": 273.7937828840004,
                    "loss_total": 0.6554829189544057,
                    "losses": {"ce": 0.6554829189544057},
                    "grad_norm_mean": 6.7234822630004185,
                    "dev": {
                        "macro_f1": 0.7066847225590859,
                        "micro_f1": 0.723,
                        "accuracy": 0.723,
                        "ECE": 0.017386078983545317,
                        "NLL": 0.6089550852775574,
                        "Brier": 0.36978474259376526,
                    },
                }
            ],
        },
        metrics={
            "dev": {
                "macro_f1": 0.7201311861596489,
                "micro_f1": 0.738,
                "accuracy": 0.738,
                "per_class_f1": [0.6525037936267072, 0.710703363914373],
                "confusion_matrix": [[215, 74, 23], [104, 581, 184]],
                "ECE": 0.06480345979332924,
                "NLL": 0.6314913034439087,
                "Brier": 0.37071138620376587,
            }
        },
    )

    path = tmp_path / "run_metadata.json"
    write_run_metadata(meta, path)

    text = path.read_text()
    payload = json.loads(text)

    assert payload["schema_version"] == "2"
    assert payload["run"] == {"run_id": "teacher-test", "stage": "teacher", "condition": None}
    assert payload["optimization"]["learning_rate"] == 2e-5
    assert payload["optimization"]["weight_decay"] == 0.01
    assert payload["optimization"]["eps"] == 1e-8
    assert payload["training"]["train_time_seconds"] == 857.00057
    assert payload["training"]["history"][0]["loss_total"] == 0.65548
    assert payload["training"]["history"][0]["epoch_time_seconds"] == 273.79378
    assert "efficiency" not in payload
    assert "best_dev_macro_f1" not in payload["training"]
    assert payload["metrics"]["dev"]["confusion_matrix"] == [[215, 74, 23], [104, 581, 184]]

    assert "train_raw_loss_ce" not in text
    assert "teacher_student_kl" not in text
    assert "git_commit" not in text
    assert "metric_definitions" not in text
    assert "cuda_available" not in text
    assert "package_versions" not in text
    assert "warn_only" not in text
    assert "family" not in text
    assert "[215, 74, 23]" in text


def test_student_schema_v2_uses_condition_and_active_losses(tmp_path):
    meta = RunMetadata(
        schema_version="2",
        run={
            "run_id": "student-ce_only-tweet_eval-sentiment-test",
            "stage": "student",
            "condition": "ce_only",
        },
        dataset={
            "name": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "num_labels": 3,
            "label_names": ["negative", "neutral", "positive"],
            "splits": {"train": 45615, "validation": 2000},
            "max_seq_length": 128,
            "truncation": True,
            "padding": "max_length",
        },
        model={
            "student_checkpoint": "huawei-noah/TinyBERT_General_4L_312D",
            "tokenizer": "bert-base-uncased",
            "parameter_count": 14351235,
        },
        optimization={
            "optimizer": "AdamW",
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "scheduler": None,
            "grad_clip": None,
            "precision": "bf16",
            "train_batch_size": 16,
            "eval_batch_size": 32,
            "num_epochs": 3,
        },
        checkpoint_selection={
            "monitor": "dev_macro_f1",
            "mode": "max",
            "patience": 2,
            "best_epoch": 1,
            "early_stopped": False,
            "checkpoint": "checkpoints/students/tweet_eval-sentiment/ce_only/best.pt",
        },
        reproducibility={
            "seed": 42,
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "shuffle_seed_scheme": "seed + epoch",
        },
        environment={
            "device": "cuda:0",
            "gpu_model": "NVIDIA GeForce RTX 3090",
            "gpu_memory_total_mb": 24118.25,
            "torch_cuda": "12.4",
        },
        training={
            "epochs_completed": 1,
            "train_time_seconds": 201.123456789,
            "history": [
                {
                    "epoch": 0,
                    "global_step": 2851,
                    "epoch_time_seconds": 64.22222222,
                    "loss_total": 0.891234567,
                    "losses": {"ce": 0.891234567},
                    "grad_norm_mean": 3.4567891,
                    "dev": {
                        "macro_f1": 0.5333333333333333,
                        "micro_f1": 0.612,
                        "accuracy": 0.612,
                        "ECE": 0.022222222,
                        "NLL": 0.988888888,
                        "Brier": 0.511111111,
                    },
                }
            ],
        },
        metrics={
            "test": {
                "macro_f1": 0.544444444,
                "micro_f1": 0.621,
                "accuracy": 0.621,
                "per_class_f1": [0.4, 0.61, 0.62],
                "confusion_matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "ECE": 0.033333333,
                "NLL": 0.977777777,
                "Brier": 0.522222222,
                "teacher_student_analysis": {
                    "top1_agreement": 0.712345678,
                    "teacher_student_kl": 0.123456789,
                    "teacher_correct_student_wrong": 44,
                    "teacher_wrong_student_correct": 31,
                    "error_copying": 0.456789123,
                },
            }
        },
    )

    path = tmp_path / "run_metadata.json"
    write_run_metadata(meta, path)

    text = path.read_text()
    payload = json.loads(text)

    assert payload["run"]["stage"] == "student"
    assert payload["run"]["condition"] == "ce_only"
    assert payload["model"]["student_checkpoint"] == "huawei-noah/TinyBERT_General_4L_312D"
    assert "teacher_checkpoint" not in payload["model"]
    assert payload["optimization"]["precision"] == "bf16"
    assert payload["training"]["history"][0]["losses"] == {"ce": 0.89123}
    assert payload["training"]["history"][0]["loss_total"] == 0.89123
    assert payload["metrics"]["test"]["teacher_student_analysis"] == {
        "top1_agreement": 0.71235,
        "teacher_student_kl": 0.12346,
        "teacher_correct_student_wrong": 44,
        "teacher_wrong_student_correct": 31,
        "error_copying": 0.45679,
    }
    assert "raw_loss" not in text


def test_student_run_id_includes_condition():
    run_id = make_run_id("student", "tweet_eval-sentiment", "ce_only")

    assert run_id.startswith("student-ce_only-tweet_eval-sentiment-")


def test_student_schema_v2_round_trips_hidden_losses(tmp_path):
    meta = RunMetadata(
        schema_version="2",
        run={"run_id": "student-kd_hidden-test", "stage": "student", "condition": "kd_hidden"},
        dataset={"name": "x", "config": None, "num_labels": 2, "label_names": ["a", "b"], "splits": {}},
        model={
            "student_checkpoint": "student",
            "tokenizer": "tokenizer",
            "teacher_checkpoint": "teacher",
            "parameter_count": 15312771,
            "projection_parameter_count": 961536,
        },
        optimization={},
        checkpoint_selection={},
        reproducibility={},
        environment={},
        training={
            "history": [
                {
                    "epoch": 0,
                    "global_step": 1,
                    "epoch_time_seconds": 1.0,
                    "loss_total": 1.75,
                    "losses": {"ce": 0.5, "hidden": 1.25},
                    "grad_norm_mean": 2.0,
                    "dev": {},
                }
            ]
        },
    )

    path = tmp_path / "run_metadata.json"
    write_run_metadata(meta, path)
    payload = json.loads(path.read_text())

    assert payload["training"]["history"][0]["losses"] == {"ce": 0.5, "hidden": 1.25}
    assert payload["model"]["projection_parameter_count"] == 961536
