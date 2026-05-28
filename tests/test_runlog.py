import json

from tinybert_xai.runlog import RunMetadata, write_run_metadata


def test_schema_v2_rounding_and_teacher_fields(tmp_path):
    meta = RunMetadata(
        schema_version="2",
        run={
            "run_id": "teacher-test",
            "stage": "teacher",
            "condition": None,
            "git_commit": "abc1234",
            "seed": 42,
        },
        dataset={
            "name": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "family": "sentiment",
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
            "warn_only": True,
            "cublas_workspace_config": ":4096:8",
            "shuffle_seed_scheme": "seed + epoch",
        },
        environment={
            "device": "cuda:0",
            "gpu_model": "NVIDIA GeForce RTX 3090",
            "gpu_memory_total_mb": 24118.25,
            "cuda_available": True,
            "torch_cuda": "12.4",
            "package_versions": {
                "torch": "2.5.1+cu124",
                "transformers": "4.49.0",
                "datasets": "3.6.0",
                "tokenizers": "0.21.4",
                "numpy": "1.26.4",
                "sklearn": "1.5.2",
                "python": "3.12.13",
            },
        },
        training={
            "epochs_completed": 1,
            "best_dev_macro_f1": 0.7201311861596489,
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
        efficiency={
            "latency_p50_ms": 49.69830322265625,
            "latency_p95_ms": 49.90097770690918,
            "throughput_samples_per_sec": 643.839053437915,
            "model_size_mb": 417.7276916503906,
            "parameter_count": 109484547,
            "gpu_memory_mb": 990.81689453125,
        },
        metric_definitions={"confusion_matrix": "rows=true, cols=pred"},
    )

    path = tmp_path / "run_metadata.json"
    write_run_metadata(meta, path)

    text = path.read_text()
    payload = json.loads(text)

    assert payload["schema_version"] == "2"
    assert payload["optimization"]["learning_rate"] == 2e-5
    assert payload["optimization"]["eps"] == 1e-8
    assert payload["training"]["best_dev_macro_f1"] == 0.7201
    assert payload["training"]["train_time_seconds"] == 857.0
    assert payload["training"]["history"][0]["loss_total"] == 0.6555
    assert payload["training"]["history"][0]["epoch_time_seconds"] == 273.8
    assert payload["efficiency"]["latency_p50_ms"] == 49.7
    assert payload["efficiency"]["throughput_samples_per_sec"] == 643.8
    assert payload["metrics"]["dev"]["confusion_matrix"] == [[215, 74, 23], [104, 581, 184]]

    assert "train_raw_loss_ce" not in text
    assert "teacher_student_kl" not in text
    assert "[215, 74, 23]" in text
