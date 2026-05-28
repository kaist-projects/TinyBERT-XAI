import pytest
import numpy as np
import sklearn.metrics as skm

from tinybert_xai.eval.metrics import _calibration_metrics


def test_calibration_metrics_match_expected_nll_and_brier():
    probs = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
        ]
    )
    labels = np.array([0, 1, 2])

    metrics = _calibration_metrics(probs, labels, num_classes=3)

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    expected_brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))

    assert metrics["NLL"] == pytest.approx(skm.log_loss(labels, probs, labels=[0, 1, 2]))
    assert metrics["Brier"] == pytest.approx(expected_brier)


def test_ece_includes_confidence_one_in_final_bin():
    probs = np.array(
        [
            [1.0, 0.0],
            [0.6, 0.4],
        ]
    )
    labels = np.array([1, 0])

    metrics = _calibration_metrics(probs, labels, num_classes=2, n_bins=10)

    # sample 1: final bin, confidence=1.0, incorrect -> contribution 0.5
    # sample 2: [0.6, 0.7) bin, confidence=0.6, correct -> contribution 0.2
    assert metrics["ECE"] == pytest.approx(0.7)
