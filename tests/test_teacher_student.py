import math

import pytest
import torch

from tinybert_xai.eval.teacher_student import compute_teacher_student_analysis


def test_compute_teacher_student_analysis_counts_and_rates():
    teacher_probs = torch.tensor(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.6, 0.2],
            [0.6, 0.3, 0.1],
        ]
    )
    student_probs = torch.tensor(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.3, 0.5],
            [0.1, 0.2, 0.7],
            [0.5, 0.4, 0.1],
        ]
    )
    labels = torch.tensor([0, 1, 2, 1])

    result = compute_teacher_student_analysis(teacher_probs, student_probs, labels)

    expected_kl = sum(
        sum(float(t) * math.log(float(t) / float(s)) for t, s in zip(t_row, s_row))
        for t_row, s_row in zip(teacher_probs, student_probs)
    ) / 4
    assert result.top1_agreement == pytest.approx(0.5)
    assert result.teacher_student_kl == pytest.approx(expected_kl)
    assert result.teacher_correct_student_wrong == 1
    assert result.teacher_wrong_student_correct == 1
    assert result.error_copying == pytest.approx(1.0)
