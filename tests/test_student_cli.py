import sys
from pathlib import Path

from tinybert_xai import format_student_eval_summary
from tinybert_xai.eval import EvaluationResult, TeacherStudentAnalysis
from tinybert_xai.pipeline.student import StudentEvaluationResult

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _student_cli import add_signal_flags, condition_from_args  # noqa: E402


def _eval_result(macro_f1: float) -> EvaluationResult:
    return EvaluationResult(
        macro_f1=macro_f1,
        micro_f1=0.70,
        accuracy=0.70,
        per_class_f1=[0.6, 0.7, 0.8],
        confusion_matrix=[[1, 0], [0, 1]],
        ECE=0.05,
        NLL=0.6,
        Brier=0.4,
    )


def _student_result(test_macro_f1: float, analysis=None) -> StudentEvaluationResult:
    return StudentEvaluationResult(
        metadata_path=Path("run_metadata.json"),
        dev_size=100,
        test_size=200,
        dev_result=_eval_result(0.68),
        test_result=_eval_result(test_macro_f1),
        test_metrics={},
        teacher_student_analysis=analysis,
    )


def test_summary_passes_dod_and_omits_analysis_when_absent():
    summary = format_student_eval_summary(_student_result(0.64))
    assert "test macro-F1 : 0.6400" in summary
    assert "DoD check (test macro-F1 >= 0.33): PASS" in summary
    assert "top1 agreement" not in summary


def test_summary_fails_dod_and_includes_analysis_when_present():
    analysis = TeacherStudentAnalysis(
        top1_agreement=0.77,
        teacher_student_kl=0.16,
        teacher_correct_student_wrong=10,
        teacher_wrong_student_correct=5,
        error_copying=0.9,
    )
    summary = format_student_eval_summary(_student_result(0.30, analysis))
    assert "DoD check (test macro-F1 >= 0.33): FAIL" in summary
    assert "top1 agreement: 0.7700" in summary
    assert "teacher->student KL: 0.1600" in summary


def test_condition_from_args_maps_flags_to_condition_name():
    import argparse

    parser = argparse.ArgumentParser()
    add_signal_flags(parser)
    assert condition_from_args(parser.parse_args([])).name == "ce_only"
    assert condition_from_args(parser.parse_args(["--logit", "--attention"])).name == "kd_logit_attn"
    assert condition_from_args(parser.parse_args(["--logit", "--hidden", "--attention"])).name == "kd_full"
