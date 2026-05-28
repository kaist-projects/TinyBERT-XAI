from tinybert_xai.checkpoints import (
    load_state_dict,
    results_dir,
    save_state_dict,
    student_dir,
    teacher_dir,
)
from tinybert_xai.config import Config
from tinybert_xai.conditions import (
    ALL_CONDITIONS,
    CE_ONLY,
    KD_ATTN,
    KD_FULL,
    KD_HIDDEN,
    KD_HIDDEN_ATTN,
    KD_LOGIT,
    KD_LOGIT_ATTN,
    KD_LOGIT_HIDDEN,
    ConditionSpec,
)
from tinybert_xai.datasets import (
    DATASET_TWEETEVAL_SENTIMENT,
    DatasetSpec,
    SentimentLabel,
    build_loader,
    encode_batch,
    load_split,
)
from tinybert_xai.earlystop import EarlyStopper
from tinybert_xai.eval import (
    EvaluationResult,
    evaluate,
)
from tinybert_xai.kdpair import KDPair, KDOutputs
from tinybert_xai.losses import compute_student_losses
from tinybert_xai.models import load_tokenizer, load_classifier
from tinybert_xai.runlog import (
    RunMetadata,
    TrainEpochEntry,
    collect_hardware,
    make_run_id,
    write_run_metadata,
)
from tinybert_xai.teacher import (
    TeacherData,
    TeacherEvaluationResult,
    TeacherEpochStats,
    TeacherModel,
    TeacherTrainingResult,
    configure_reproducibility,
    evaluate_saved_teacher,
    fine_tune_teacher,
    load_teacher_data,
    prepare_teacher_model,
    resolve_device,
    save_teacher_evaluation_result,
    save_teacher_training_result,
    start_teacher_metadata,
    train_teacher_epoch,
)
from tinybert_xai.student import (
    StudentData,
    StudentEpochStats,
    StudentEvaluationResult,
    StudentModel,
    StudentTrainingResult,
    evaluate_saved_student,
    fine_tune_student,
    load_student_data,
    prepare_student_model,
    save_student_evaluation_result,
    save_student_training_result,
    start_student_metadata,
    train_student_epoch,
)
from tinybert_xai.utils import (
    clone_state_dict_cpu,
    count_params,
    move_batch_to_device,
    training_autocast,
)

__all__ = [
    # config
    "Config",
    # datasets
    "DatasetSpec",
    "SentimentLabel",
    "DATASET_TWEETEVAL_SENTIMENT",
    "build_loader",
    "encode_batch",
    "load_split",
    # conditions
    "ConditionSpec",
    "CE_ONLY",
    "KD_LOGIT",
    "KD_HIDDEN",
    "KD_ATTN",
    "KD_LOGIT_HIDDEN",
    "KD_LOGIT_ATTN",
    "KD_HIDDEN_ATTN",
    "KD_FULL",
    "ALL_CONDITIONS",
    # eval
    "EvaluationResult",
    "evaluate",
    # models / kdpair
    "KDPair",
    "KDOutputs",
    "compute_student_losses",
    "load_tokenizer",
    "load_classifier",
    # runlog
    "RunMetadata",
    "TrainEpochEntry",
    "make_run_id",
    "collect_hardware",
    "write_run_metadata",
    # teacher pipeline
    "TeacherData",
    "TeacherEvaluationResult",
    "TeacherEpochStats",
    "TeacherModel",
    "TeacherTrainingResult",
    "configure_reproducibility",
    "evaluate_saved_teacher",
    "fine_tune_teacher",
    "load_teacher_data",
    "prepare_teacher_model",
    "resolve_device",
    "save_teacher_evaluation_result",
    "save_teacher_training_result",
    "start_teacher_metadata",
    "train_teacher_epoch",
    # student pipeline
    "StudentData",
    "StudentEpochStats",
    "StudentEvaluationResult",
    "StudentModel",
    "StudentTrainingResult",
    "evaluate_saved_student",
    "fine_tune_student",
    "load_student_data",
    "prepare_student_model",
    "save_student_evaluation_result",
    "save_student_training_result",
    "start_student_metadata",
    "train_student_epoch",
    # checkpoints
    "teacher_dir",
    "student_dir",
    "results_dir",
    "save_state_dict",
    "load_state_dict",
    # earlystop
    "EarlyStopper",
    # utils
    "count_params",
    "move_batch_to_device",
    "training_autocast",
    "clone_state_dict_cpu",
]
