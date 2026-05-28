# Training pipeline layering

## Purpose

The teacher pipeline is the first complete training pipeline in the repo. It
sets the shape for later student KD work, so the code should expose clear
contracts between stages without becoming a framework.

The guiding rule: each function should operate at one abstraction level.
Scripts describe the workflow. Package-level pipeline functions describe local
tasks. Private helpers handle tensor, optimizer, and JSON mechanics.

## Pipeline contracts

| Layer | Contract | Inputs | Outputs | Owns | Does not own |
|---|---|---|---|---|---|
| Run setup | Configure reproducibility and device | `Config.seed`, `Config.device` | deterministic runtime, device string | seed, deterministic PyTorch flag, device resolution | data/model construction |
| Run metadata | Start the design-doc metadata skeleton | `Config`, `DatasetSpec`, device | `RunMetadata` | run id, config snapshot, package/hardware fields | training metrics |
| Data | Build encoded loaders | `Config`, `DatasetSpec` | `TeacherData` | tokenizer, train/dev loader construction, split sizes | model or optimizer |
| Model | Prepare the teacher model state | `Config`, `DatasetSpec`, device | `TeacherModel` | classifier load, optimizer construction | data or metrics |
| Epoch training | Train one CE epoch | model, train loader, optimizer, device, seed, epoch, global step | `TeacherEpochStats` | batch device transfer, CE loss, backward, optimizer step, per-epoch aggregates | dev evaluation or checkpoint policy |
| Fine-tuning | Coordinate epochs | config, data, model bundle | `TeacherTrainingResult` | early stopping, dev evaluation, epoch checkpoints, best-state selection | final metadata write |
| Training save | Persist final training artifacts | `RunMetadata`, `TeacherTrainingResult`, spec | best checkpoint path, metadata path | `best.pt`, `run_metadata.json` training block | evaluating test split |
| Evaluation | Evaluate saved teacher | config, spec, device | `TeacherEvaluationResult` | fresh checkpoint load, dev/test metrics, efficiency metrics | metadata mutation |
| Evaluation save | Patch metadata with evaluation | `TeacherEvaluationResult` | updated JSON file | dev/test/efficiency JSON fields | metric computation |

## Implemented shape

High-level scripts now read as workflows:

1. configure reproducibility
2. resolve device
3. start metadata
4. load data
5. prepare model
6. fine-tune
7. save artifacts

The lower-level details moved into `tinybert_xai/teacher.py`:

- dataclass contracts: `TeacherData`, `TeacherModel`, `TeacherEpochStats`,
  `TeacherTrainingResult`, `TeacherEvaluationResult`
- workflow contracts: `load_teacher_data`, `prepare_teacher_model`,
  `fine_tune_teacher`, `evaluate_saved_teacher`
- save contracts: `save_teacher_training_result`,
  `save_teacher_evaluation_result`
- low-level training helper: `train_teacher_epoch`

The generic tensor transfer primitive lives in `tinybert_xai/utils.py` as
`move_batch_to_device`, and evaluation reuses it.

## Boundary decisions

- The teacher CE training loop is extracted because it is now a named local
  task with a clear contract, not because it is a universal trainer.
- Student KD should not be forced through the teacher loop. Iteration 2 can
  introduce a separate student epoch contract, then common structure can be
  extracted only after there are two real callers.
- Scripts still print progress and summaries. Package functions return data
  contracts and perform the actual work.
- Metadata schema remains in `runlog.py`; checkpoint path conventions remain
  in `checkpoints.py`.

## Roadmap impact

Iteration 1 now delivers a layered teacher pipeline rather than a script that
owns every detail. Future iterations should follow the same shape: define the
contract for each new KD step first, then decide whether it belongs in the
script, a teacher/student pipeline module, or a lower-level utility.
