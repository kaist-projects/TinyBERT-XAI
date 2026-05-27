"""KDPair — the primary user-facing interface for teacher/student pair operations.

End-user usage:
    pair = KDPair.for_dataset("tweet_eval/sentiment")
    batch = pair.sample_batch(n=4)
    out   = pair.forward(batch)
    out.summary()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase

from tinybert_xai.models import _load_classifier, _load_tokenizer
from tinybert_xai.data import _load_batch
from tinybert_xai.utils import count_params, get_device, set_seed


# ---------------------------------------------------------------------------
# Dataset registry — one entry today, 9 by iter 7.
# Each entry: (hf_path, hf_config_name, num_labels, label_names, split)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _DatasetSpec:
    hf_path: str
    hf_config: str | None
    num_labels: int
    label_names: list[str]
    text_column: str = "text"
    label_column: str = "label"
    default_split: str = "train"


_DATASET_REGISTRY: dict[str, _DatasetSpec] = {
    "tweet_eval/sentiment": _DatasetSpec(
        hf_path="cardiffnlp/tweet_eval",
        hf_config="sentiment",
        num_labels=3,
        label_names=["negative", "neutral", "positive"],
    ),
}


# ---------------------------------------------------------------------------
# KDOutputs — result of a KDPair.forward() call
# ---------------------------------------------------------------------------

@dataclass
class KDOutputs:
    teacher: Any   # transformers ModelOutput
    student: Any   # transformers ModelOutput
    num_labels: int
    batch_size: int
    seq_len: int
    teacher_params: int = 0
    student_params: int = 0

    def assert_shapes_consistent(self) -> None:
        """Verify logit / hidden-state / attention shapes without hard-coding constants."""
        t, s = self.teacher, self.student
        bs, sl, nl = self.batch_size, self.seq_len, self.num_labels

        # Logits
        assert t.logits.shape == (bs, nl), f"teacher logits {t.logits.shape}"
        assert s.logits.shape == (bs, nl), f"student logits {s.logits.shape}"

        # Hidden states: list length = num_layers + 1 (includes embedding)
        t_layers = len(t.hidden_states) - 1   # e.g. 12
        s_layers = len(s.hidden_states) - 1   # e.g. 4
        t_hidden = t.hidden_states[0].shape[-1]
        s_hidden = s.hidden_states[0].shape[-1]
        for i, h in enumerate(t.hidden_states):
            assert h.shape == (bs, sl, t_hidden), f"teacher h[{i}] {h.shape}"
        for i, h in enumerate(s.hidden_states):
            assert h.shape == (bs, sl, s_hidden), f"student h[{i}] {h.shape}"

        # Attentions: list length = num_layers, shape = (bs, heads, seq, seq)
        assert len(t.attentions) == t_layers, f"teacher attentions len={len(t.attentions)}"
        assert len(s.attentions) == s_layers, f"student attentions len={len(s.attentions)}"
        t_heads = t.attentions[0].shape[1]
        s_heads = s.attentions[0].shape[1]
        for i, a in enumerate(t.attentions):
            assert a.shape == (bs, t_heads, sl, sl), f"teacher a[{i}] {a.shape}"
        for i, a in enumerate(s.attentions):
            assert a.shape == (bs, s_heads, sl, sl), f"student a[{i}] {a.shape}"

    def summary(self) -> str:
        t, s = self.teacher, self.student
        lines = [
            f"  teacher params    : {self.teacher_params / 1e6:.1f}M",
            f"  student params    : {self.student_params / 1e6:.1f}M",
            f"  logits            : teacher {tuple(t.logits.shape)}  student {tuple(s.logits.shape)}",
            f"  hidden_states     : teacher ×{len(t.hidden_states)} ({t.hidden_states[0].shape[-1]}-d)"
            f"  student ×{len(s.hidden_states)} ({s.hidden_states[0].shape[-1]}-d)",
            f"  attentions        : teacher ×{len(t.attentions)}  student ×{len(s.attentions)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# KDPair
# ---------------------------------------------------------------------------

class KDPair:
    """Owns teacher, student, tokenizer, and dataset spec for one classification task."""

    def __init__(
        self,
        teacher: PreTrainedModel,
        student: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        spec: _DatasetSpec,
        device: str,
        max_seq_length: int,
        teacher_checkpoint: str = "",
        student_checkpoint: str = "",
    ) -> None:
        self._teacher = teacher
        self._student = student
        self._tokenizer = tokenizer
        self._spec = spec
        self._device = device
        self._max_seq_length = max_seq_length
        self._teacher_checkpoint = teacher_checkpoint
        self._student_checkpoint = student_checkpoint

    # ------------------------------------------------------------------
    # Public constructor
    # ------------------------------------------------------------------

    @classmethod
    def for_dataset(
        cls,
        dataset: str,
        *,
        teacher_checkpoint: str = "bert-base-uncased",
        student_checkpoint: str = "huawei-noah/TinyBERT_General_4L_312D",
        device: str | None = None,
        seed: int = 42,
        max_seq_length: int = 128,
    ) -> "KDPair":
        """Load teacher + student ready for `dataset`.

        Args:
            dataset: Registry key, e.g. "tweet_eval/sentiment".
            teacher_checkpoint: HF model ID for the teacher (default: bert-base-uncased).
            student_checkpoint: HF model ID for the student (default: TinyBERT 4L).
            device: "cuda" / "cpu". Auto-detected if None.
            seed: RNG seed (design doc mandates 42).
            max_seq_length: Token sequence length cap (design doc mandates 128).
        """
        if dataset not in _DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset {dataset!r}. Registered: {list(_DATASET_REGISTRY)}")

        spec = _DATASET_REGISTRY[dataset]
        resolved_device = device or get_device()
        set_seed(seed)

        tokenizer = _load_tokenizer(teacher_checkpoint)
        teacher = _load_classifier(teacher_checkpoint, spec.num_labels, resolved_device)
        student = _load_classifier(student_checkpoint, spec.num_labels, resolved_device)

        return cls(
            teacher, student, tokenizer, spec, resolved_device, max_seq_length,
            teacher_checkpoint=teacher_checkpoint,
            student_checkpoint=student_checkpoint,
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def sample_batch(self, n: int = 4, split: str | None = None) -> BatchEncoding:
        """Return n examples from the dataset, tokenized and moved to device."""
        chosen_split = split or self._spec.default_split
        encoding = _load_batch(
            self._spec.hf_path,
            self._spec.hf_config,
            self._spec.text_column,
            self._spec.label_column,
            self._tokenizer,
            batch_size=n,
            max_length=self._max_seq_length,
            split=chosen_split,
        )
        return encoding.to(self._device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: BatchEncoding, *, train_mode: bool = False) -> KDOutputs:
        """Run teacher + student forward on `batch`.

        Args:
            batch: Output of sample_batch() or any compatible BatchEncoding on device.
            train_mode: If True, models stay in train() mode (for iters 1+).
                        Default False wraps in torch.no_grad().
        """
        inputs = {k: v for k, v in batch.items() if k != "labels"}

        if train_mode:
            t_out = self._teacher(**inputs)
            s_out = self._student(**inputs)
        else:
            with torch.no_grad():
                t_out = self._teacher(**inputs)
                s_out = self._student(**inputs)

        return KDOutputs(
            teacher=t_out,
            student=s_out,
            num_labels=self._spec.num_labels,
            batch_size=batch["input_ids"].shape[0],
            seq_len=batch["input_ids"].shape[1],
            teacher_params=count_params(self._teacher),
            student_params=count_params(self._student),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def teacher(self) -> PreTrainedModel:
        return self._teacher

    @property
    def student(self) -> PreTrainedModel:
        return self._student

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def num_labels(self) -> int:
        return self._spec.num_labels

    @property
    def label_names(self) -> list[str]:
        return self._spec.label_names

    @property
    def device(self) -> str:
        return self._device

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @property
    def spec(self) -> _DatasetSpec:
        return self._spec

    @property
    def teacher_checkpoint(self) -> str:
        return self._teacher_checkpoint

    @property
    def student_checkpoint(self) -> str:
        return self._student_checkpoint
