"""Internal model loading — not part of the public API.

KDPair (kdpair.py) is the user-facing entry point.
"""

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def _load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load the shared tokenizer. Per design doc §2, teacher and student share one tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)


def _load_classifier(
    checkpoint: str,
    num_labels: int,
    device: str,
) -> PreTrainedModel:
    """Load a BERT-family model for sequence classification with KD hooks enabled.

    Returns model on `device` in eval() mode.
    Callers that need train() (iters 1+) switch mode explicitly.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        output_hidden_states=True,
        output_attentions=True,
        attn_implementation="eager",
    )
    return model.to(device).eval()
