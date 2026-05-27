"""Teacher and student model loaders for sequence classification.

Both functions configure the underlying BERT-family model to expose hidden states and
attention probabilities — the hooks iters 3–5 depend on for KD signal extraction.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_teacher_for_classification(model_name: str, num_labels: int, device: str):
    """Load the teacher (e.g. bert-base-uncased) with KD output hooks enabled.

    Returns (model, tokenizer). Model is on `device` and in eval() mode.
    Iter 1's teacher-fine-tune script will switch to train() explicitly.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # design doc §2: same tokenizer for both models
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        output_hidden_states=True,
        output_attentions=True,
        attn_implementation="eager",
    )
    return model.to(device).eval(), tokenizer


def load_student_for_classification(model_name: str, num_labels: int, device: str):
    """Load the student (e.g. TinyBERT_General_4L_312D) with KD output hooks enabled.

    Returns (model, tokenizer). Model is on `device` and in eval() mode.
    Iter 2's student-training script will switch to train() explicitly.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # design doc §2: same tokenizer for both models
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        output_hidden_states=True,
        output_attentions=True,
        attn_implementation="eager",
    )
    return model.to(device).eval(), tokenizer
