import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def load_tokenizer(checkpoint: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(checkpoint)


def load_classifier(
    checkpoint: str,
    num_labels: int,
    device: str,
) -> PreTrainedModel:
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        output_hidden_states=True,
        output_attentions=True,
        attn_implementation="eager",
    )
    return model.to(torch.device(device)).eval()
