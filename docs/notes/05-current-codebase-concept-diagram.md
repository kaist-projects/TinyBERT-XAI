# Current Codebase Concept Diagram

This document describes what the current Iteration 0 codebase does. It is a
foundation and smoke-test layer: it proves that the project can load a teacher,
a student, a tokenizer, and a small dataset batch, then run both models on the
same inputs and inspect their outputs.

It does not yet train anything.

## High-Level Purpose

The current code answers one question:

> Can we wire the TinyBERT-XAI dependencies together correctly before building
> teacher fine-tuning, student training, and KD losses?

The answer is checked by `scripts/00_smoke_test.py`.

## Module Relationship Diagram

```mermaid
flowchart TD
    Smoke["scripts/00_smoke_test.py<br/>wiring / executable entry point"]

    Config["config.py<br/>Config dataclass<br/>seed, checkpoints, max sequence length"]
    Utils["utils.py<br/>set_seed, get_device, count_params"]
    Datasets["datasets.py<br/>DatasetSpec<br/>DATASET_TWEETEVAL_SENTIMENT"]
    RowType["TweetEvalSentimentData<br/>text: str<br/>label: SentimentLabel<br/>from_row"]
    Models["models.py<br/>load_tokenizer<br/>load_classifier"]
    Loader["data.py<br/>DatasetLoader<br/>loads DatasetDict"]
    Encoder["data.py<br/>BatchEncoder<br/>raw rows -> BatchEncoding"]
    KDPair["kdpair.py<br/>KDPair.forward"]
    KDOutputs["kdpair.py<br/>KDOutputs<br/>shape checks + summary"]

    HFTokenizer["HuggingFace tokenizer<br/>bert-base-uncased"]
    HFTeacher["HuggingFace teacher model<br/>bert-base-uncased"]
    HFStudent["HuggingFace student model<br/>TinyBERT_General_4L_312D"]
    HFDataset["HuggingFace dataset<br/>cardiffnlp/tweet_eval sentiment"]
    Batch["BatchEncoding<br/>input_ids, attention_mask,<br/>token_type_ids if present, labels"]
    Outputs["Teacher + student outputs<br/>logits, hidden_states, attentions"]

    Smoke --> Config
    Smoke --> Utils
    Smoke --> Datasets
    Smoke --> Models
    Smoke --> Loader
    Smoke --> Encoder
    Smoke --> KDPair

    Datasets --> HFDataset
    Models --> HFTokenizer
    Models --> HFTeacher
    Models --> HFStudent

    Loader --> Datasets
    Encoder --> Datasets
    Datasets --> RowType
    Encoder --> RowType
    Encoder --> HFTokenizer
    Loader --> HFDataset
    HFDataset --> Encoder
    Encoder --> Batch

    KDPair --> HFTeacher
    KDPair --> HFStudent
    KDPair --> Batch
    KDPair --> Outputs
    KDPair --> KDOutputs

    KDOutputs --> Outputs
```

## Dependency Direction

The code is intentionally simple and dependency-injection oriented.

The important rule is:

> Construction happens in the script. Core modules receive dependencies as
> arguments instead of creating them secretly.

That means:

- `scripts/00_smoke_test.py` is responsible for wiring the pieces together.
- `models.py` loads models and tokenizers, but does not know about datasets.
- `data.py` loads and tokenizes batches, but does not know about teacher or student models.
- `datasets.py` stores dataset metadata, but does not load models or run training.
- `KDPair` receives already-created teacher, student, and tokenizer objects.
- `KDPair.forward(...)` only runs both models on the same batch.

This keeps responsibilities separated before the training loop becomes more
complex in later iterations.

## Execution Flow

```mermaid
sequenceDiagram
    participant Script as 00_smoke_test.py
    participant Config as Config
    participant Utils as utils.py
    participant Registry as datasets.py
    participant Models as models.py
    participant Loader as DatasetLoader
    participant Encoder as BatchEncoder
    participant Pair as KDPair
    participant Outputs as KDOutputs

    Script->>Config: create Config()
    Script->>Utils: set_seed(cfg.seed)
    Script->>Utils: get_device()
    Script->>Registry: read DATASET_TWEETEVAL_SENTIMENT
    Script->>Models: load_tokenizer(cfg.tokenizer_checkpoint)
    Script->>Models: load_classifier(cfg.teacher_checkpoint, num_labels, device)
    Script->>Models: load_classifier(cfg.student_checkpoint, num_labels, device)
    Script->>Pair: KDPair(teacher, student, tokenizer)
    Script->>Loader: DatasetLoader(spec)
    Script->>Encoder: BatchEncoder(spec, tokenizer, max_length=128)
    Script->>Loader: get_split("train")
    Loader-->>Script: HuggingFace Dataset
    Script->>Encoder: encode(train_ds, batch_size=4)
    Encoder-->>Script: BatchEncoding
    Script->>Pair: forward(batch)
    Pair-->>Script: KDOutputs
    Script->>Outputs: assert_shapes_consistent()
    Script->>Outputs: summary()
    Script->>Utils: count_params(teacher), count_params(student)
```

## File Responsibilities

| File | Responsibility | What It Does Not Do |
|---|---|---|
| `src/tinybert_xai/config.py` | Holds basic project settings in `Config`. | Does not load files, models, or datasets. |
| `src/tinybert_xai/datasets.py` | Defines `DatasetSpec`, `SentimentLabel`, `TweetEvalSentimentData`, and `DATASET_TWEETEVAL_SENTIMENT`. | Does not call HuggingFace loading directly. |
| `src/tinybert_xai/models.py` | Loads tokenizer and sequence-classification models. | Does not decide which dataset to use. |
| `src/tinybert_xai/data.py` | `DatasetLoader` loads all HuggingFace splits into a `DatasetDict`; `BatchEncoder` parses rows through `spec.data_cls.from_row`, tokenizes text, adds labels, and returns `BatchEncoding`. | Does not train or evaluate. |
| `src/tinybert_xai/kdpair.py` | Runs teacher and student forward passes on the same batch. | Does not construct models or compute losses. |
| `src/tinybert_xai/utils.py` | Provides generic helpers for seed, device, and parameter counts. | Does not contain project-specific training logic. |
| `scripts/00_smoke_test.py` | Wires all pieces together and checks output shapes. | Does not perform optimization or save checkpoints. |

## Current Data Flow

```mermaid
flowchart LR
    RawDataset["Raw TweetEval rows"]
    ParsedRows["TweetEvalSentimentData rows<br/>text + SentimentLabel"]
    Tokenizer["bert-base-uncased tokenizer"]
    Batch["BatchEncoding<br/>input_ids<br/>attention_mask<br/>labels"]
    Teacher["Teacher<br/>bert-base-uncased classifier"]
    Student["Student<br/>TinyBERT classifier"]
    TOut["Teacher outputs<br/>logits<br/>13 hidden states<br/>12 attentions"]
    SOut["Student outputs<br/>logits<br/>5 hidden states<br/>4 attentions"]
    Check["Shape consistency check"]

    RawDataset --> ParsedRows
    ParsedRows --> Tokenizer
    Tokenizer --> Batch
    Batch --> Teacher
    Batch --> Student
    Teacher --> TOut
    Student --> SOut
    TOut --> Check
    SOut --> Check
```

## Expected Model Output Shapes

For the current smoke test settings:

- Batch size: `4`
- Max sequence length: `128`
- Number of labels: `3` for TweetEval sentiment

Expected teacher outputs:

- Logits: `[4, 3]`
- Hidden states: `13` tensors
  - embedding output plus 12 BERT layers
  - each tensor shaped `[4, 128, 768]`
- Attentions: `12` tensors
  - one per BERT layer
  - each tensor shaped `[4, 12, 128, 128]`

Expected student outputs:

- Logits: `[4, 3]`
- Hidden states: `5` tensors
  - embedding output plus 4 TinyBERT layers
  - each tensor shaped `[4, 128, 312]`
- Attentions: `4` tensors
  - one per TinyBERT layer
  - each tensor shaped `[4, 12, 128, 128]`

These shape differences explain why later iterations need hidden-state
projection layers for KD:

```text
student hidden size: 312
teacher hidden size: 768
needed projection: 312 -> 768
```

## What Exists Now vs. Later

```mermaid
flowchart TD
    Now["Current Iteration 0"]
    Later["Later Iterations"]

    Now --> N1["Load config"]
    Now --> N2["Load dataset spec"]
    Now --> N3["Load tokenizer"]
    Now --> N4["Load teacher + student"]
    Now --> N5["Load one tokenized batch"]
    Now --> N6["Forward both models"]
    Now --> N7["Check output shapes"]

    Later --> L1["Teacher fine-tuning"]
    Later --> L2["Student training loop"]
    Later --> L3["CE loss"]
    Later --> L4["Logit KD loss"]
    Later --> L5["Hidden KD loss"]
    Later --> L6["Attention KD loss"]
    Later --> L7["8-condition factorial sweep"]
    Later --> L8["Evaluation and analysis"]
```

The current codebase is therefore best understood as the dependency skeleton
that later training code will reuse.

## Current Dataset Assumption

`DatasetSpec` intentionally stays small in Iteration 0. It stores:

- HuggingFace dataset path
- optional HuggingFace config name
- row data class

It does not store `text_column` or `label_column`.

The current smoke-test batch loader parses raw rows through:

```python
spec.data_cls.from_row(row)
```

For the pilot dataset, `spec.data_cls` is `TweetEvalSentimentData`:

```python
@dataclass(frozen=True)
class TweetEvalSentimentData:
    text: str
    label: SentimentLabel

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "TweetEvalSentimentData":
        ...
```

So raw HuggingFace column names are isolated inside the dataset-specific row
class instead of being stored in generic dataset metadata.
