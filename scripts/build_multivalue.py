"""scripts/build_multivalue.py — Generate the Multi-VALUE SAE-vs-dialect dataset.

Multi-VALUE ships as a transformation toolkit, not a labeled dataset. This script
builds a binary detection task: half of a Standard American English (SAE) base
corpus is left as-is (label ``sae``), the other half is rewritten into a sampled
English dialect with the Multi-VALUE toolkit (label ``dialect``). The result is a
seed-42 stratified 80/10/10 CSV that ``DATASET_MULTIVALUE`` loads via its
``local_csv`` source.

Usage
-----
    conda activate tinybert-xai
    pip install value-nlp                 # one-time: Multi-VALUE toolkit
    python scripts/build_multivalue.py    # writes data/multivalue/multivalue.csv

This is a slow one-time job: constructing each Multi-VALUE dialect takes ~2 min,
and transforming each sentence ~0.5s, so the default size runs on the order of
tens of minutes. Run it once, like the DynaHate manual-download step.

Output
------
    data/multivalue/multivalue.csv  with columns: text, label, split
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import random

from datasets import load_dataset
from multivalue import Dialects

SEED = 42
BASE_DATASET = ("glue", "sst2")
BASE_TEXT_KEY = "sentence"
# A spread of well-attested dialects with rich rule sets, so transformed text
# differs from SAE often enough for a meaningful detection signal. Kept small
# because constructing each Dialect is expensive (~2 min one-time setup each).
DIALECT_CLASSES = [
    "AppalachianDialect",
    "ChicanoDialect",
    "IndianDialect",
    "JamaicanDialect",
]
OUTPUT_PATH = pathlib.Path("data/multivalue/multivalue.csv")


def main() -> None:
    args = _parse_args()
    rng = random.Random(SEED)

    sentences = load_base_sentences(args.num_samples, rng)
    dialects = build_dialect_pool()
    examples = build_examples(sentences, dialects, rng)
    rows = partition_splits(examples, rng)

    write_csv(OUTPUT_PATH, rows)
    print(f"[OK] wrote {len(rows)} examples to {OUTPUT_PATH}")
    _print_summary(rows)


def load_base_sentences(num_samples: int, rng: random.Random) -> list[str]:
    """Sample distinct, non-trivial SAE sentences from the base corpus."""
    ds = load_dataset(*BASE_DATASET, split="train")
    pool = [s.strip() for s in ds[BASE_TEXT_KEY] if len(s.split()) >= 5]
    rng.shuffle(pool)
    return pool[:num_samples]


def build_dialect_pool() -> list:
    """Instantiate each dialect transformer once (construction is expensive)."""
    return [getattr(Dialects, name)() for name in DIALECT_CLASSES if hasattr(Dialects, name)]


def build_examples(sentences: list[str], dialects: list, rng: random.Random) -> list[dict]:
    """Label half the sentences ``sae`` and rewrite the other half into a dialect.

    Sentences no dialect can transform (the toolkit raises on some inputs) are
    dropped rather than mislabeled, slightly shrinking the ``dialect`` side.
    """
    examples = []
    for i, sentence in enumerate(sentences):
        if i % 2 == 0:
            examples.append({"text": sentence, "label": "sae"})
            continue
        transformed = to_dialect(sentence, dialects, rng)
        if transformed is not None:
            examples.append({"text": transformed, "label": "dialect"})
    return examples


def to_dialect(sentence: str, dialects: list, rng: random.Random) -> str | None:
    """Rewrite a sentence with a sampled dialect; ``None`` if none apply or all fail.

    The Multi-VALUE toolkit raises on some inputs (e.g. spaCy/Stanza tokenization
    mismatches), so each attempt is guarded and we fall through to the next dialect.
    """
    for dialect in rng.sample(dialects, k=len(dialects)):
        try:
            transformed = dialect.transform(sentence)
        except Exception:
            continue
        if transformed and transformed.strip() != sentence:
            return transformed.strip()
    return None


def partition_splits(examples: list[dict], rng: random.Random) -> list[dict]:
    """Assign a seed-42 stratified 80/10/10 train/dev/test split per label."""
    rows = []
    for label in ("sae", "dialect"):
        items = [e for e in examples if e["label"] == label]
        rng.shuffle(items)
        n_test = len(items) // 10
        n_dev = len(items) // 10
        for split, chunk in _named_chunks(items, n_dev, n_test):
            rows.extend({**item, "split": split} for item in chunk)
    return rows


def _named_chunks(items: list[dict], n_dev: int, n_test: int) -> list[tuple[str, list[dict]]]:
    test, dev, train = items[:n_test], items[n_test : n_test + n_dev], items[n_test + n_dev :]
    return [("test", test), ("dev", dev), ("train", train)]


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict]) -> None:
    for split in ("train", "dev", "test"):
        counts = {label: sum(r["split"] == split and r["label"] == label for r in rows) for label in ("sae", "dialect")}
        print(f"  {split:5s} sae={counts['sae']} dialect={counts['dialect']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10_000,
        help="total base sentences to label (half SAE, half dialect); default 10000",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
