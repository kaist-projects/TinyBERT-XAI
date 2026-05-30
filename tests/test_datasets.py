import pytest
from datasets import ClassLabel, Dataset, Features, Value

from tinybert_xai import DATASET_ANLI, DATASET_IMDB, DATASET_TWEETEVAL_SENTIMENT, dataset_by_name
from tinybert_xai.datasets import DatasetSpec, _tokenize_split, stratified_dev_partition

MAX_LEN = 8


class FakeTokenizer:
    """Records arity and emits BERT-shaped fields; segment B is 1 only for pairs."""

    def __call__(self, *texts, padding, truncation, max_length):
        n = len(texts[0])
        is_pair = len(texts) == 2
        half = max_length // 2
        return {
            "input_ids": [[1] * max_length for _ in range(n)],
            "attention_mask": [[1] * max_length for _ in range(n)],
            "token_type_ids": [[0] * half + [int(is_pair)] * (max_length - half) for _ in range(n)],
        }


def _single_spec() -> DatasetSpec:
    return DatasetSpec("t", "f", "p", None, 2, ["a", "b"], text_keys=("text",))


def _pair_spec() -> DatasetSpec:
    return DatasetSpec("t", "f", "p", None, 2, ["a", "b"], text_keys=("premise", "hypothesis"))


def test_single_text_tokenization_has_zero_token_type_ids():
    ds = Dataset.from_dict({"text": ["hi", "yo"], "label": [0, 1]})
    out = _tokenize_split(ds, _single_spec(), FakeTokenizer(), MAX_LEN)
    assert "token_type_ids" in out.column_names
    assert out["token_type_ids"].sum().item() == 0
    assert out["labels"].tolist() == [0, 1]


def test_sentence_pair_tokenization_marks_segment_b():
    ds = Dataset.from_dict({"premise": ["a", "c"], "hypothesis": ["b", "d"], "label": [1, 0]})
    out = _tokenize_split(ds, _pair_spec(), FakeTokenizer(), MAX_LEN)
    assert out["token_type_ids"].sum().item() > 0  # segment B present


def test_stratified_dev_partition_is_disjoint_deterministic_and_stratified():
    n = 200
    features = Features({"idx": Value("int64"), "label": ClassLabel(num_classes=2)})
    ds = Dataset.from_dict({"idx": list(range(n)), "label": [i % 2 for i in range(n)]}, features=features)

    train, dev = stratified_dev_partition(ds, 0.25, "label")
    train_idx, dev_idx = set(train["idx"]), set(dev["idx"])

    assert train_idx.isdisjoint(dev_idx)
    assert train_idx | dev_idx == set(range(n))
    assert len(dev_idx) == 50
    assert sum(dev["label"]) == 25  # 50/50 classes preserved

    train2, dev2 = stratified_dev_partition(ds, 0.25, "label")
    assert set(dev2["idx"]) == dev_idx  # reproducible


def test_registry_round_trips_known_datasets():
    assert dataset_by_name("imdb") is DATASET_IMDB
    assert dataset_by_name("anli") is DATASET_ANLI
    assert dataset_by_name("tweet_eval-sentiment") is DATASET_TWEETEVAL_SENTIMENT
    with pytest.raises(KeyError):
        dataset_by_name("does-not-exist")


def test_spec_input_type_and_split_scheme():
    assert DATASET_IMDB.input_type == "single_text"
    assert DATASET_IMDB.split_scheme == "stratified_dev_0.1_seed42"
    assert DATASET_ANLI.input_type == "sentence_pair"
    assert DATASET_ANLI.split_scheme == "official"
