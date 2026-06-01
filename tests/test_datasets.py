import pytest
from datasets import ClassLabel, Dataset, Features, Value

from tinybert_xai import (
    DATASET_ANLI,
    DATASET_DAVIDSON,
    DATASET_DYNAHATE,
    DATASET_FEVER,
    DATASET_HATEVAL,
    DATASET_IMDB,
    DATASET_TWEETEVAL_SENTIMENT,
    DATASET_VARDIAL,
    LocalCsvSource,
    dataset_by_name,
    load_split,
)
from tinybert_xai.data.datasets import (
    DatasetSpec,
    _apply_label_map,
    _maybe_subsample_train,
    _tokenize_split,
    stratified_split_partition,
)

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


def _labelled_dataset(n: int) -> Dataset:
    features = Features({"idx": Value("int64"), "label": ClassLabel(num_classes=2)})
    return Dataset.from_dict({"idx": list(range(n)), "label": [i % 2 for i in range(n)]}, features=features)


def test_dev_only_partition_is_disjoint_deterministic_and_stratified():
    ds = _labelled_dataset(200)

    parts = stratified_split_partition(ds, dev_split=0.25, test_split=None, label_key="label")
    assert "test" not in parts
    train_idx, dev_idx = set(parts["train"]["idx"]), set(parts["validation"]["idx"])

    assert train_idx.isdisjoint(dev_idx)
    assert train_idx | dev_idx == set(range(200))
    assert len(dev_idx) == 50
    assert sum(parts["validation"]["label"]) == 25  # 50/50 classes preserved

    again = stratified_split_partition(ds, dev_split=0.25, test_split=None, label_key="label")
    assert set(again["validation"]["idx"]) == dev_idx  # reproducible


def test_three_way_partition_sizes_disjoint_and_stratified():
    ds = _labelled_dataset(200)

    parts = stratified_split_partition(ds, dev_split=0.1, test_split=0.1, label_key="label")
    train_idx = set(parts["train"]["idx"])
    dev_idx = set(parts["validation"]["idx"])
    test_idx = set(parts["test"]["idx"])

    assert train_idx.isdisjoint(dev_idx) and train_idx.isdisjoint(test_idx) and dev_idx.isdisjoint(test_idx)
    assert train_idx | dev_idx | test_idx == set(range(200))
    assert (len(train_idx), len(dev_idx), len(test_idx)) == (160, 20, 20)  # 80/10/10
    assert sum(parts["validation"]["label"]) == 10 and sum(parts["test"]["label"]) == 10

    again = stratified_split_partition(ds, dev_split=0.1, test_split=0.1, label_key="label")
    assert set(again["test"]["idx"]) == test_idx  # reproducible


def test_subsample_caps_train_pool_stratified_and_deterministic():
    ds = _labelled_dataset(200)
    spec = DatasetSpec("t", "f", "p", None, 2, ["a", "b"], text_keys=("text",), train_subsample=50)

    out = _maybe_subsample_train(ds, spec)
    assert len(out) == 50
    assert sum(out["label"]) == 25  # 50/50 classes preserved

    again = _maybe_subsample_train(ds, spec)
    assert set(again["idx"]) == set(out["idx"])  # reproducible


def test_subsample_is_noop_when_pool_within_cap():
    ds = _labelled_dataset(30)
    spec = DatasetSpec("t", "f", "p", None, 2, ["a", "b"], text_keys=("text",), train_subsample=50)
    assert _maybe_subsample_train(ds, spec) is ds


def test_subsample_is_noop_when_unset():
    ds = _labelled_dataset(30)
    spec = DatasetSpec("t", "f", "p", None, 2, ["a", "b"], text_keys=("text",))
    assert _maybe_subsample_train(ds, spec) is ds


def test_hf_label_map_maps_string_labels_to_ints():
    ds = Dataset.from_dict({"sentence": ["a", "b"], "label": ["ZH", "BE"]})
    spec = DatasetSpec(
        "t", "f", "p", None, 4, ["BE", "BS", "LU", "ZH"],
        text_keys=("sentence",),
        hf_label_map={"BE": 0, "BS": 1, "LU": 2, "ZH": 3},
    )
    out = _apply_label_map(ds, spec)
    assert out["label"] == [3, 0]


def test_hf_label_map_is_noop_when_unset():
    ds = Dataset.from_dict({"text": ["a"], "label": [1]})
    spec = DatasetSpec("t", "f", "p", None, 2, ["a", "b"], text_keys=("text",))
    assert _apply_label_map(ds, spec) is ds


def test_local_csv_split_filters_by_split_column_and_maps_labels(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("text,label,split\na,hate,train\nb,nothate,train\nc,hate,dev\nd,nothate,test\n")
    spec = DatasetSpec(
        name="x",
        family="hate",
        hf_path=None,
        hf_config=None,
        num_labels=2,
        label_names=["nothate", "hate"],
        text_keys=("text",),
        label_key="label",
        split_sources={"train": "train", "validation": "dev", "test": "test"},
        local_csv=LocalCsvSource(path=str(csv), split_column="split", label_map={"nothate": 0, "hate": 1}),
    )

    train = load_split(spec, "train")
    assert sorted(train["text"]) == ["a", "b"]
    assert set(train["label"]) == {0, 1}  # strings mapped to ints

    dev = load_split(spec, "validation")
    assert dev["text"] == ["c"] and dev["label"] == [1]
    test = load_split(spec, "test")
    assert test["text"] == ["d"] and test["label"] == [0]


@pytest.mark.parametrize("spec", [DATASET_DYNAHATE])
def test_local_csv_missing_file_raises_helpful_error(spec):
    with pytest.raises(FileNotFoundError, match="Download it"):
        load_split(spec, "train")


def test_registry_round_trips_known_datasets():
    assert dataset_by_name("imdb") is DATASET_IMDB
    assert dataset_by_name("anli") is DATASET_ANLI
    assert dataset_by_name("davidson") is DATASET_DAVIDSON
    assert dataset_by_name("dynahate") is DATASET_DYNAHATE
    assert dataset_by_name("tweet_eval-sentiment") is DATASET_TWEETEVAL_SENTIMENT
    assert dataset_by_name("fever") is DATASET_FEVER
    assert dataset_by_name("hateval") is DATASET_HATEVAL
    assert dataset_by_name("vardial") is DATASET_VARDIAL
    with pytest.raises(KeyError):
        dataset_by_name("does-not-exist")


def test_spec_input_type_and_split_scheme():
    assert DATASET_IMDB.input_type == "single_text"
    assert DATASET_IMDB.split_scheme == "stratified_dev_0.1_seed42"
    assert DATASET_ANLI.input_type == "sentence_pair"
    assert DATASET_ANLI.split_scheme == "official"
    assert DATASET_DAVIDSON.split_scheme == "stratified_dev_0.1_test_0.1_seed42"
    assert DATASET_DYNAHATE.input_type == "single_text"
    assert DATASET_DYNAHATE.split_scheme == "official_csv"
    assert DATASET_FEVER.input_type == "sentence_pair"
    assert DATASET_FEVER.split_scheme == "stratified_test_0.1_sub50000_seed42"
    assert DATASET_HATEVAL.input_type == "single_text"
    assert DATASET_HATEVAL.split_scheme == "official"
    assert DATASET_VARDIAL.input_type == "single_text"
    assert DATASET_VARDIAL.split_scheme == "stratified_dev_0.1_test_0.1_seed42"
