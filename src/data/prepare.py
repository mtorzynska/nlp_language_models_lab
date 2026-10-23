#pobranie i podzial zbioru pod dwa zadaia:
#text: recenzja -> sentyment całej recenzji (sentence classification)
#token: każde słowo recenzji -> sentyment zdania, w którym stoi (token classification)
# zeby uruchomic: uv run python -m src.data.prepare [--max-train N]

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value, load_dataset

from src.config import DATASET_ID, DATASET_REVISION, LABEL_NAMES, PROCESSED_DIR, SEED, SPLITS

TEXT_FEATURES = Features({"id": Value("string"), "text": Value("string"), "label": ClassLabel(names=LABEL_NAMES)})
TOKEN_FEATURES = Features({"id": Value("string"), "words": Sequence(Value("string")), "labels": Sequence(ClassLabel(names=LABEL_NAMES))})
_WHITESPACE = re.compile(r"\s+")

Words = tuple[str, ...]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return _WHITESPACE.sub(" ", text).strip()


def load_raw(level: str) -> DatasetDict:
    # level is sentence or text (ie. review)
    return load_dataset(DATASET_ID, data_dir=f"all_{level}", revision=DATASET_REVISION)


def clean_texts(raw: DatasetDict) -> DatasetDict:
    #normalize, drop empty/duplicate, remove reviews from val/test that were in previous splits
    seen: set[str] = set()
    cleaned = {}
    for split in SPLITS:
        rows = {"id": [], "text": [], "label": []}
        for i, (text, label) in enumerate(zip(raw[split]["text"], raw[split]["target"])):
            text = normalize_text(text)
            if not text or text in seen:
                continue
            seen.add(text)
            rows["id"].append(f"{split}-{i:05d}")
            rows["text"].append(text)
            rows["label"].append(label)
        cleaned[split] = Dataset.from_dict(rows, features=TEXT_FEATURES)
    return DatasetDict(cleaned)


def build_sentence_lookup(raw_sentences: DatasetDict) -> dict[Words, int]:
    # dict sentence -> label from all splits; sentenxes with inconsistent labels are omitted
    labels: dict[Words, set[int]] = defaultdict(set)
    for split in raw_sentences:
        for text, label in zip(raw_sentences[split]["text"], raw_sentences[split]["target"]):
            words = tuple(normalize_text(text).split())
            if words:
                labels[words].add(label)
    return {words: next(iter(found)) for words, found in labels.items() if len(found) == 1}


def index_by_first_word(lookup: dict[Words, int]) -> dict[str, list[Words]]:
    index: dict[str, list[Words]] = defaultdict(list)
    for words in lookup:
        index[words[0]].append(words)
    return index


def align_sentence_labels(
    words: list[str], lookup: dict[Words, int], index: dict[str, list[Words]]
) -> list[int] | None:
    #split review to known sentences and return label for each word, None if split doent exist
    n = len(words)
    prev: list[Words | None] = [None] * (n + 1)
    reachable = [False] * (n + 1)
    reachable[0] = True
    for start in range(n):
        if not reachable[start]:
            continue
        for sentence in index.get(words[start], ()):
            end = start + len(sentence)
            if end <= n and not reachable[end] and tuple(words[start:end]) == sentence:
                reachable[end] = True
                prev[end] = sentence
    if not reachable[n]:
        return None

    labels: list[int] = []
    end = n
    while end > 0:
        sentence = prev[end]
        labels[:0] = [lookup[sentence]] * len(sentence)
        end -= len(sentence)
    return labels


def build_token_dataset(texts: DatasetDict, lookup: dict[Words, int]) -> tuple[DatasetDict, dict[str, float]]:
    # zostawiamy tylko recenzje ktore w całości da się złożyć z oznaczonych zdań
    index = index_by_first_word(lookup)
    token_splits, coverage = {}, {}
    for split in SPLITS:
        rows = {"id": [], "words": [], "labels": []}
        for review_id, text in zip(texts[split]["id"], texts[split]["text"]):
            words = text.split()
            labels = align_sentence_labels(words, lookup, index)
            if labels is not None:
                rows["id"].append(review_id)
                rows["words"].append(words)
                rows["labels"].append(labels)
        token_splits[split] = Dataset.from_dict(rows, features=TOKEN_FEATURES)
        coverage[split] = len(rows["id"]) / len(texts[split])
    return DatasetDict(token_splits), coverage


def subsample_train(texts: DatasetDict, max_train: int) -> DatasetDict:
    # stratified subsampling
    if max_train >= len(texts["train"]):
        return texts
    sampled = texts["train"].train_test_split(train_size=max_train, stratify_by_column="label", seed=SEED)["train"]
    return DatasetDict({**texts, "train": sampled.sort("id")})


def label_distribution(labels: list[int]) -> dict[str, int]:
    counts = Counter(labels)
    return {name: counts.get(i, 0) for i, name in enumerate(LABEL_NAMES)}


def prepare(out_dir: Path = PROCESSED_DIR, max_train: int | None = None) -> dict:
    raw_texts = load_raw("text")
    texts = clean_texts(raw_texts)
    if max_train is not None:
        texts = subsample_train(texts, max_train)
    tokens, coverage = build_token_dataset(texts, build_sentence_lookup(load_raw("sentence")))

    for task, dataset in {"text": texts, "token": tokens}.items():
        (out_dir / task).mkdir(parents=True, exist_ok=True)
        for split, ds in dataset.items():
            ds.to_parquet(out_dir / task / f"{split}.parquet")

    metadata = {
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "seed": SEED,
        "max_train": max_train,
        "label_names": LABEL_NAMES,
        "text": {
            split: {
                "raw": len(raw_texts[split]),
                "clean": len(texts[split]),
                "labels": label_distribution(texts[split]["label"]),
            }
            for split in SPLITS
        },
        "token": {
            split: {
                "reviews": len(tokens[split]),
                "review_coverage": round(coverage[split], 4),
                "labels": label_distribution([lab for seq in tokens[split]["labels"] for lab in seq]),
            }
            for split in SPLITS
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def load_processed(task: str, data_dir: Path = PROCESSED_DIR) -> DatasetDict:
    features = TEXT_FEATURES if task == "text" else TOKEN_FEATURES
    files = {split: str(data_dir / task / f"{split}.parquet") for split in SPLITS}
    return load_dataset("parquet", data_files=files, features=features)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--max-train", type=int, default=None, help="limit recenzji treningowych (stratyfikowany)")
    args = parser.parse_args()
    metadata = prepare(args.out_dir, args.max_train)
    print(json.dumps({task: metadata[task] for task in ("text", "token")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
