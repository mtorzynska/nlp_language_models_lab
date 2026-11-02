"""Augmentacja danych przez podmiany [MASK] / fill-mask z HerBERTa (Etap 7).

Dla każdego zdania ze zbioru treningowego losowo wybierane są słowa do zastąpienia
tokenem [MASK], a następnie HerBERT (fill-mask) sugeruje zastępcze słowo.
Tworzone są 2 pliki z augmentowanymi danymi (różne proporcje maskowania).

Uruchomienie:
    uv run python -m src.data.augmentation --task text --split train [--max-samples 1000]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

from src.config import BERT_MODEL, DATA_DIR, LABEL_NAMES, PROCESSED_DIR
from src.data.prepare import load_processed
from src.logger import logger

MASK_RATIOS = [0.10, 0.20]  # dwa warianty augmentacji -> 2 pliki wyjściowe


def augment_with_mask(
    texts: list[str],
    labels: list[int],
    *,
    model_name: str = BERT_MODEL,
    mask_ratio: float = 0.1,
    max_samples: int | None = None,
    seed: int = 42,
    device: int = 0 if torch.cuda.is_available() else -1,
) -> list[dict]:
    """Zwraca listę augmentowanych przykładów (dict z text, label, mask_ratio)."""
    random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    filler = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=1,
    )

    mask_token = tokenizer.mask_token  # np. [MASK]

    augmented: list[dict] = []
    n = len(texts)
    if max_samples is not None:
        n = min(n, max_samples)
        indices = random.sample(range(len(texts)), n)
    else:
        indices = list(range(n))

    for idx in indices:
        text = texts[idx]
        label = labels[idx]
        words = text.split()
        n_mask = max(1, int(len(words) * mask_ratio))
        mask_positions = random.sample(range(len(words)), min(n_mask, len(words)))

        masked_words = list(words)
        for pos in mask_positions:
            masked_words[pos] = mask_token
        masked_text = " ".join(masked_words)

        # HerBERT fill-mask działa najlepiej na jednym [MASK] naraz;
        # dla wielu [MASK] wykonujemy iteracyjnie (pierwszy znaleziony).
        augmented_text = masked_text
        remaining_mask = mask_token in augmented_text
        max_iter = len(mask_positions)
        for _ in range(max_iter):
            if not remaining_mask:
                break
            result = filler(augmented_text)
            # fill-mask zwraca listę list (dla top_k=1: [[{'sequence': ...}]])
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            augmented_text = result["sequence"]
            remaining_mask = mask_token in augmented_text

        # Usuń ewentualne spacje przed znakami interpunkcyjnymi wprowadzone przez tokenizer
        augmented_text = augmented_text.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")

        augmented.append(
            {
                "id": f"aug_{mask_ratio}_{idx}",
                "text": augmented_text,
                "label": label,
                "mask_ratio": mask_ratio,
                "source_id": idx,
            }
        )

    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--task",
        choices=["text", "token"],
        default="text",
        help="zadanie (domyślnie: text)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="train",
        help="zbiór źródłowy (domyślnie: train, zgodnie z notatką w tabeli)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR / "augmented",
        help="katalog wyjściowy dla plików augmentowanych danych",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="maksymalna liczba przykładów do augmentacji (domyślnie: wszystkie)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="ziarno losowości",
    )
    args = parser.parse_args()

    dataset = load_processed(args.task, data_dir=PROCESSED_DIR)
    split_ds = dataset[args.split]
    texts = split_ds["text"]
    labels = split_ds["label"]

    out_dir = args.out_dir / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    all_augmented: list[dict] = []

    for mask_ratio in MASK_RATIOS:
        logger.info(f"Augmentacja z mask_ratio={mask_ratio}...")
        variant = augment_with_mask(
            texts,
            labels,
            mask_ratio=mask_ratio,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        variant_file = out_dir / f"augmented_mask_{int(mask_ratio * 100)}pct_{args.split}.json"
        with variant_file.open("w", encoding="utf-8") as f:
            json.dump(variant, f, ensure_ascii=False, indent=2)
        logger.info(f"Zapisano {len(variant)} przykładów do {variant_file}")
        all_augmented.extend(variant)

    # Plik łączny z oboma wariantami
    combined_file = out_dir / f"augmented_combined_{args.split}.json"
    with combined_file.open("w", encoding="utf-8") as f:
        json.dump(all_augmented, f, ensure_ascii=False, indent=2)
    logger.info(f"Zapisano łącznie {len(all_augmented)} przykładów do {combined_file}")
    logger.info("Gotowe — 2 pliki z augmentowanymi danymi (plus plik łączny).")


if __name__ == "__main__":
    main()