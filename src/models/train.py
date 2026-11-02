# zeby uruchomic: uv run python -m src.models.train --task text [--max-train N]
# z augmentacja: uv run python -m src.models.train --task text --augmented --run-name augmented

"""Fine-tuning HerBERTa na przygotowanym zbiorze PolEmo2.

Domyślnie czyta dane z data/processed/ (wygenerowane przez src.data.prepare).
Z --augmented trenuje na danych augmentowanych z data/augmented/,
zachowując oryginalny podział walidacyjny i testowy.
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
from src.config import DATA_DIR, PROCESSED_DIR, LABEL_NAMES
from src.data import load_processed
from src.data.prepare import subsample_train
from src.logger import logger
from src.models.encoders.herbert_for_sequence_classification import (
    HerbertForSequenceClassification,
)
from src.models.encoders.herbert_for_token_classification import (
    HerbertForTokenClassification,
)

TEXT_FEATURES = Features(
    {
        "id": Value("string"),
        "text": Value("string"),
        "label": ClassLabel(names=LABEL_NAMES),
    }
)

ENCODERS = {
    "text": HerbertForSequenceClassification,
    "token": HerbertForTokenClassification,
}


def load_augmented_dataset(
    task: str,
    aug_dir: Path = DATA_DIR / "augmented",
    processed_dir: Path = PROCESSED_DIR,
) -> DatasetDict:
    """Ładuje dane augmentowane jako zbiór treningowy,
    a walidację i test z oryginalnie przetworzonych danych.

    Augmentacja zawiera tylko split train; walidacja i test
    pochodzą z data/processed/.
    """
    aug_task_dir = aug_dir / task
    if not aug_task_dir.exists():
        raise FileNotFoundError(f"Katalog augmentacji nie istnieje: {aug_task_dir}")

    # Szukaj pliku combined lub pierwszego dostępnego pliku augmentacji train
    combined = aug_task_dir / "augmented_combined_train.json"
    if combined.exists():
        train_file = combined
    else:
        train_files = sorted(aug_task_dir.glob("augmented_mask_*_train.json"))
        if not train_files:
            raise FileNotFoundError(
                f"Brak plików augmentacji train w {aug_task_dir}"
            )
        train_file = train_files[0]

    logger.info(f"Ładowanie danych augmentowanych z {train_file}")
    with train_file.open(encoding="utf-8") as f:
        train_data = json.load(f)

    train_ds = Dataset.from_dict(
        {
            "id": [ex["id"] for ex in train_data],
            "text": [ex["text"] for ex in train_data],
            "label": [ex["label"] for ex in train_data],
        },
        features=TEXT_FEATURES,
    )

    processed = load_processed(task, data_dir=processed_dir)

    return DatasetDict(
        {
            "train": train_ds,
            "validation": processed["validation"],
            "test": processed["test"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=sorted(ENCODERS), default="text")
    parser.add_argument(
        "--mode",
        choices=["lora", "head"],
        default="lora",
        help="lora: adaptery PEFT (domyślnie), head: tylko głowa klasyfikacyjna",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="nazwa przebiegu (domyślnie: tryb, np. lora). Wyniki trafiają do data/<task>/<run-name>/",
    )
    parser.add_argument(
        "--augmented",
        action="store_true",
        help="trenuj na augmentowanych danych z data/augmented/ (tylko train; walidacja i test z processed)",
    )
    parser.add_argument(
        "--augmented-dir",
        type=Path,
        default=DATA_DIR / "augmented",
        help="katalog z danymi augmentowanymi (domyślnie: data/augmented)",
    )
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="losuje N recenzji treningowych ze stratyfikacją po etykiecie",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    args = parser.parse_args()

    if args.augmented:
        dataset = load_augmented_dataset(
            args.task, aug_dir=args.augmented_dir, processed_dir=args.out_dir
        )
        logger.info(
            f"Trening na augmentowanych danych: {len(dataset['train'])} przykładów train"
        )
    else:
        dataset = load_processed(args.task, data_dir=args.out_dir)
        if args.max_train is not None:
            dataset = subsample_train(dataset, args.max_train)
            logger.info(f"Train subsampled to {len(dataset['train'])} examples")

    run_name = args.run_name or args.mode
    encoder = ENCODERS[args.task](run_name=run_name)
    logger.info(f"Run: {run_name} — results in {encoder._results_dir}")
    encoder.fine_tune(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora=args.mode == "lora",
    )
    metrics = encoder.evaluate(dataset)
    logger.info(f"Test metrics: {metrics}")
    encoder.save()


if __name__ == "__main__":
    main()