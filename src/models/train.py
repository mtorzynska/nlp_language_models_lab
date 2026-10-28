# zeby uruchomic: uv run python -m src.models.train --task text [--max-train N]

"""Fine-tuning HerBERTa na przygotowanym zbiorze PolEmo2.

Domyślnie czyta dane z data/processed/ (wygenerowane przez src.data.prepare).
"""

import argparse
from pathlib import Path

from src.config import PROCESSED_DIR
from src.data import load_processed
from src.data.prepare import subsample_train
from src.logger import logger
from src.models.encoders.herbert_for_sequence_classification import (
    HerbertForSequenceClassification,
)
from src.models.encoders.herbert_for_token_classification import (
    HerbertForTokenClassification,
)

ENCODERS = {
    "text": HerbertForSequenceClassification,
    "token": HerbertForTokenClassification,
}


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

    dataset = load_processed(args.task, data_dir=args.out_dir)
    if args.max_train is not None:
        dataset = subsample_train(dataset, args.max_train)
        logger.info(f"Train subsampled to {len(dataset['train'])} examples")

    encoder = ENCODERS[args.task]()
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
