"""Wyciąganie embedingów [CLS] z HerBERTa i tworzenie interaktywnych wykresów (Etap 6).

Używa bazowego modelu HerBERTa (HerBERT base cased) do wyciągnięcia wektora [CLS]
dla każdej recenzji ze zbioru testowego, a następnie redukuje wymiarowość UMAP i t-SNE
i zapisuje jako jeden interaktywny plik HTML z zakładkami.

Uruchomienie:
    uv run python -m src.models.embeddings --task text [--run-name lora] [--max-samples 2000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict
from transformers import AutoModel, AutoTokenizer

from src.config import BERT_MODEL, DATA_DIR, ID2LABEL, LABEL_NAMES, PROCESSED_DIR
from src.data import load_processed
from src.logger import logger
from src.utils import plot_embeddings


def extract_cls_embeddings(
    dataset: DatasetDict,
    *,
    model_name: str = BERT_MODEL,
    batch_size: int = 32,
    max_samples: int | None = None,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Wyciąga wektory [CLS] i etykiety ze zbioru testowego.

    Zwraca (embeddings, labels) jako numpy arrays.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Używam urządzenia: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Używamy tylko splitu test
    test_ds = dataset["test"]
    if max_samples is not None and len(test_ds) > max_samples:
        test_ds = test_ds.shuffle(seed=42).select(range(max_samples))
        logger.info(f"Ograniczono do {max_samples} próbek testowych")

    all_embeddings: list[np.ndarray] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for i in range(0, len(test_ds), batch_size):
            batch = test_ds[i : i + batch_size]
            texts = batch["text"]
            labels = batch["label"]

            inputs = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            # last_hidden_state: (batch, seq_len, hidden)
            # [CLS] to pierwszy token (indeks 0)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            all_embeddings.append(cls_embeddings)
            all_labels.extend(labels)

    embeddings = np.vstack(all_embeddings)
    labels_arr = np.asarray(all_labels)
    logger.info(f"Wyciągnięto embedingi: {embeddings.shape}, etykiety: {labels_arr.shape}")
    return embeddings, labels_arr


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--task",
        choices=["text", "token"],
        default="text",
        help="zadanie (domyślnie: text — tylko dla tekstu mają sens embedingi [CLS] recenzji)",
    )
    parser.add_argument(
        "--run-name",
        default="lora",
        help="nazwa przebiegu (domyślnie: lora)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR / "HerbertEmbeddings",
        help="katalog wyjściowy dla wykresów",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="maksymalna liczba próbek testowych (domyślnie: wszystkie)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="rozmiar partii przy wyciąganiu embedingów",
    )
    args = parser.parse_args()

    dataset = load_processed(args.task, data_dir=PROCESSED_DIR)

    embeddings, labels = extract_cls_embeddings(
        dataset,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    label_names = [LABEL_NAMES[i] for i in labels]
    label_str = [str(l) for l in labels]

    out_dir = args.out_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / "cls_embeddings.html"
    plot_embeddings(
        embeddings,
        label_str,
        output_path=html_path,
        title=f"HerBERT [CLS] embeddings ({args.task}, {args.run_name})",
        color_col="etykieta",
        interactive=True,
    )
    logger.info(f"Wykresy zapisane: {html_path}")

    # Zapisz też surowe embedingi i etykiety do pliku npz dla dalszych analiz
    npz_path = out_dir / "cls_embeddings.npz"
    np.savez(npz_path, embeddings=embeddings, labels=labels)
    logger.info(f"Surowe embedingi zapisane: {npz_path}")


if __name__ == "__main__":
    main()