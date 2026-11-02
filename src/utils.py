"""Wspólne funkcje: metryki i wykresy (Etap 2).

Udostępniane zarówno dla modelu GPT-2, jak i HerBERTa.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

__all__ = [
    "compute_metrics",
    "plot_confusion_matrix",
    "plot_embeddings",
    "save_metrics",
]


def compute_metrics(
    predictions: Sequence[int],
    references: Sequence[int],
    *,
    average: str = "macro",
) -> dict[str, float]:
    """Oblicza accuracy i F1 (macro) dla predykcji i referencji."""
    predictions = np.asarray(predictions)
    references = np.asarray(references)
    return {
        "accuracy": float(accuracy_score(references, predictions)),
        "f1_score": float(f1_score(references, predictions, average=average)),
    }


def classification_report_text(
    predictions: Sequence[int],
    references: Sequence[int],
    label_names: Sequence[str],
) -> str:
    """Zwraca tekstowy raport klasyfikacji (sklearn)."""
    return classification_report(
        references, predictions, target_names=list(label_names)
    )


def save_metrics(
    metrics: dict[str, float],
    path: Path | str,
) -> None:
    """Zapisuje metryki do pliku JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_confusion_matrix(
    predictions: Sequence[int],
    references: Sequence[int],
    *,
    label_names: Sequence[str],
    output_path: Path | str,
    normalize: str | None = "true",
    title: str = "Macierz pomyłek",
) -> Path:
    """Rysuje i zapisuje macierz pomyłek jako PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(references, predictions, normalize=normalize)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=list(label_names)
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path


def plot_embeddings(
    embeddings: np.ndarray,
    labels: Sequence[int | str],
    *,
    output_path: Path | str,
    title: str = "Embeddingi [CLS]",
    color_col: str = "etykieta",
    interactive: bool = True,
) -> Path:
    """Rysuje 2 interaktywne wykresy (UMAP i t-SNE) i zapisuje jako HTML.

    Zwraca ścieżkę do pliku HTML (oba wykresy w jednym pliku jako zakładki).
    """
    try:
        import umap
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "umap-learn jest wymagane do wykresów embedingów: uv add umap-learn"
        ) from exc

    from sklearn.manifold import TSNE

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels_arr = np.asarray(labels)

    # UMAP
    umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42).fit_transform(
        embeddings
    )
    fig_umap = px.scatter(
        x=umap_emb[:, 0],
        y=umap_emb[:, 1],
        color=labels_arr.astype(str),
        title=f"{title} — UMAP",
        labels={"x": "UMAP 1", "y": "UMAP 2", color_col: color_col},
    )

    # t-SNE
    tsne_emb = TSNE(
        n_components=2, perplexity=30, metric="cosine", random_state=42, init="pca"
    ).fit_transform(embeddings)
    fig_tsne = px.scatter(
        x=tsne_emb[:, 0],
        y=tsne_emb[:, 1],
        color=labels_arr.astype(str),
        title=f"{title} — t-SNE",
        labels={"x": "t-SNE 1", "y": "t-SNE 2", color_col: color_col},
    )

    # Połącz oba wykresy w jeden plik HTML z zakładkami
    fig = go.Figure()
    for trace in fig_umap.data:
        fig.add_trace(trace)
    for trace in fig_tsne.data:
        fig.add_trace(trace)

    fig.update_layout(
        title=title,
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "buttons": [
                    {
                        "label": "UMAP",
                        "method": "update",
                        "args": [
                            {"visible": [True] * len(fig_umap.data) + [False] * len(fig_tsne.data)},
                            {"title": f"{title} — UMAP"},
                        ],
                    },
                    {
                        "label": "t-SNE",
                        "method": "update",
                        "args": [
                            {"visible": [False] * len(fig_umap.data) + [True] * len(fig_tsne.data)},
                            {"title": f"{title} — t-SNE"},
                        ],
                    },
                ],
            }
        ],
    )
    # Na start widoczny tylko UMAP
    n_umap = len(fig_umap.data)
    for i in range(len(fig.data)):
        fig.data[i].visible = i < n_umap

    fig.write_html(str(output_path))
    return output_path