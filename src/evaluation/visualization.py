# wykres przestrzeni wektorowej: embeddingi (n, dim) + etykiety -> interaktywny scatter 2d
# embeddingi licza sie osobno dla modelu ([CLS] w bercie, ostatni token w gpt-2)

import html
import textwrap
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.config import LABEL_DESCRIPTIONS, LABEL_NAMES, SEED

# colorblind-checked palette; red/green pair is close for protanopia, so each class also has its own marker
LABEL_STYLE = {
    "zero": ("#2a78d6", "circle"),
    "minus": ("#e34948", "square"),
    "plus": ("#008300", "diamond"),
    "amb": ("#eda100", "triangle-up"),
}


def reduce_embeddings(embeddings, method: str = "umap", seed: int = SEED) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float32)
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(x)
    if method == "tsne":
        perplexity = min(30.0, (len(x) - 1) / 3)
        return TSNE(n_components=2, perplexity=perplexity, metric="cosine", init="pca", random_state=seed).fit_transform(x)
    if method == "umap":
        import umap  # heavy import (numba), only when used

        return umap.UMAP(n_components=2, metric="cosine", random_state=seed, n_jobs=1).fit_transform(x)
    raise ValueError(f"unknown method: {method} (pca, tsne, umap)")


def _hover_text(text: str | None, gold: int, pred: int | None) -> str:
    lines = [f"etykieta: {LABEL_DESCRIPTIONS[LABEL_NAMES[gold]]}"]
    if pred is not None:
        mark = "✓" if pred == gold else "✗"
        lines.append(f"predykcja: {LABEL_DESCRIPTIONS[LABEL_NAMES[pred]]} {mark}")
    if text:
        snippet = textwrap.shorten(text, width=200, placeholder=" …")
        lines.append("<br>".join(html.escape(line) for line in textwrap.wrap(snippet, width=60)))
    return "<br>".join(lines)


def plot_embeddings(
    embeddings,
    labels,
    texts: list[str] | None = None,
    predictions=None,
    method: str = "umap",
    title: str = "Przestrzeń wektorowa przykładów testowych",
    save_path: str | Path | None = None,
) -> go.Figure:
    # labels/predictions are class ids; with predictions, misclassified points get an open marker
    coords = reduce_embeddings(embeddings, method)
    labels = np.asarray(labels)
    preds = None if predictions is None else np.asarray(predictions)

    fig = go.Figure()
    for label_id, name in enumerate(LABEL_NAMES):
        idx = np.flatnonzero(labels == label_id)
        if len(idx) == 0:
            continue
        color, symbol = LABEL_STYLE[name]
        symbols = [symbol if preds is None or preds[i] == label_id else f"{symbol}-open" for i in idx]
        fig.add_trace(
            go.Scatter(
                x=coords[idx, 0],
                y=coords[idx, 1],
                mode="markers",
                name=f"{LABEL_DESCRIPTIONS[name]} ({len(idx)})",
                marker={"color": color, "symbol": symbols, "size": 8, "line": {"width": 1, "color": "#fcfcfb"}},
                hovertext=[
                    _hover_text(None if texts is None else texts[i], int(labels[i]), None if preds is None else int(preds[i]))
                    for i in idx
                ],
                hoverinfo="text",
                hoverlabel={"bordercolor": color},
            )
        )

    subtitle = f"{method.upper()}, n = {len(labels)}" + (" · puste znaczniki = błędna predykcja" if preds is not None else "")
    fig.update_layout(
        title={"text": f"{title}<br><sup>{subtitle}</sup>"},
        template="plotly_white",
        legend={"title": {"text": "etykieta"}, "itemsizing": "constant"},
        xaxis={"showticklabels": False, "title": None, "zeroline": False},
        yaxis={"showticklabels": False, "title": None, "zeroline": False, "scaleanchor": "x"},
        hoverlabel={"align": "left", "bgcolor": "#ffffff", "font": {"color": "#0b0b0b"}},
        width=900,
        height=700,
    )
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path, include_plotlyjs="cdn")
    return fig
