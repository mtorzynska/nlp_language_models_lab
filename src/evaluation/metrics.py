# wspolne metryki dla obu zadan (text/token) i obu modeli (bert/gpt-2)

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import EvalPrediction

from src.config import LABEL_NAMES
from src.data.tokenization import IGNORE_INDEX

LABEL_IDS = list(range(len(LABEL_NAMES)))


def logits_to_predictions(logits, labels=None):
    # for Trainer(preprocess_logits_for_metrics=...): keeps argmax instead of full logits (saves memory in token task)
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def flatten_predictions(predictions, labels) -> tuple[np.ndarray, np.ndarray]:
    # accepts logits or class ids, drops positions with -100 (special tokens, padding, eos, next subwords)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    if predictions.ndim == labels.ndim + 1:
        predictions = predictions.argmax(-1)
    mask = labels != IGNORE_INDEX
    return predictions[mask], labels[mask]


def classification_metrics(predictions, labels) -> dict[str, float]:
    preds, labels = flatten_predictions(predictions, labels)
    per_class = f1_score(labels, preds, labels=LABEL_IDS, average=None, zero_division=0)
    # macro/weighted only over classes that occur, so a class missing from a small eval set doesnt count as 0
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        **{f"f1_{name}": float(score) for name, score in zip(LABEL_NAMES, per_class)},
    }


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    # Trainer(compute_metrics=compute_metrics), the same for text and token task
    return classification_metrics(eval_pred.predictions, eval_pred.label_ids)


def evaluation_report(predictions, labels) -> tuple[str, np.ndarray]:
    # final test evaluation: per-class precision/recall/f1 + confusion matrix (rows = gold, cols = predicted)
    preds, labels = flatten_predictions(predictions, labels)
    report = classification_report(labels, preds, labels=LABEL_IDS, target_names=LABEL_NAMES, digits=4, zero_division=0)
    return report, confusion_matrix(labels, preds, labels=LABEL_IDS)
