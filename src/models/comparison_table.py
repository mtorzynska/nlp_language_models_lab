"""Tabela porównawcza przed vs po augmentacji dla BERT (Etap 8).

Czyta metryki "przed augmentacji" z lokalnych wyników treningu
(data/HerbertFor{Sequence,Token}Classification/{lora,head}/metrics.json)
oraz "po augmentacji" z treningu na augmentowanych danych
(data/HerbertFor{Sequence,Token}Classification/augmented/metrics.json).

Uruchomienie:
    uv run python -m src.models.comparison_table
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import DATA_DIR

def results_dir(task: str) -> Path:
    return DATA_DIR / f"HerbertFor{task}Classification"


def load_metrics_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_trainer_state(path: Path) -> dict:
    """Wyciąga ostatnie eval_f1 z trainer_state.json."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in reversed(data.get("log_history", [])):
        if "eval_f1" in entry:
            return {
                "f1_score": entry["eval_f1"],
                "eval_loss": entry.get("eval_loss"),
                "step": entry.get("step"),
                "epoch": entry.get("epoch"),
            }
    return {}


def get_seq_metrics(mode: str, downloaded_root: Path | None = None) -> dict:
    """Metryki klasyfikacji sekwencji (tekst) dla danego trybu.

    Najpierw próbuje odczytać z lokalnego metrics.json, potem z trainer_state.json,
    na koniec z pobranych wyników (downloaded_root).
    """
    local_metrics = results_dir("Sequence") / mode / "metrics.json"
    m = load_metrics_json(local_metrics)
    if m:
        return m

    local_trainer = (
        results_dir("Sequence") / mode / "results" / "checkpoint-1233" / "trainer_state.json"
    )
    m = load_trainer_state(local_trainer)
    if m:
        return m

    if downloaded_root:
        dl_trainer = (
            downloaded_root
            / f"HerbertForSequenceClassification/{mode}"
            / "results"
            / "checkpoint-1233"
            / "trainer_state.json"
        )
        m = load_trainer_state(dl_trainer)
        if m:
            return m

    return {}


def main() -> None:
    downloaded_root = DATA_DIR / "downloaded" / "lab2_results"

    # === PRZED augmentacji ===
    seq_lora = get_seq_metrics("lora", downloaded_root)
    seq_head = get_seq_metrics("head", downloaded_root)

    tok_head = load_metrics_json(
        results_dir("Token") / "head" / "metrics.json"
    )
    tok_lora = load_metrics_json(
        results_dir("Token") / "lora" / "metrics.json"
    )

    # === PO augmentacji ===
    seq_lora_aug = load_metrics_json(
        results_dir("Sequence") / "augmented" / "metrics.json"
    )
    seq_head_aug = load_metrics_json(
        results_dir("Sequence") / "augmented" / "metrics.json"
    )
    # Dla tekstu augmented używa tego samego run-name dla lora i head
    # (rozróżniane przez --run-name podczas treningu)
    tok_head_aug = load_metrics_json(
        results_dir("Token") / "augmented" / "metrics.json"
    )
    tok_lora_aug = load_metrics_json(
        results_dir("Token") / "augmented" / "metrics.json"
    )

    def fmt_val(d: dict | None, key: str) -> str:
        if d and key in d:
            return f"{d[key]:>10.4f}"
        return f"{'—':>10}"

    def fmt_delta(before: dict | None, after: dict | None, key: str) -> str:
        if before and after and key in before and key in after:
            return f"{after[key] - before[key]:>+10.4f}"
        return f"{'—':>10}"

    print("=" * 82)
    print("BERT — wyniki PRZED i PO augmentacji (Etap 8)")
    print("=" * 82)
    print()
    print(
        f"{'Zadanie':<24} {'Tryb':<10} {'F1 przed':>10} {'F1 po':>10} {'Δ F1':>10} "
        f"{'Acc przed':>10} {'Acc po':>10}"
    )
    print("-" * 82)

    rows = [
        ("Klasyfikacja tekstu", "LoRA", seq_lora, seq_lora_aug),
        ("Klasyfikacja tekstu", "head", seq_head, seq_head_aug),
        ("Klasyfikacja tokenów", "head", tok_head, tok_head_aug),
        ("Klasyfikacja tokenów", "LoRA", tok_lora, tok_lora_aug),
    ]

    for task, mode, before, after in rows:
        print(
            f"{task:<24} {mode:<10} "
            f"{fmt_val(before, 'f1_score')} {fmt_val(after, 'f1_score')} "
            f"{fmt_delta(before, after, 'f1_score')} "
            f"{fmt_val(before, 'accuracy')} {fmt_val(after, 'accuracy')}"
        )

    print()
    print("Legenda: F1 przed / F1 po / Δ F1 — metryki ze zbioru testowego.")
    print("Acc przed / Acc po — accuracy ze zbioru testowego.")
    print("Jeśli komórka pokazuje '—', oznacza brak metryk (trening jeszcze nie uruchomiony).")
    print()

    # Zapisz surowe dane do JSON dla raportu
    report = {
        "przed_augmentacji": {
            "sekwencja_lora": seq_lora,
            "sekwencja_head": seq_head,
            "tokeny_head": tok_head,
            "tokeny_lora": tok_lora,
        },
        "po_augmentacji": {
            "sekwencja_lora": seq_lora_aug,
            "sekwencja_head": seq_head_aug,
            "tokeny_head": tok_head_aug,
            "tokeny_lora": tok_lora_aug,
        },
    }
    out = DATA_DIR / "bert_comparison_table.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Surowe dane zapisane: {out}")


if __name__ == "__main__":
    main()