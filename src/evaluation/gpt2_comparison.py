# tabela wynikow gpt-2 na zbiorze testowym: tekst/tokeny, lora/head, przed i po augmentacji
# uv run python -m src.evaluation.gpt2_comparison [--root outputs]
# czyta <root>/*/results.json z src.training.gpt2, zapisuje <root>/gpt2_comparison.md

import argparse
import json
from pathlib import Path

from src.config import GPT2_MODEL, LABEL_NAMES, PROJECT_ROOT

TASKS = {"text": "tekst", "token": "tokeny"}


def load_runs(root: Path) -> list[dict]:
    runs = []
    for path in sorted(root.glob("*/results.json")):
        results = json.loads(path.read_text(encoding="utf-8"))
        if results.get("model") != GPT2_MODEL:
            continue
        runs.append({
            "run": path.parent.name,
            "task": results["task"],
            "mode": results.get("mode", "lora"),
            "augmented": bool(results["args"].get("augmented")),
            "keep_original": bool(results["args"].get("keep_original")),
            "train_size": results.get("train_size"),
            **{key.removeprefix("test_"): value for key, value in results["test"].items()},
        })
    return sorted(runs, key=lambda r: (r["task"] != "text", r["mode"] != "lora", r["augmented"]))


def comparison_table(runs: list[dict]) -> str:
    # delta = macro f1 minus the run with the same task and mode trained without augmentation
    baseline = {(r["task"], r["mode"]): r["f1_macro"] for r in runs if not r["augmented"]}
    header = ["przebieg", "zadanie", "tryb", "augmentacja", "train", "accuracy", "macro F1", "Δ macro F1"]
    header += [f"F1 {name}" for name in LABEL_NAMES]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in runs:
        base = baseline.get((r["task"], r["mode"]))
        delta = f"{r['f1_macro'] - base:+.4f}" if r["augmented"] and base is not None else "—"
        cells = [r["run"], TASKS[r["task"]], r["mode"], ("tak + oryginał" if r["keep_original"] else "tak") if r["augmented"] else "nie", str(r["train_size"] or "—"),
                 f"{r['accuracy']:.4f}", f"{r['f1_macro']:.4f}", delta]
        cells += [f"{r[f'f1_{name}']:.4f}" for name in LABEL_NAMES]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(PROJECT_ROOT / "outputs"), help="katalog z przebiegami, np. na dysku google")
    args = parser.parse_args()
    root = Path(args.root)
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"brak results.json w {root}/*/")
    table = comparison_table(runs)
    print(table)
    (root / "gpt2_comparison.md").write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
