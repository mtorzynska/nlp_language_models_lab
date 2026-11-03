# trening gpt-2 + lora, ocena na zbiorze testowym i zapis adaptera do outputs/<run-name>/ (domyslnie gpt2-{task})
# uv run python -m src.training.gpt2 --task text
# tryby: lora albo head (zamrozony gpt-2, trenowana tylko glowica), z --augmented train to dane z augmentacji

import argparse
import json
import shutil
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import Trainer, TrainingArguments, set_seed

from src.config import GPT2_MODEL, LABEL_NAMES, PROJECT_ROOT, SEED
from src.data import get_collator, load_processed, load_tokenizer
from src.data.prepare import TEXT_FEATURES
from src.data.tokenization import IGNORE_INDEX
from src.evaluation import compute_metrics, evaluation_report, logits_to_predictions
from src.models.gpt2 import ENCODERS, HEADS, build_gpt2_model

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_augmented(paths: list[str]) -> Dataset:
    # json from augmentation scripts (gpt-2 and herbert): list of {"id", "text", "label", ...}
    rows = [row for path in paths for row in json.loads(Path(path).read_text(encoding="utf-8"))]
    return Dataset.from_dict({key: [row[key] for row in rows] for key in TEXT_FEATURES}, features=TEXT_FEATURES)


def load_subsets(
    task: str, max_train: int | None, max_eval: int | None, augmented: list[str] | None = None, keep_original: bool = False
) -> DatasetDict:
    dataset = load_processed(task)
    if augmented:
        if task != "text":
            raise ValueError("augmentacja daje tylko tekst + etykiete recenzji, wiec jest tylko dla --task text")
        # as in the herbert part: train = augmented examples only, validation and test stay original
        train = load_augmented(augmented)
        dataset["train"] = concatenate_datasets([dataset["train"], train]) if keep_original else train
    limits = {"train": max_train, "validation": max_eval, "test": max_eval}
    subsets = {}
    for split, ds in dataset.items():
        if limits[split]:
            # train is shuffled first so a small subset also contains augmented examples
            ds = (ds.shuffle(seed=SEED) if split == "train" else ds).select(range(min(limits[split], len(ds))))
        subsets[split] = ds
    return DatasetDict(subsets)


def test_predictions(task: str, predictions, label_ids, encoded_labels: list) -> list:
    # checks that predict kept the dataset order, returns one prediction per review (text) or per word (token)
    if task == "text":
        assert label_ids.tolist() == encoded_labels, "predictions are not in dataset order"
        return predictions.tolist()
    # token: labelled positions are first subwords of words (words cut by truncation have no prediction)
    gold = [[label for label in row if label != IGNORE_INDEX] for row in encoded_labels]
    assert [row[row != IGNORE_INDEX].tolist() for row in label_ids] == gold, "predictions are not in dataset order"
    return [pred[row != IGNORE_INDEX].tolist() for pred, row in zip(predictions, label_ids)]


def save_head(model, task: str, path: Path) -> None:
    # head mode: the base model is unchanged, so only the trained head is saved (not 1.4 GB of gpt-2)
    path.mkdir(parents=True, exist_ok=True)
    head = {name: tensor for name, tensor in model.state_dict().items() if name.startswith(HEADS[task])}
    torch.save(head, path / "head.pt")


def train(args: argparse.Namespace) -> dict:
    set_seed(SEED)
    out_dir = Path(args.output_dir or OUTPUT_DIR / (args.run_name or f"gpt2-{args.task}"))
    tokenizer = load_tokenizer(GPT2_MODEL, max_length=args.max_length)
    raw = load_subsets(args.task, args.max_train_samples, args.max_eval_samples, args.augmented, args.keep_original)
    encoded = raw.map(
        ENCODERS[args.task], batched=True, fn_kwargs={"tokenizer": tokenizer}, remove_columns=raw["train"].column_names
    )
    model = build_gpt2_model(args.task, tokenizer, mode=args.mode, r=args.lora_r, lora_alpha=2 * args.lora_r)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=2 * args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=0.01,
        warmup_steps=0.06,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        # no optimizer state in checkpoints, only the best model is reloaded at the end
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        # no group_by_length: in transformers 5 it also reorders eval/predict, so predictions dont match ids
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        # non-reentrant checkpointing works with frozen base weights (lora)
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20,
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        data_collator=get_collator(tokenizer, args.task),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=logits_to_predictions,
    )
    trainer.train()

    # best checkpoint (by validation macro f1) is loaded
    validation = trainer.evaluate()
    test = trainer.predict(encoded["test"], metric_key_prefix="test")
    predictions = test_predictions(args.task, test.predictions, test.label_ids, encoded["test"]["labels"])
    report, confusion = evaluation_report(test.predictions, test.label_ids)
    print(report)

    if args.mode == "head":
        save_head(model, args.task, out_dir / "head")
    else:
        model.save_pretrained(out_dir / "adapter")
        tokenizer.save_pretrained(out_dir / "adapter")
    # best model is saved above, checkpoints are redundant (in head mode 1.4 GB each)
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    raw["test"].add_column("prediction", predictions).to_parquet(out_dir / "test_predictions.parquet")

    results = {
        "model": GPT2_MODEL,
        "task": args.task,
        "mode": args.mode,
        "args": vars(args),
        "train_size": len(raw["train"]),
        "label_names": LABEL_NAMES,
        "validation": validation,
        "test": test.metrics,
        "confusion_matrix": confusion.tolist(),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "test_report.txt").write_text(report, encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["text", "token"], default="text")
    parser.add_argument("--mode", choices=["lora", "head"], default="lora", help="head: zamrozony gpt-2, trenowana tylko glowica")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--gradient-checkpointing", action="store_true", help="mniej pamieci gpu, wolniej")
    parser.add_argument("--augmented", nargs="+", default=None, help="pliki json z augmentacji, zastepuja train")
    parser.add_argument("--keep-original", action="store_true", help="z --augmented: oryginalny train + augmentacja")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None, help="katalog w outputs/ (domyslnie gpt2-{task})")
    parser.add_argument("--output-dir", type=str, default=None, help="pelna sciezka wynikow, np. na dysku google")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
