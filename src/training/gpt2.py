# trening gpt-2 + lora, ocena na zbiorze testowym i zapis adaptera do outputs/gpt2-{task}/
# uv run python -m src.training.gpt2 --task text
# test na cpu: --max-train-samples 16 --max-eval-samples 8 --epochs 1 --max-length 64

import argparse
import json
from pathlib import Path

import torch
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments, set_seed

from src.config import GPT2_MODEL, LABEL_NAMES, PROJECT_ROOT, SEED
from src.data import get_collator, load_processed, load_tokenizer
from src.evaluation import compute_metrics, evaluation_report, logits_to_predictions
from src.models.gpt2 import ENCODERS, build_gpt2_model

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_subsets(task: str, max_train: int | None, max_eval: int | None) -> DatasetDict:
    dataset = load_processed(task)
    limits = {"train": max_train, "validation": max_eval, "test": max_eval}
    return DatasetDict({
        split: ds.select(range(min(limits[split], len(ds)))) if limits[split] else ds
        for split, ds in dataset.items()
    })


def train(args: argparse.Namespace) -> dict:
    set_seed(SEED)
    out_dir = Path(args.output_dir or OUTPUT_DIR / f"gpt2-{args.task}")
    tokenizer = load_tokenizer(GPT2_MODEL, max_length=args.max_length)
    raw = load_subsets(args.task, args.max_train_samples, args.max_eval_samples)
    encoded = raw.map(
        ENCODERS[args.task], batched=True, fn_kwargs={"tokenizer": tokenizer}, remove_columns=raw["train"].column_names
    )
    model = build_gpt2_model(args.task, tokenizer, r=args.lora_r, lora_alpha=2 * args.lora_r)

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
    if args.task == "text":
        assert test.label_ids.tolist() == encoded["test"]["labels"], "predictions are not in dataset order"
    report, confusion = evaluation_report(test.predictions, test.label_ids)
    print(report)

    model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    if args.task == "text":
        raw["test"].add_column("prediction", test.predictions.tolist()).to_parquet(out_dir / "test_predictions.parquet")

    results = {
        "model": GPT2_MODEL,
        "task": args.task,
        "args": vars(args),
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
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--gradient-checkpointing", action="store_true", help="mniej pamieci gpu, wolniej")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
