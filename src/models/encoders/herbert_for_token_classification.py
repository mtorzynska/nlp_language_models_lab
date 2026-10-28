"""HerBERT for token-level classification with PEFT-based fine tuning."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless save, no display needed
import matplotlib.pyplot as plt
from datasets import DatasetDict
from evaluate import load
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from transformers import (
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src.config import BERT_MODEL, DATA_DIR, ID2LABEL, LABEL2ID, MODELS_DIR
from src.data import encode_token_classification, load_processed
from src.data.tokenization import (
    IGNORE_INDEX,
    get_collator,
    load_tokenizer,
    model_kwargs,
)
from src.logger import logger


class HerbertForTokenClassification:
    def __init__(self, run_name: str = "default") -> None:
        self.run_name = run_name
        self._results_dir = DATA_DIR / "HerbertForTokenClassification" / run_name
        self._tokeniser = load_tokenizer(BERT_MODEL)
        self._model = AutoModelForTokenClassification.from_pretrained(
            BERT_MODEL, **model_kwargs(self._tokeniser)
        )
        self._model.config.id2label = ID2LABEL
        self._model.config.label2id = LABEL2ID
        logger.debug(self._model.config)
        self._trainer: Trainer | None = None

    def fine_tune(
        self,
        dataset: DatasetDict | None = None,
        *,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
    ) -> None:
        if dataset is None:
            dataset = load_processed("token")

        if lora:
            # PEFT: adapt attention weights with LoRA, keep the classifier trainable.
            lora_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                target_modules=["query", "value"],
                modules_to_save=["classifier"],
            )
            self._model = get_peft_model(self._model, lora_config)
            self._model.print_trainable_parameters()
        else:
            logger.info(
                f"Trainable parameters: {sum(p.numel() for p in self._model.parameters() if p.requires_grad)}"
            )

        tokenised_dataset = dataset.map(
            encode_token_classification,
            batched=True,
            fn_kwargs={"tokenizer": self._tokeniser, "label_all_subwords": True},
            remove_columns=["id", "words"],
        )

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=str(self._results_dir / "results"),  # training checkpoints
            eval_strategy="epoch",  # Evaluate at the end of each epoch.
            save_strategy="epoch",  # Must match eval strategy when load_best_model_at_end=True.
            learning_rate=learning_rate,  # Start with a small learning rate
            per_device_train_batch_size=batch_size,  # Batch size per GPU
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,  # Number of epochs
            weight_decay=0.01,  # Regularization
            save_total_limit=2,  # Limit checkpoints to save space
            load_best_model_at_end=True,  # Automatically load the best checkpoint
            # logging_dir="./logs",  # Directory for logs
            logging_steps=100,  # Log every 100 steps
            fp16=True,  # Enable mixed precision for faster training
        )

        print(training_args)

        # Load a metric (F1-score in this case)
        metric = load("f1")

        # Define a custom compute_metrics function
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = logits.argmax(axis=-1).flatten()
            references = labels.flatten()
            mask = references != IGNORE_INDEX
            return metric.compute(
                predictions=predictions[mask].tolist(),
                references=references[mask].tolist(),
                average="macro",
            )

        self._trainer = Trainer(
            model=self._model,  # Pre-trained BERT model
            args=training_args,  # Training arguments
            train_dataset=tokenised_dataset["train"],
            eval_dataset=tokenised_dataset["validation"],
            data_collator=get_collator(self._tokeniser, "token"),  # Efficient batching
            compute_metrics=compute_metrics,  # Custom metric
        )

        # Start training
        self._trainer.train()

    def evaluate(self, dataset: DatasetDict | None = None) -> dict[str, float]:
        if self._trainer is None:
            raise RuntimeError("Call fine_tune() before evaluate().")
        if dataset is None:
            dataset = load_processed("token")

        tokenised_dataset = dataset.map(
            encode_token_classification,
            batched=True,
            fn_kwargs={"tokenizer": self._tokeniser, "label_all_subwords": True},
            remove_columns=["id", "words"],
        )

        # Generate predictions
        predictions = self._trainer.predict(tokenised_dataset["test"])
        predicted_labels = predictions.predictions.argmax(axis=-1).flatten()
        references = predictions.label_ids.flatten()
        mask = references != IGNORE_INDEX

        # Confusion matrix over masked (labelled) tokens
        cm = confusion_matrix(
            references[mask],
            predicted_labels[mask],
            labels=range(len(ID2LABEL)),
            normalize="true",
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=list(ID2LABEL.values())
        )
        disp.plot()
        results_dir = self._results_dir
        results_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(results_dir / "confusion_matrix.png", bbox_inches="tight")
        plt.close()

        # Calculate metrics.
        metric = load("f1")
        metrics = {
            "f1_score": float(
                metric.compute(
                    predictions=predicted_labels[mask].tolist(),
                    references=references[mask].tolist(),
                    average="macro",
                )["f1"]
            ),
            "accuracy": float((predicted_labels[mask] == references[mask]).mean()),
        }
        (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics

    def save(self, path: Path | None = None) -> None:
        save_path = path or MODELS_DIR / "HerbertForTokenClassification" / self.run_name
        save_path.mkdir(exist_ok=True, parents=True)
        self._model.save_pretrained(save_path)
        self._tokeniser.save_pretrained(save_path)
        # Keep metrics next to the model for provenance.
        for name in ("metrics.json", "confusion_matrix.png"):
            results_file = self._results_dir / name
            if results_file.exists():
                (save_path / name).write_bytes(results_file.read_bytes())


def main() -> None:
    token_bert = HerbertForTokenClassification()
    token_bert.fine_tune()
    token_bert.evaluate()
    token_bert.save()


if __name__ == "__main__":
    main()
