"""HerBERT for sequence-level classification with PEFT-based fine tuning."""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless save, no display needed
import matplotlib.pyplot as plt
import torch
from datasets import DatasetDict
from evaluate import load
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import (
    BERT_MODEL,
    DATA_DIR,
    DATASET_ID,
    ID2LABEL,
    LABEL2ID,
    LABEL_NAMES,
    MODELS_DIR,
)
from src.data import encode_text_classification, load_processed
from src.logger import logger


class HerbertForSequenceClassification:
    MODEL_PATH: Path = MODELS_DIR / "HerbertForSequenceClassification"
    CHECKPOINTS_DIR: Path = DATA_DIR / "HerbertForSequenceClassification" / "results"

    def __init__(self) -> None:
        self._tokeniser = AutoTokenizer.from_pretrained(BERT_MODEL)
        label_count = len(LABEL_NAMES)
        self._model = BertForSequenceClassification.from_pretrained(
            BERT_MODEL,
            num_labels=label_count,  # Creates an unitialised head.
        )
        # TODO: add softmax + argmax or solely argmax
        #

        self._data_collator = DataCollatorWithPadding(tokenizer=self._tokeniser)

        # weights
        self._model.config.id2label = ID2LABEL
        self._model.config.label2id = LABEL2ID
        print(self._model.config.id2label)
        print(self._model.config.label2id)

        logger.debug(self._model.config)

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
            dataset = load_processed("text")

        if lora:
            # PEFT: adapt attention weights with LoRA, keep the classifier trainable.
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                target_modules=["query", "value"],
                modules_to_save=["classifier"],
            )
            self._model = get_peft_model(self._model, lora_config)
        else:
            # Freeze all layers except the classifier.
            for param in self._model.bert.parameters():
                param.requires_grad = False

            # Keep only the classification head trainable.
            for param in self._model.classifier.parameters():
                param.requires_grad = True

        self._model.print_trainable_parameters()

        tokenised_dataset = dataset.map(
            encode_text_classification,
            batched=True,
            fn_kwargs={"tokenizer": self._tokeniser},
            remove_columns=["id", "text"],
        )

        # Freeze all layers except the classifier.
        for param in self._model.bert.parameters():
            param.requires_grad = False

        # Keep only the classification head trainable.
        for param in self._model.classifier.parameters():
            param.requires_grad = True

        logger.debug(
            f"Trainable parameters: {sum(p.numel() for p in self._model.parameters() if p.requires_grad)}"
        )

        logger.debug(
            f"Trainable parameters: {sum(p.numel() for p in self._model.parameters() if p.requires_grad)}"
        )

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=str(self.CHECKPOINTS_DIR),  # training checkpoints
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
            predictions = logits.argmax(axis=-1)
            return metric.compute(
                predictions=predictions, references=labels, average="macro"
            )

        self._trainer = Trainer(
            model=self._model,  # Pre-trained BERT model
            args=training_args,  # Training arguments
            train_dataset=tokenised_dataset["train"],
            eval_dataset=tokenised_dataset["validation"],
            # tokenizer=self._tokeniser,
            data_collator=self._data_collator,  # Efficient batching
            compute_metrics=compute_metrics,  # Custom metric
        )

        # Start training
        self._trainer.train()

    def evaluate(self, dataset: DatasetDict | None = None) -> dict[str, float]:
        if dataset is None:
            dataset = load_processed("text")
        tokenised_dataset = dataset.map(
            encode_text_classification,
            batched=True,
            fn_kwargs={"tokenizer": self._tokeniser},
            remove_columns=["id", "text"],
        )

        # Generate predictions
        predictions = self._trainer.predict(tokenised_dataset["test"])
        predicted_labels = predictions.predictions.argmax(axis=-1)
        references = tokenised_dataset["test"]["label"]

        # Classification report
        print(
            classification_report(references, predicted_labels)
        )

        # Confusion matrix
        cm = confusion_matrix(references, predicted_labels)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=list(LABEL_NAMES)
        )
        disp.plot()
        results_dir = self.CHECKPOINTS_DIR.parent
        results_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(results_dir / "confusion_matrix.png", bbox_inches="tight")
        plt.close()

        # Calculate metrics.
        metric = load("f1")
        metrics = {
            "f1_score": float(
                metric.compute(
                    predictions=predicted_labels, references=references, average="macro"
                )["f1"]
            ),
            "accuracy": float((predicted_labels == references).mean()),
        }
        (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics

    def __call__(self, sentences: list[tuple[str, str]], *args, **kwargs) -> Any:

        texts, text_pairs = zip(*sentences)

        inputs = self._tokeniser(
            list(texts),
            list(text_pairs),
            padding="longest",
            add_special_tokens=True,
            return_tensors="pt",
        )
        return self._model(**inputs)

    def save(self, path: Path | None = None) -> None:
        save_path = path or self.MODEL_PATH
        save_path.mkdir(exist_ok=True, parents=True)
        self._model.save_pretrained(save_path)
        self._tokeniser.save_pretrained(save_path)
        # Keep metrics next to the model for provenance.
        for name in ("metrics.json", "confusion_matrix.png"):
            results_file = self.CHECKPOINTS_DIR.parent / name
            if results_file.exists():
                (save_path / name).write_bytes(results_file.read_bytes())


def main() -> None:
    bert = HerbertForSequenceClassification()
    bert.fine_tune()
    bert.evaluate()
    bert.save()


if __name__ == "__main__":
    main()
