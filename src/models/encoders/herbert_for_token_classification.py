"""HerBERT for token-level classification with PEFT-based fine tuning."""

from pathlib import Path

from datasets import DatasetDict
from evaluate import load
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
    CHECKPOINTS_DIR: Path = DATA_DIR / "HerbertForTokenClassification" / "results"

    def __init__(self) -> None:
        self._tokeniser = load_tokenizer(BERT_MODEL)
        self._model = AutoModelForTokenClassification.from_pretrained(
            BERT_MODEL, **model_kwargs(self._tokeniser)
        )
        self._model.config.id2label = ID2LABEL
        self._model.config.label2id = LABEL2ID
        logger.debug(self._model.config)
        self._trainer: Trainer | None = None

    def fine_tune(self, dataset: DatasetDict | None = None) -> None:
        if dataset is None:
            dataset = load_processed("token")

        tokenised_dataset = dataset.map(
            encode_token_classification,
            batched=True,
            fn_kwargs={"tokenizer": self._tokeniser, "label_all_subwords": True},
            remove_columns=["id", "words"],
        )

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=str(self.CHECKPOINTS_DIR),  # training checkpoints
            eval_strategy="epoch",  # Evaluate at the end of each epoch.
            save_strategy="epoch",  # Must match eval strategy when load_best_model_at_end=True.
            learning_rate=5e-5,  # Start with a small learning rate
            per_device_train_batch_size=16,  # Batch size per GPU
            per_device_eval_batch_size=16,
            num_train_epochs=3,  # Number of epochs
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

    def evaluate(self) -> dict[str, float]:
        if self._trainer is None:
            raise RuntimeError("Call fine_tune() before evaluate().")

        tokenised_dataset = load_processed("token").map(
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

        # Calculate metrics.
        metric = load("f1")
        return {
            "f1_score": float(
                metric.compute(
                    predictions=predicted_labels[mask].tolist(),
                    references=references[mask].tolist(),
                )["f1"]
            ),
            "accuracy": float((predicted_labels[mask] == references[mask]).mean()),
        }

    def save(self, path: Path | None = None) -> None:
        MODELS_DIR.parent.mkdir(exist_ok=True, parents=True)
        self._model.save(path or MODELS_DIR / "HerbertForTokenClassification")


token_bert = HerbertForTokenClassification()
token_bert.fine_tune()
token_bert.evaluate()
token_bert.save()
