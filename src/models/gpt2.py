import argparse  # noqa: I001

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    GPT2ForSequenceClassification,
    GPT2ForTokenClassification,
    PreTrainedTokenizerBase,
)

from src.config import GPT2_MODEL, SEED
from src.data import (
    encode_text_classification,
    encode_token_classification,
    get_collator,
    load_processed,
    load_tokenizer,
)
from src.data.tokenization import IGNORE_INDEX, model_kwargs
# gpt-2 (sdadas/polish-gpt2-medium) + adapter lora do klasyfikacji tekstu i tokenow
# sprawdzenie (forward + backward na kilku przykladach): uv run python -m src.models.gpt2 --task token


# c_attn = q, k, v in one Conv1D layer, Conv1D stores weights transposed -> fan_in_fan_out
LORA_CONFIG = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.1, "target_modules": ["c_attn"], "fan_in_fan_out": True}

MODEL_CLASSES = {"text": GPT2ForSequenceClassification, "token": GPT2ForTokenClassification}
TASK_TYPES = {"text": TaskType.SEQ_CLS, "token": TaskType.TOKEN_CLS}
ENCODERS = {"text": encode_text_classification, "token": encode_token_classification}


def build_gpt2_model(task: str, tokenizer: PreTrainedTokenizerBase, model_name: str = GPT2_MODEL, **lora_overrides) -> PeftModel:
    # pad_token_id in config is needed: seq classification takes the last token that is not padding
    model = MODEL_CLASSES[task].from_pretrained(model_name, **model_kwargs(tokenizer))
    # peft adds the new classification head (score / classifier) to modules_to_save, so it is trained too
    lora = LoraConfig(task_type=TASK_TYPES[task], **{**LORA_CONFIG, **lora_overrides})
    return get_peft_model(model, lora)


def smoke_test(task: str, n: int, max_length: int) -> None:
    torch.manual_seed(SEED)
    tokenizer = load_tokenizer(GPT2_MODEL, max_length=max_length)
    model = build_gpt2_model(task, tokenizer)
    print(type(model.base_model.model).__name__)
    model.print_trainable_parameters()

    split = load_processed(task)["train"].select(range(n))
    encoded = split.map(ENCODERS[task], batched=True, fn_kwargs={"tokenizer": tokenizer}, remove_columns=split.column_names)
    batch = get_collator(tokenizer, task)([encoded[i] for i in range(n)])

    if task == "token":
        # every padding and eos position must be ignored in the loss
        special = (batch["input_ids"] == tokenizer.eos_token_id) | (batch["attention_mask"] == 0)
        assert (batch["labels"][special] == IGNORE_INDEX).all()

    model.train()
    output = model(**batch)
    output.loss.backward()
    trained = {name for name, p in model.named_parameters() if p.grad is not None}
    assert trained and all("lora_" in name or "modules_to_save" in name for name in trained)
    print(f"logits {tuple(output.logits.shape)}, loss {output.loss.item():.4f}, gradients in {len(trained)} tensors (lora + head)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["text", "token"], default="text")
    parser.add_argument("--n", type=int, default=4, help="liczba przykladow w batchu testowym")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    smoke_test(args.task, args.n, args.max_length)


if __name__ == "__main__":
    main()
