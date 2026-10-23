# wspolna tokenizacja

from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
)

from src.config import ID2LABEL, LABEL2ID, MAX_LENGTH

IGNORE_INDEX = -100
DECODER_MODEL_TYPES = {"gpt2"}


def load_tokenizer(model_name: str, max_length: int = MAX_LENGTH) -> PreTrainedTokenizerBase:
    is_decoder = AutoConfig.from_pretrained(model_name).model_type in DECODER_MODEL_TYPES
    # gpt-2 bpe needs add_prefix_space for is_split_into_words
    extra = {"add_prefix_space": True} if is_decoder else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **extra)
    # gpt-2 has no pad token -> use eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = min(tokenizer.model_max_length, max_length)
    return tokenizer


def appends_eos(tokenizer: PreTrainedTokenizerBase) -> bool:
    # true if tokenizer doesnt close the sequence with a special token itself (gpt-2)
    return tokenizer("a")["input_ids"][-1] not in tokenizer.all_special_ids


def model_kwargs(tokenizer: PreTrainedTokenizerBase) -> dict:
    # kwargs for *ForSequenceClassification / *ForTokenClassification.from_pretrained
    return {
        "num_labels": len(ID2LABEL),
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "pad_token_id": tokenizer.pad_token_id,
    }


def encode_text_classification(batch: dict, tokenizer: PreTrainedTokenizerBase) -> dict:
    # no eos here: gpt-2 takes the last non-pad token and pad == eos, so it would be skipped anyway
    encoded = tokenizer(batch["text"], truncation=True)
    encoded["labels"] = batch["label"]
    return encoded


def encode_token_classification(
    batch: dict, tokenizer: PreTrainedTokenizerBase, label_all_subwords: bool = False
) -> dict:
    # align word labels to subtokens, by default only the first subtoken gets the label
    add_eos = appends_eos(tokenizer)
    encoded = tokenizer(
        batch["words"],
        is_split_into_words=True,
        truncation=True,
        max_length=tokenizer.model_max_length - int(add_eos),
    )

    all_labels = []
    for i, word_labels in enumerate(batch["labels"]):
        labels, previous = [], None
        for word_id in encoded.word_ids(i):
            if word_id is None:
                labels.append(IGNORE_INDEX)
            elif word_id != previous or label_all_subwords:
                labels.append(word_labels[word_id])
            else:
                labels.append(IGNORE_INDEX)
            previous = word_id
        # gpt-2: append eos and mask it with -100
        if add_eos:
            encoded["input_ids"][i].append(tokenizer.eos_token_id)
            encoded["attention_mask"][i].append(1)
            if "token_type_ids" in encoded:
                encoded["token_type_ids"][i].append(0)
            labels.append(IGNORE_INDEX)
        all_labels.append(labels)

    encoded["labels"] = all_labels
    return encoded


def get_collator(tokenizer: PreTrainedTokenizerBase, task: str):
    # dynamic padding, for token task labels are padded with -100
    if task == "text":
        return DataCollatorWithPadding(tokenizer)
    return DataCollatorForTokenClassification(tokenizer, label_pad_token_id=IGNORE_INDEX)
