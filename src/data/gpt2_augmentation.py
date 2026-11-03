# augmentacja generatywna gpt-2: poczatek recenzji treningowej jest promptem, gpt-2 dopisuje reszte,
# a nowy tekst dostaje etykiete recenzji zrodlowej. dwa warianty probkowania (temperature, top-k, top-p)
# uv run python -m src.data.gpt2_augmentation --max-samples 2000
# wynik: data/augmented/text/augmented_gpt2_{wariant}_train.json + combined (format jak augmentacja herberta)

import argparse
import json
import textwrap
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, set_seed

from src.config import DATA_DIR, GPT2_MODEL, LABEL_NAMES, SEED
from src.data.prepare import load_processed, normalize_text, subsample_train

# conservative stays close to typical review language, diverse gives more varied but noisier texts
SAMPLING = {
    "conservative": {"temperature": 0.7, "top_k": 50, "top_p": 0.9},
    "diverse": {"temperature": 1.0, "top_k": 0, "top_p": 0.95},
}
SENTENCE_END = ".!?"


def make_prompt(text: str, prompt_ratio: float, max_prompt_words: int) -> str | None:
    words = text.split()
    # too short to split into a prompt and a continuation
    if len(words) < 6:
        return None
    return " ".join(words[: min(max_prompt_words, max(3, int(len(words) * prompt_ratio)))])


def trim_to_sentence(continuation: str) -> str:
    # generation stops at max_new_tokens, so the last sentence is usually unfinished
    end = max(continuation.rfind(char) for char in SENTENCE_END)
    return continuation[: end + 1] if end > 0 else continuation


@torch.no_grad()
def generate(
    model, tokenizer: PreTrainedTokenizerBase, prompts: list[str], sampling: dict, max_new_tokens: int, batch_size: int
) -> list[str]:
    # prompts sorted by length -> less padding, results go back to the original order
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    continuations = [""] * len(prompts)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        output = model.generate(
            **batch, do_sample=True, max_new_tokens=max_new_tokens, no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id, **sampling,
        )
        # left padding: new tokens start right after the prompt
        for i, text in zip(idx, tokenizer.batch_decode(output[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)):
            continuations[i] = text
        print(f"{min(start + batch_size, len(order))}/{len(order)}", end="\r")
    return continuations


def augment(train, name: str, model, tokenizer, args: argparse.Namespace) -> list[dict]:
    set_seed(SEED)
    sampling = SAMPLING[name]
    prompts = [make_prompt(text, args.prompt_ratio, args.max_prompt_words) for text in train["text"]]
    keep = [i for i, prompt in enumerate(prompts) if prompt]
    continuations = generate(model, tokenizer, [prompts[i] for i in keep], sampling, args.max_new_tokens, args.batch_size)

    seen, rows = set(train["text"]), []
    for i, continuation in zip(keep, continuations):
        continuation = trim_to_sentence(continuation)
        # drop empty continuations and copies of existing reviews
        text = normalize_text(prompts[i] + continuation)
        if len(continuation.split()) < 3 or text in seen:
            continue
        seen.add(text)
        rows.append({
            "id": f"gpt2_{name}_{train['id'][i]}",
            "text": text,
            "label": train["label"][i],
            "source_id": train["id"][i],
            "prompt": prompts[i],
            "sampling": name,
            **sampling,
        })
    return rows


def examples_table(train, variants: dict[str, list[dict]], n: int = 3) -> str:
    # a few examples per variant for the report
    source = dict(zip(train["id"], train["text"]))
    lines = ["| wariant | etykieta | oryginal | po augmentacji (prompt + generacja) |", "|---|---|---|---|"]
    for name, rows in variants.items():
        for row in rows[:n]:
            original, generated = (textwrap.shorten(t, 300, placeholder=" …").replace("|", "\\|") for t in (source[row["source_id"]], row["text"]))
            lines.append(f"| {name} | {LABEL_NAMES[row['label']]} | {original} | {generated} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=2000, help="recenzje train (losowane ze stratyfikacja), 0 = wszystkie")
    parser.add_argument("--sampling", nargs="+", choices=list(SAMPLING), default=list(SAMPLING))
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--prompt-ratio", type=float, default=0.5, help="jaka czesc slow recenzji idzie do promptu")
    parser.add_argument("--max-prompt-words", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR / "augmented" / "text"))
    args = parser.parse_args()

    dataset = load_processed("text")
    train = subsample_train(dataset, args.max_samples)["train"] if args.max_samples else dataset["train"]

    cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(GPT2_MODEL, dtype=torch.float16 if cuda else torch.float32)
    model.to("cuda" if cuda else "cpu").eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = {}
    for name in args.sampling:
        print(f"{name}: {SAMPLING[name]}")
        variants[name] = augment(train, name, model, tokenizer, args)
        path = out_dir / f"augmented_gpt2_{name}_train.json"
        path.write_text(json.dumps(variants[name], indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"zapisano {len(variants[name])} przykladow do {path}")

    combined = [row for rows in variants.values() for row in rows]
    (out_dir / "augmented_gpt2_combined_train.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "augmented_gpt2_examples.md").write_text(examples_table(train, variants), encoding="utf-8")
    print(f"razem {len(combined)} przykladow, przyklady do sprawozdania: {out_dir / 'augmented_gpt2_examples.md'}")


if __name__ == "__main__":
    main()
