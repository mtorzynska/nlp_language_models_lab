# embeddingi ostatniego tokenu gpt-2 dla recenzji testowych + interaktywne wykresy
# uv run python -m src.evaluation.gpt2_embeddings --adapter outputs/gpt2-text/adapter
# z adapterem liczy dwa warianty: bazowy gpt-2 (lora wylaczona) i po strojeniu, bez adaptera tylko bazowy

import argparse
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from transformers import GPT2ForSequenceClassification

from src.config import GPT2_MODEL, MAX_LENGTH, PROJECT_ROOT
from src.data import encode_text_classification, get_collator, load_processed, load_tokenizer
from src.data.tokenization import model_kwargs
from src.evaluation.visualization import plot_embeddings

OUTPUT_DIR = PROJECT_ROOT / "outputs"
METHODS = ["umap", "tsne"]
TITLES = {"base": "GPT-2 bazowy", "finetuned": "GPT-2 po strojeniu (LoRA)"}


@torch.no_grad()
def last_token_embeddings(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    # hidden state of the last non-pad token after the final layer norm = input of the score head
    hf_model = model.get_base_model() if isinstance(model, PeftModel) else model
    embeddings, predictions = [], []
    for batch in loader:
        input_ids, attention_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            hidden = hf_model.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            # right padding, so the last real token is at length - 1
            last = hidden[torch.arange(len(hidden), device=device), attention_mask.sum(-1) - 1]
            logits = hf_model.score(last)
        embeddings.append(last.float().cpu())
        predictions.append(logits.argmax(-1).cpu())
    return torch.cat(embeddings).numpy(), torch.cat(predictions).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None, help="katalog adapter/ z treningu (src.training.gpt2)")
    parser.add_argument("--output-dir", type=str, default=None, help="domyslnie <adapter>/../plots")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()

    out_dir = Path(args.output_dir or (Path(args.adapter).parent if args.adapter else OUTPUT_DIR / "gpt2-text") / "plots")
    tokenizer = load_tokenizer(GPT2_MODEL, max_length=args.max_length)
    test = load_processed("text")["test"]
    if args.max_samples:
        test = test.select(range(min(args.max_samples, len(test))))
    encoded = test.map(encode_text_classification, batched=True, fn_kwargs={"tokenizer": tokenizer}, remove_columns=test.column_names)
    loader = DataLoader(encoded, batch_size=args.batch_size, collate_fn=get_collator(tokenizer, "text"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2ForSequenceClassification.from_pretrained(GPT2_MODEL, **model_kwargs(tokenizer))
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device).eval()

    # predictions only for the fine-tuned model, the head of the base model is random
    variants = {}
    if args.adapter:
        variants["finetuned"] = last_token_embeddings(model, loader, device)
        with model.disable_adapter():
            variants["base"] = (last_token_embeddings(model, loader, device)[0], None)
    else:
        variants["base"] = (last_token_embeddings(model, loader, device)[0], None)

    labels = np.asarray(test["label"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant, (embeddings, predictions) in variants.items():
        # how well the classes separate in the original space (cosine), higher = better
        print(f"{variant}: silhouette (cosine) = {silhouette_score(embeddings, labels, metric='cosine'):.4f}")
        if predictions is not None:
            print(f"{variant}: accuracy = {(predictions == labels).mean():.4f}")
        for method in METHODS:
            path = out_dir / f"gpt2_{variant}_{method}.html"
            plot_embeddings(
                embeddings, labels, texts=test["text"], predictions=predictions, method=method,
                title=f"{TITLES[variant]}: reprezentacja ostatniego tokenu", save_path=path,
            )
            print(f"zapisano {path}")

    arrays = {f"{variant}_embeddings": embeddings for variant, (embeddings, _) in variants.items()}
    if "finetuned" in variants:
        arrays["predictions"] = variants["finetuned"][1]
    np.savez_compressed(out_dir / "gpt2_embeddings.npz", ids=np.asarray(test["id"]), labels=labels, **arrays)


if __name__ == "__main__":
    main()
