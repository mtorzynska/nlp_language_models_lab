# Modele językowe: HerBERT i polish-gpt2 z PEFT

Strojenie `allegro/herbert-base-cased` i `sdadas/polish-gpt2-medium` (adaptery PEFT) na zbiorze
[PolEmo2](https://huggingface.co/datasets/clarin-pl/polemo2-official): klasyfikacja tekstu i tokenów,
wizualizacja przestrzeni wektorowej, augmentacja danych.

## Środowisko

```bash
uv sync
```

Kod nie jest instalowany jako pakiet. Moduły uruchamiasz z katalogu głównego repo jako `src.*`.

## Dane

```bash
uv run python -m src.data.prepare
```

Skrypt pobiera PolEmo2 z Hubu (przypięta rewizja w `src/config.py`), czyści dane
i zapisuje do `data/processed/` (katalog jest w `.gitignore`, każdy generuje dane u siebie):

| Zadanie | Plik | Przykład | Etykieta |
|---|---|---|---|
| klasyfikacja tekstu | `text/{split}.parquet` | `id`, `text` | sentyment recenzji |
| klasyfikacja tokenów | `token/{split}.parquet` | `id`, `words` | sentyment zdania, w którym stoi słowo |

Etykiety: `zero` (neutralny), `minus` (negatywny), `plus` (pozytywny), `amb` (ambiwalentny).
Statystyki trafiają do `data/processed/metadata.json`.

Szczegóły:
- **Podział**: oficjalny train/validation/test recenzji, wspólny dla obu zadań. Oficjalny podział
  zdań w PolEmo2 jest niezależny od podziału recenzji, więc zdania służą tylko jako słownik etykiet.
- **Czyszczenie**: normalizacja Unicode (NFC) i białych znaków, usuwanie pustych tekstów, duplikatów
  i recenzji z validation/test, które występują we wcześniejszych splitach.
- **Etykiety tokenów**: recenzja jest dzielona na oznaczone zdania (programowanie dynamiczne).
  Zostają tylko recenzje, które da się w całości pokryć (~68%).
- **Mniejszy zbiór treningowy**: `--max-train N` losuje N recenzji ze stratyfikacją po etykiecie.

## Użycie w treningu

```bash
# fine-tuning z CLI (pełny zbiór lub próbka, zadanie tekst/token)
uv run python -m src.models.train --task text                # klasyfikacja recenzji
uv run python -m src.models.train --task token               # klasyfikacja tokenów
uv run python -m src.models.train --task text --max-train 1000 --epochs 1  # szybki test

# Notebook do uruchomienia w Google Colab (T4 GPU): notebooks/colab_training.ipynb
```

```python
from src.data import load_processed, load_tokenizer, encode_token_classification, get_collator
from src.data.tokenization import model_kwargs

tokenizer = load_tokenizer("sdadas/polish-gpt2-medium")
dataset = load_processed("token").map(
    encode_token_classification, batched=True, fn_kwargs={"tokenizer": tokenizer},
    remove_columns=["id", "words", "labels"],
)
collator = get_collator(tokenizer, "token")
# AutoModelForTokenClassification.from_pretrained(name, **model_kwargs(tokenizer))
```

Ten sam kod działa dla obu modeli. Różnice obsługuje `tokenization.py`:
- GPT-2 dostaje `<eos>` jako token paddingu, a `pad_token_id` trafia do konfiguracji modelu przez `model_kwargs`.
- W klasyfikacji tokenów GPT-2 dostaje dopisany `<eos>`. Tokeny specjalne, `<eos>`, padding
  i kolejne subtokeny słowa mają etykietę `-100`.
- Padding jest dynamiczny (collator), a długość jest obcinana do 512 tokenów.
