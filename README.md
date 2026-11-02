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

## Część BERT (HerBERT)

### Model
- `allegro/herbert-base-cased` — polski model BERT bazowy (cased).
- LoRA (PEFT) 적용 на moduły uwagi: `target_modules=["query", "value"]` (tylko query i value, nie `c_attn` ani embeddingu).
- Klasyfikacja tekstu: `BertForSequenceClassification` + LoRA.
- Klasyfikacja tokenów: `BertForTokenClassification` (z `AutoModelForTokenClassification`) + LoRA.
- Głowa klasyfikacyjna (`classifier`) jest trenowana (`modules_to_save=["classifier"]`).

### Trening
```bash
uv run python -m src.models.train --task text    # klasyf. recenzji (tekst)
uv run python -m src.models.train --task token   # klasyf. tokenów
```
Wyniki (metryki, macierz pomyłek) trafiają do `data/HerbertFor{Sequence,Token}Classification/<run>/`.

### Embedingi [CLS] (Etap 6)
Wektory `[CLS]` z bazowego HerBERTa wyciągane są dla każdej recenzji ze zbioru testowego i wizualizowane dwoma interaktywnymi wykresami (UMAP i t-SNE) w jednym pliku HTML:

```bash
uv run python -m src.models.embeddings --task text --max-samples 2000
```
Wynik: `data/HerbertEmbeddings/<run>/cls_embeddings.html` (zakładki UMAP/t-SNE) oraz `cls_embeddings.npz` (surowe wektory i etykiety).

### Augmentacja przez [MASK] / fill-mask (Etap 7)
Zamiast generowania (jak GPT-2 z `temperature`/`top_p`/`top_k`), BERT augmentuje dane przez iteracyjne podmiany słów na token `[MASK]` z uzupełnieniem przez `fill-mask` (pipeline HerBERT `BertForMaskedLM`). Przygotowano dwa warianty (różny `mask_ratio`):

```bash
uv run python -m src.data.augmentation --task text --split train --max-samples 2000
```
Wyniki: `data/augmented/text/augmented_mask_10pct_train.json`, `augmented_mask_20pct_train.json` oraz plik łączny `augmented_combined_train.json`.

Przykładowe augmentacje ([MASK] → zastępstwo HerBERTa):

| # | mask_ratio | Etykieta | Tekst oryginalny (fragment) | Tekst po augmentacji (fragment) |
|---|-----------|----------|-----------------------------|---------------------------------|
| 1 | 0.10 | plus | `Gra się bardzo dobrze. Głównie za sprawą pięknie odwzorowanego z filmu Śródziemia.` | `Gra się bardzo dobrze. Głównie za sprawą pięknie odwzorowanego z filmu Śródziemia. Ścieżka dźwiękowa prosto z filmu powoduje uśmiech na twarzy.` |
| 2 | 0.10 | minus | `Robi bardzo dobre wrażenie, ma dobre podejście do pacjentek, natomiast jest typowym lekarzem...` | `Robi bardzo dobre wrażenie, ma dobre podejście do pacjentek, natomiast jest typowym lekarzem, który ponad wszystko stawia nie na dobro pacjentki, tylko na ilość pieniędzy jakie może uzyskać` |
| 3 | 0.20 | amb | `Hotel zasługuje na swoje 4 miejsce jeżeli chodzi o wyposażenie i komfort...` | `Hotel zasługuje na swoje 4 miejsce jezeli chodzi o wyposażenie i komfort. Zalety dalej ale zaczne od tego, że nie jest to miejsce dla tych co lubia spacery i spacery wzdłóż morza.` |

### Raport
Opis części BERT w sprawozdaniu obejmuje: wybór modelu (`herbert-base-cased`), konfigurację LoRA (`["query", "value"]`), oba zadania klasyfikacji (tekst + tokeny), wyniki treningu (tabela przed/po augmentacji), embedingi `[CLS]` i przykłady augmentacji `[MASK]` (powyżej).
