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

## Model GPT-2 z LoRA

```python
from src.data import load_tokenizer
from src.models.gpt2 import build_gpt2_model

tokenizer = load_tokenizer("sdadas/polish-gpt2-medium")
model = build_gpt2_model("token", tokenizer)   # "text" -> GPT2ForSequenceClassification
```

- LoRA na `c_attn` (warstwa, która w GPT-2 liczy q, k, v), `r=16`, `alpha=32`, `dropout=0.1`.
  Ustawienia nadpisujesz argumentami, np. `build_gpt2_model("text", tokenizer, r=8)`.
- Głowica klasyfikacyjna jest trenowana razem z adapterem, reszta modelu jest zamrożona.
- `pad_token_id` trafia do konfiguracji modelu, bo `GPT2ForSequenceClassification` szuka po nim
  ostatniego prawdziwego tokenu.

Sprawdzenie, czy model się ładuje i liczy gradienty (forward + backward na 4 przykładach):

```bash
uv run python -m src.models.gpt2 --task text
uv run python -m src.models.gpt2 --task token
```

## Trening GPT-2

```bash
uv run python -m src.training.gpt2 --task text
```

Trening z `Trainer`: AdamW, `lr=2e-4`, 3 epoki, batch 8 × gradient accumulation 2 (efektywnie 16),
warmup 6%, długość do 512 tokenów. Na GPU włącza się bf16 albo fp16.
Po każdej epoce liczona jest walidacja. Na koniec ładowany jest checkpoint z najlepszym macro F1
na walidacji i oceniany na zbiorze testowym.

Wyniki w `outputs/gpt2-text/`:
- `adapter/`: wagi LoRA i głowicy klasyfikacyjnej + tokenizer (kilka MB, bez modelu bazowego),
- `results.json`: metryki walidacji i testu, macierz pomyłek, użyte argumenty,
- `test_report.txt`: precision/recall/F1 dla każdej klasy,
- `test_predictions.parquet`: `id`, `text`, `label`, `prediction` dla zbioru testowego (do wykresu).

Gdy brakuje pamięci GPU, zmniejsz batch i zwiększ akumulację, żeby efektywny batch został ten sam,
albo włącz gradient checkpointing:

```bash
uv run python -m src.training.gpt2 --task text --batch-size 4 --grad-accum 4 --gradient-checkpointing
```

Szybki test całego przebiegu na CPU (kilka minut):

```bash
uv run python -m src.training.gpt2 --task text --max-train-samples 16 --max-eval-samples 8 --epochs 1 --max-length 64
```

## Metryki i wykres

```python
from src.evaluation import compute_metrics, evaluation_report, logits_to_predictions, plot_embeddings

trainer = Trainer(
    ...,
    compute_metrics=compute_metrics,                       # ta sama funkcja dla text i token
    preprocess_logits_for_metrics=logits_to_predictions,   # opcjonalnie: mniej pamięci w zadaniu token
)

output = trainer.predict(dataset["test"])
report, confusion = evaluation_report(output.predictions, output.label_ids)

# embeddings: (n, dim) z [CLS] (BERT) albo ostatniego tokenu (GPT-2), liczone osobno dla modelu
plot_embeddings(embeddings, test["label"], texts=test["text"], predictions=preds,
                method="umap", save_path="outputs/plots/herbert_text.html")
```

- `compute_metrics`: accuracy, macro F1, weighted F1 i F1 dla każdej klasy. Pozycje z etykietą `-100` są pomijane.
- `evaluation_report`: raport precision/recall/F1 i macierz pomyłek do oceny na zbiorze testowym.
- `plot_embeddings`: redukcja do 2D (`umap`, `tsne` albo `pca`) i interaktywny wykres plotly zapisany do HTML.
  Po najechaniu na punkt widać tekst recenzji, etykietę i predykcję. Błędne predykcje mają puste znaczniki.

Notebooki trzymaj w katalogu głównym repo, wtedy `import src...` działa bez zmian. Jeśli notebook
leży w podfolderze, dodaj na początku `sys.path.insert(0, "..")`.
