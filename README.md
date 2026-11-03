# Modele językowe: HerBERT i polish-gpt2 z PEFT

Strojenie `allegro/herbert-base-cased` i `sdadas/polish-gpt2-medium` (adaptery PEFT) na zbiorze
[PolEmo2](https://huggingface.co/datasets/clarin-pl/polemo2-official): klasyfikacja tekstu i tokenów,
wizualizacja przestrzeni wektorowej, augmentacja danych.

## Środowisko

```bash
uv sync
```

Kod nie jest instalowany jako pakiet, moduły są uruchamiane z katalogu głównego repo jako `src.*`.

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
  Ustawienia można nadpisać argumentami, np. `build_gpt2_model("text", tokenizer, r=8)`.
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

Wyniki w `outputs/gpt2-{task}/` (inna nazwa przez `--run-name`, pełna ścieżka przez `--output-dir`):
- `adapter/`: wagi LoRA i głowicy klasyfikacyjnej + tokenizer (kilka MB, bez modelu bazowego),
- `results.json`: metryki walidacji i testu, macierz pomyłek, użyte argumenty, rozmiar zbioru treningowego,
- `test_report.txt`: precision/recall/F1 dla każdej klasy,
- `test_predictions.parquet`: zbiór testowy z kolumną `prediction` (dla tokenów lista predykcji dla kolejnych słów;
  słowa ucięte przez limit 512 tokenów nie mają predykcji).

Klasyfikacja tokenów to ten sam skrypt z `--task token`. Baseline bez LoRA (`--mode head --lr 1e-3`) to
zamrożony GPT-2 z trenowaną tylko głowicą `score`/`classifier`. Zamiast `adapter/` zapisywane są wtedy
same wagi głowicy (`head/head.pt`). W Colabie (T4) trening idzie z batchem 4 i akumulacją 4, czyli
z tym samym efektywnym batchem 16.

### Embeddingi ostatniego tokenu

```bash
uv run python -m src.evaluation.gpt2_embeddings --adapter outputs/gpt2-text/adapter
```

Reprezentacją recenzji jest stan ukryty ostatniego tokenu (wejście głowicy klasyfikacyjnej), liczony dla
GPT-2 bazowego i po strojeniu. Wykresy UMAP i t-SNE trafiają do `outputs/gpt2-text/plots/`.

### Augmentacja generatywna

```bash
uv run python -m src.data.gpt2_augmentation --max-samples 2000
```

Losowane są recenzje treningowe (ze stratyfikacją po etykiecie). Połowa słów recenzji, ale maksymalnie 40, jest
promptem. GPT-2 (`GPT2LMHeadModel`, bez strojenia) dopisuje do 80 tokenów, a urwane ostatnie zdanie jest obcinane.
Nowy tekst dostaje etykietę recenzji źródłowej. Odrzucane są generacje krótsze niż 3 słowa i kopie istniejących tekstów.
`no_repeat_ngram_size=3` ogranicza zapętlanie.

| Wariant | temperature | top-k | top-p | Efekt |
|---|---|---|---|---|
| `conservative` | 0.7 | 50 | 0.9 | tekst bliski typowym recenzjom, mniej różnorodny |
| `diverse` | 1.0 | wyłączone | 0.95 | większa różnorodność, więcej szumu i odejść od tematu |

- **temperature** spłaszcza (>1) albo wyostrza (<1) rozkład następnego tokenu.
- **top-k** losuje tylko spośród k najbardziej prawdopodobnych tokenów.
- **top-p** losuje z najmniejszego zbioru tokenów o łącznym prawdopodobieństwie p.

Wyniki trafiają do `data/augmented/text/`, w tym samym formacie co augmentacja HerBERTa (`id`, `text`, `label`
+ parametry):
- `augmented_gpt2_{conservative,diverse}_train.json`,
- `augmented_gpt2_combined_train.json`,
- `augmented_gpt2_examples.md`: przykłady do sprawozdania.

Augmentacja dotyczy tylko klasyfikacji tekstu. Wygenerowane zdania nie mają etykiet zdań, więc nie da się
zbudować z nich przykładów do klasyfikacji tokenów.

### Trening na augmentacji i porównanie

```bash
uv run python -m src.training.gpt2 --task text --run-name gpt2-text-augmented \
    --augmented data/augmented/text/augmented_gpt2_combined_train.json data/augmented/text/augmented_combined_train.json
uv run python -m src.evaluation.gpt2_comparison
```

Tak jak w części HerBERTa, z `--augmented` zbiorem treningowym są tylko przykłady z podanych plików. Walidacja
i test zostają oryginalne. Z `--keep-original` augmentacja jest dopisywana do oryginalnego zbioru treningowego.
Można podać kilka plików naraz, np. generację GPT-2 i `[MASK]` z HerBERTa, czyli jeden model na połączonych
augmentacjach. `gpt2_comparison` zbiera wszystkie `outputs/*/results.json` w jedną tabelę.
Tabela zawiera macro F1, F1 per klasa i różnicę względem przebiegu bez augmentacji, a zapisuje się do
`outputs/gpt2_comparison.md`.

### Colab

`notebooks/gpt2_colab.ipynb`: klasyfikacja tekstu i tokenów, embeddingi, augmentacja i porównanie dla GPT-2, z zapisem wyników na Dysku Google.

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
