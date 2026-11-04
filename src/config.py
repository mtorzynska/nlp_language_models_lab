from pathlib import Path

DATASET_ID = "clarin-pl/polemo2-official"
DATASET_REVISION = "b2ced85046c77edf649d6ac991cc0ef5b8bca13c"

LABEL_NAMES = ["zero", "minus", "plus", "amb"]
LABEL_DESCRIPTIONS = {
    "zero": "neutralny",
    "minus": "negatywny",
    "plus": "pozytywny",
    "amb": "ambiwalentny",
}
ID2LABEL = dict(enumerate(LABEL_NAMES))
LABEL2ID = {name: i for i, name in ID2LABEL.items()}

BERT_MODEL = "allegro/herbert-base-cased"
GPT2_MODEL = "sdadas/polish-gpt2-medium"
MAX_LENGTH = 512

SPLITS = ["train", "validation", "test"]
SEED = 1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = DATA_DIR / "processed"
