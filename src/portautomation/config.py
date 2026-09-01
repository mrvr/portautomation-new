"""Shared configuration for the port-operations classifiers."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "boat_type_classification_dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
SPLIT_DIR = OUTPUT_DIR / "splits"

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)
BATCH_SIZE = 32
NUM_CLASSES = 9
IMAGE_SCALING = 1.0 / 255.0
SEED = 42

CNN_EPOCHS = 20
CNN_TEST_SIZE = 0.2
CNN_VAL_SIZE = 0.2
CNN_RANDOM_STATE = 43

MOBILE_EPOCHS = 50
MOBILE_TEST_SIZE = 0.3
MOBILE_VAL_SIZE = 0.2
MOBILE_RANDOM_STATE = 1
MOBILE_EARLY_STOPPING_PATIENCE = 5

CLASS_NAMES = [
    "buoy",
    "cruise_ship",
    "ferry_boat",
    "freight_boat",
    "gondola",
    "inflatable_boat",
    "kayak",
    "paper_boat",
    "sailboat",
]
