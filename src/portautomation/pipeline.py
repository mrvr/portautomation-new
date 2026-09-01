"""Run CNN and/or MobileNetV2 training pipelines."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from portautomation.config import (
    CNN_EPOCHS,
    CNN_RANDOM_STATE,
    CNN_TEST_SIZE,
    CNN_VAL_SIZE,
    DATA_DIR,
    FIGURES_DIR,
    MOBILE_EPOCHS,
    MOBILE_RANDOM_STATE,
    MOBILE_TEST_SIZE,
    MOBILE_VAL_SIZE,
    MODELS_DIR,
    SEED,
    SPLIT_DIR,
)
from portautomation.data import (
    build_split_datasets,
    compute_class_weights,
    load_images_to_dataframe,
    seed_everything,
    split_dataframe,
    summarize_labels,
)
from portautomation.dataset import ensure_dataset
from portautomation.evaluate import (
    build_confusion_matrix,
    collect_predictions,
    evaluate_model,
    print_classification_report,
)
from portautomation.models import build_cnn, build_mobilenet, compile_classifier
from portautomation.train import save_model, train_cnn, train_mobilenet
from portautomation.visualize import plot_confusion_matrix, plot_training_history

logger = logging.getLogger("portautomation")


def _prepare_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def _run_pipeline(
    name: str,
    df,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
    builder,
    trainer,
    epochs: int,
    split_subdir: str,
) -> dict[str, float]:
    train_df, val_df, test_df = split_dataframe(
        df, test_size=test_size, random_state=random_state, val_size=val_size
    )
    logger.info(
        "%s splits — train=%s val=%s test=%s",
        name,
        train_df.shape,
        val_df.shape,
        test_df.shape,
    )

    train_ds, val_ds, test_ds, class_names = build_split_datasets(
        train_df, val_df, test_df, SPLIT_DIR / split_subdir
    )
    class_weights = compute_class_weights(train_df["label"].value_counts(), class_names)

    model = compile_classifier(builder())
    model.summary(print_fn=logger.info)
    history = trainer(model, train_ds, val_ds, class_weight=class_weights, epochs=epochs)

    plot_training_history(
        history,
        title=f"{name} Training History",
        output_path=FIGURES_DIR / f"{split_subdir}_history.png",
    )
    metrics = evaluate_model(model, test_ds)
    true_classes, predicted_classes = collect_predictions(model, test_ds)
    cm = build_confusion_matrix(true_classes, predicted_classes)
    plot_confusion_matrix(
        cm,
        class_names=class_names,
        title=f"{name} Confusion Matrix",
        output_path=FIGURES_DIR / f"{split_subdir}_confusion_matrix.png",
    )
    print_classification_report(true_classes, predicted_classes, class_names)
    save_model(model, MODELS_DIR / f"{split_subdir}.keras")
    return metrics


def run_cnn(df) -> dict[str, float]:
    return _run_pipeline(
        "CNN",
        df,
        test_size=CNN_TEST_SIZE,
        val_size=CNN_VAL_SIZE,
        random_state=CNN_RANDOM_STATE,
        builder=build_cnn,
        trainer=train_cnn,
        epochs=CNN_EPOCHS,
        split_subdir="cnn",
    )


def run_mobilenet(df) -> dict[str, float]:
    return _run_pipeline(
        "MobileNetV2",
        df,
        test_size=MOBILE_TEST_SIZE,
        val_size=MOBILE_VAL_SIZE,
        random_state=MOBILE_RANDOM_STATE,
        builder=build_mobilenet,
        trainer=train_mobilenet,
        epochs=MOBILE_EPOCHS,
        split_subdir="mobilenet",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train boat-type classifiers from the Automating Port Operations project."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing class-labeled boat images.",
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "mobilenet", "both"],
        default="both",
        help="Which model pipeline to run.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    data_dir = ensure_dataset(Path(args.data_dir))

    _prepare_dirs()
    seed_everything(SEED)

    df = load_images_to_dataframe(data_dir)
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Class distribution:\n%s", summarize_labels(df))

    results: dict[str, dict[str, float]] = {}
    if args.model in {"cnn", "both"}:
        results["cnn"] = run_cnn(df)
    if args.model in {"mobilenet", "both"}:
        results["mobilenet"] = run_mobilenet(df)

    for name, metrics in results.items():
        logger.info(
            "%s test results: loss=%.4f accuracy=%.4f precision=%.4f recall=%.4f",
            name,
            metrics["loss"],
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
        )
