from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def sample_image_tree(tmp_path: Path) -> Path:
    root = tmp_path / "boat_type_classification_dataset"
    for label in ("class_a", "class_b"):
        label_dir = root / label
        label_dir.mkdir(parents=True)
        for index in range(3):
            image_path = label_dir / f"{index}.jpg"
            Image.new("RGB", (32, 32), color=(index * 20, 50, 100)).save(image_path)
    return root


@pytest.fixture
def sample_dataframe(sample_image_tree: Path) -> pd.DataFrame:
    rows = []
    for label_dir in sorted(sample_image_tree.iterdir()):
        for image_path in sorted(label_dir.glob("*.jpg")):
            rows.append(
                {
                    "file_name": image_path.name,
                    "label": label_dir.name,
                    "file_path": str(image_path),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def project_with_zip(tmp_path: Path, sample_image_tree: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    shutil.move(sample_image_tree, project / sample_image_tree.name)
    archive_path = shutil.make_archive(
        str(project / "data" / "boat_type_classification_dataset"),
        "zip",
        root_dir=project,
        base_dir=sample_image_tree.name,
    )
    shutil.rmtree(project / sample_image_tree.name)
    Path(archive_path)
    return project


@pytest.fixture
def sample_label_arrays() -> tuple[np.ndarray, np.ndarray]:
    true_classes = np.array([0, 0, 1, 1, 2])
    predicted_classes = np.array([0, 1, 1, 1, 2])
    return true_classes, predicted_classes
