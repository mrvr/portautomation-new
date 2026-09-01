from pathlib import Path

import pytest

from portautomation.dataset import ensure_dataset


def test_ensure_dataset_uses_existing_folder(sample_image_tree: Path, monkeypatch):
    monkeypatch.setattr("portautomation.config.PROJECT_ROOT", sample_image_tree.parent)
    monkeypatch.setattr("portautomation.config.DATA_DIR", sample_image_tree)
    monkeypatch.setattr("portautomation.config.DATA_ZIP", sample_image_tree.parent / "data" / "missing.zip")
    monkeypatch.setattr("portautomation.config.DATA_ZIP_PARTS", [])

    result = ensure_dataset(sample_image_tree)
    assert result == sample_image_tree


def test_ensure_dataset_extracts_zip(project_with_zip: Path, monkeypatch):
    data_dir = project_with_zip / "boat_type_classification_dataset"
    zip_path = project_with_zip / "data" / "boat_type_classification_dataset.zip"
    monkeypatch.setattr("portautomation.config.PROJECT_ROOT", project_with_zip)
    monkeypatch.setattr("portautomation.config.DATA_DIR", data_dir)
    monkeypatch.setattr("portautomation.config.DATA_ZIP", zip_path)
    monkeypatch.setattr("portautomation.config.DATA_ZIP_PARTS", [])

    result = ensure_dataset(data_dir)
    assert result.exists()
    assert len(list(result.rglob("*.jpg"))) == 6


def test_ensure_dataset_rejects_invalid_force(sample_image_tree: Path):
    with pytest.raises(TypeError):
        ensure_dataset(sample_image_tree, force="yes")
