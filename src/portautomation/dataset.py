"""Dataset archive extraction utilities."""

from __future__ import annotations

import logging
import warnings
import zipfile
from pathlib import Path

from portautomation import config
from portautomation.validation import validate_directory

logger = logging.getLogger(__name__)


def _dataset_is_ready(data_dir: Path) -> bool:
    if not data_dir.is_dir():
        return False
    return any(path.suffix.lower() == ".jpg" for path in data_dir.rglob("*.jpg"))


def _reassemble_zip(parts: list[Path], destination: Path) -> None:
    if not parts:
        raise ValueError("parts must not be empty")
    for part in parts:
        if not part.is_file():
            raise FileNotFoundError(f"Archive part not found: {part}")
    with destination.open("wb") as output:
        for part in parts:
            output.write(part.read_bytes())


def _resolve_archive() -> Path:
    if config.DATA_ZIP.exists():
        return config.DATA_ZIP

    missing_parts = [part for part in config.DATA_ZIP_PARTS if not part.exists()]
    if missing_parts:
        missing = ", ".join(str(part) for part in missing_parts)
        raise FileNotFoundError(
            f"Dataset archive not found. Expected {config.DATA_ZIP} or zip parts under data/. "
            f"Missing parts: {missing}"
        )

    archive_path = config.PROJECT_ROOT / "data" / ".boat_type_classification_dataset.zip"
    logger.info("Reassembling dataset archive from %s parts", len(config.DATA_ZIP_PARTS))
    _reassemble_zip(config.DATA_ZIP_PARTS, archive_path)
    return archive_path


def ensure_dataset(
    data_dir: Path | str = config.DATA_DIR,
    force: bool = False,
) -> Path:
    """Extract the boat dataset from zip archives when needed."""
    data_dir = validate_directory(data_dir, "data_dir", must_exist=False)
    if not isinstance(force, bool):
        raise TypeError("force must be a bool")

    if _dataset_is_ready(data_dir) and not force:
        logger.info("Dataset already available at %s", data_dir)
        return data_dir

    if force and data_dir.exists():
        warnings.warn(
            "force=True: re-extracting dataset even if files already exist.",
            UserWarning,
            stacklevel=2,
        )
        logger.warning("force=True: re-extracting dataset even if files already exist")

    archive_path = _resolve_archive()
    temp_archive = archive_path.name.startswith(".")
    logger.info("Extracting dataset from %s", archive_path.name)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(config.PROJECT_ROOT)

    if temp_archive:
        archive_path.unlink(missing_ok=True)

    if not _dataset_is_ready(data_dir):
        raise RuntimeError(f"Dataset extraction completed but {data_dir} is still missing images.")

    logger.info("Dataset ready at %s", data_dir)
    return data_dir
