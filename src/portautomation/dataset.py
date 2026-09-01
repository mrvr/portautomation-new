"""Dataset archive extraction utilities."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from portautomation.config import DATA_DIR, DATA_ZIP, DATA_ZIP_PARTS, PROJECT_ROOT

logger = logging.getLogger(__name__)


def _dataset_is_ready(data_dir: Path) -> bool:
    if not data_dir.is_dir():
        return False
    return any(path.suffix.lower() == ".jpg" for path in data_dir.rglob("*.jpg"))


def _reassemble_zip(parts: list[Path], destination: Path) -> None:
    with destination.open("wb") as output:
        for part in parts:
            output.write(part.read_bytes())


def _resolve_archive() -> Path:
    if DATA_ZIP.exists():
        return DATA_ZIP

    missing_parts = [part for part in DATA_ZIP_PARTS if not part.exists()]
    if missing_parts:
        missing = ", ".join(str(part) for part in missing_parts)
        raise FileNotFoundError(
            f"Dataset archive not found. Expected {DATA_ZIP} or zip parts under data/. "
            f"Missing parts: {missing}"
        )

    archive_path = PROJECT_ROOT / "data" / ".boat_type_classification_dataset.zip"
    logger.info("Reassembling dataset archive from %s parts", len(DATA_ZIP_PARTS))
    _reassemble_zip(DATA_ZIP_PARTS, archive_path)
    return archive_path


def ensure_dataset(
    data_dir: Path | str = DATA_DIR,
    force: bool = False,
) -> Path:
    """Extract the boat dataset from zip archives when needed."""
    data_dir = Path(data_dir)

    if _dataset_is_ready(data_dir) and not force:
        logger.info("Dataset already available at %s", data_dir)
        return data_dir

    archive_path = _resolve_archive()
    temp_archive = archive_path.name.startswith(".")
    logger.info("Extracting dataset from %s", archive_path.name)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(PROJECT_ROOT)

    if temp_archive:
        archive_path.unlink(missing_ok=True)

    if not _dataset_is_ready(data_dir):
        raise RuntimeError(f"Dataset extraction completed but {data_dir} is still missing images.")

    logger.info("Dataset ready at %s", data_dir)
    return data_dir
