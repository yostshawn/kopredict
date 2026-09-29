"""Paths and loaders for metadata CSVs that list public inputs.

Which DepMap files to fetch is metadata/default_depmap_files_to_download_and_process.csv.
Publication workbooks are metadata/additional_datasets_to_download.csv.
Signed DepMap links live in metadata/20260920_DepMap_available_downloads.csv.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

DEPMAP_DOWNLOADS_CSV = "20260920_DepMap_available_downloads.csv"
DEPMAP_FILES_CSV = "default_depmap_files_to_download_and_process.csv"
ADDITIONAL_DATASETS_CSV = "additional_datasets_to_download.csv"

def metadata_dir() -> Path:
    """Directory of the DepMap catalog, the file list, and the extra datasets."""
    candidates = [
        Path(__file__).resolve().parents[1] / "metadata",
        Path.cwd() / "metadata",
    ]
    for path in candidates:
        if (path / DEPMAP_DOWNLOADS_CSV).is_file() and (path / DEPMAP_FILES_CSV).is_file():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"metadata directory not found. Looked in: {searched}")


def _read_metadata_csv(name: str) -> pd.DataFrame:
    path = metadata_dir() / name
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def load_depmap_files() -> pd.DataFrame:
    """Rows of release and filename to pull from the DepMap download catalog."""
    frame = _read_metadata_csv(DEPMAP_FILES_CSV)
    for column in ("release", "filename"):
        if column not in frame.columns:
            raise ValueError(f"metadata/{DEPMAP_FILES_CSV} is missing column {column!r}")
    return frame


def load_additional() -> pd.DataFrame:
    """Publication files that are not rows in the DepMap catalog."""
    frame = _read_metadata_csv(ADDITIONAL_DATASETS_CSV)
    for column in ("id", "local_filename", "url", "citation", "download"):
        if column not in frame.columns:
            raise ValueError(f"metadata/{ADDITIONAL_DATASETS_CSV} is missing column {column!r}")
    return frame


def additional_record(record_id: str) -> pd.Series:
    frame = load_additional()
    hit = frame[frame["id"] == record_id]
    if hit.empty:
        raise KeyError(f"metadata/{ADDITIONAL_DATASETS_CSV} has no id {record_id!r}")
    return hit.iloc[0]
