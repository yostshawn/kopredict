"""Paths and loaders for metadata CSVs that list public inputs.

Which DepMap files to fetch is metadata/depmap_files.csv. Publication workbooks
are metadata/additional.csv. Signed DepMap links live in metadata/downloads.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

NO_CAPTCHA_CATALOG_URL = "https://depmap.org/portal/api/no-captcha/download/files"
# Portal catalog with signed URLs. A browser save replaces metadata/downloads.csv.
CATALOG_URL = "https://depmap.org/portal/api/download/files"

FEATURES_FILENAME = "final_depmap_features_26Q1.parquet.gz"

HRD_SCORE_CUTOFF = 42
ZSCORE_CUTOFF = 1.64
LOF_CLASSES = (
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "Nonstop_Mutation",
)

DEFAULT_DEPMAP_RELEASE = "26Q1"
# In metadata/depmap_files.csv this release token follows --depmap-release.
DEPMAP_RELEASE_TOKEN = "{depmap}"


def metadata_dir() -> Path:
    """Directory of downloads.csv, depmap_files.csv, and additional.csv."""
    candidates = [
        Path(__file__).resolve().parents[2] / "metadata",
        Path.cwd() / "metadata",
    ]
    for path in candidates:
        if (path / "downloads.csv").is_file() and (path / "depmap_files.csv").is_file():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"metadata directory not found. Looked in: {searched}")


def _read_metadata_csv(name: str) -> pd.DataFrame:
    path = metadata_dir() / name
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def load_depmap_files() -> pd.DataFrame:
    """Rows of release and filename to pull from metadata/downloads.csv."""
    frame = _read_metadata_csv("depmap_files.csv")
    for column in ("release", "filename"):
        if column not in frame.columns:
            raise ValueError(f"metadata/depmap_files.csv is missing column {column!r}")
    return frame


def load_additional() -> pd.DataFrame:
    """Publication files that are not rows in the DepMap catalog."""
    frame = _read_metadata_csv("additional.csv")
    for column in ("id", "local_filename", "url", "citation", "download"):
        if column not in frame.columns:
            raise ValueError(f"metadata/additional.csv is missing column {column!r}")
    return frame


def additional_record(record_id: str) -> pd.Series:
    frame = load_additional()
    hit = frame[frame["id"] == record_id]
    if hit.empty:
        raise KeyError(f"metadata/additional.csv has no id {record_id!r}")
    return hit.iloc[0]
