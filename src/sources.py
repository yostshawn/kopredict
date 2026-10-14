"""Paths and loaders for metadata CSVs that list public inputs.

Which DepMap files to fetch is metadata/default_depmap_files_to_download_and_process.csv.
Publication workbooks are metadata/additional_datasets_to_download.csv.
Signed DepMap links live in metadata/20260920_DepMap_available_downloads.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

VERSION = "1.0"

DEPMAP_DOWNLOADS_CSV = "20260920_DepMap_available_downloads.csv"
DEPMAP_FILES_CSV = "default_depmap_files_to_download_and_process.csv"
ADDITIONAL_DATASETS_CSV = "additional_datasets_to_download.csv"
KNOWN_MD5_CSV = "known_md5_hashes.csv"

def metadata_dir() -> Path:
    """Directory of the DepMap catalog, the file list, and the extra datasets.

    A checkout keeps this next to ``src/``. An install keeps it under the
    environment prefix. The working directory is the last place checked.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "metadata",
        here.parent / "metadata",
        Path(sys.prefix) / "metadata",
        Path.cwd() / "metadata",
    ]
    for path in candidates:
        if (path / DEPMAP_DOWNLOADS_CSV).is_file() and (path / DEPMAP_FILES_CSV).is_file():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"metadata directory not found. Looked in: {searched}")


def source_root() -> Path:
    """Directory that contains ``metadata/`` for this checkout or install."""
    return metadata_dir().parent


def resolve_metadata_path(value: str) -> Path:
    """Resolve a cohort CSV path.

    An absolute path is used as given. A relative path such as
    ``metadata/oncotree_cohorts_solid.csv`` is resolved from the source root,
    then from the current directory.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [source_root() / path, metadata_dir() / path.name, Path.cwd() / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    looked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Cohort CSV not found: {value}. Looked in: {looked}")


def _read_metadata_csv(name: str) -> pd.DataFrame:
    path = metadata_dir() / name
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [str(column).strip() for column in df.columns]
    return df


def load_depmap_files() -> pd.DataFrame:
    """Rows of release and filename to pull from the DepMap download catalog."""
    df = _read_metadata_csv(DEPMAP_FILES_CSV)
    for column in ("release", "filename"):
        if column not in df.columns:
            raise ValueError(f"metadata/{DEPMAP_FILES_CSV} is missing column {column!r}")
    return df


def load_additional() -> pd.DataFrame:
    """Publication files that are not rows in the DepMap catalog."""
    df = _read_metadata_csv(ADDITIONAL_DATASETS_CSV)
    for column in ("id", "local_filename", "url", "citation", "download"):
        if column not in df.columns:
            raise ValueError(f"metadata/{ADDITIONAL_DATASETS_CSV} is missing column {column!r}")
    return df


def additional_record(record_id: str) -> pd.Series:
    df = load_additional()
    hit = df[df["id"] == record_id]
    if hit.empty:
        raise KeyError(f"metadata/{ADDITIONAL_DATASETS_CSV} has no id {record_id!r}")
    return hit.iloc[0]
