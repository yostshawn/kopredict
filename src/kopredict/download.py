"""Download pinned DepMap catalog files and the HRD supplementary workbook."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from kopredict.local_data import get_data_dir
from kopredict.sources import (
    HRD_DOI,
    HRD_RAW_NAME,
    HRD_XLSX_URL,
    NO_CAPTCHA_CATALOG_URL,
    PETLJAK_RAW_NAME,
    PETLJAK_URL,
    PINNED_FILES,
)


def _raw_dir(data_dir: Path) -> Path:
    path = data_dir / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    if shutil.which("wget"):
        subprocess.run(["wget", "-q", "-O", str(tmp), url], check=True)
    else:
        urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def _load_catalog(path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"release", "filename"}
    missing = required - set(catalog.columns)
    if missing:
        raise SystemExit(f"Catalog {path} is missing columns: {sorted(missing)}")
    return catalog


def _lookup(catalog: pd.DataFrame, release: str, filename: str) -> pd.Series | None:
    hit = catalog[(catalog["release"] == release) & (catalog["filename"] == filename)]
    if hit.empty:
        return None
    return hit.iloc[0]


def download_pinned(catalog_path: Path, data_dir: Path) -> None:
    raw = _raw_dir(data_dir)
    print(f"Fetching filename catalog {NO_CAPTCHA_CATALOG_URL}")
    catalog_check_path = raw / "depmap_no_captcha_catalog.csv"
    _fetch(NO_CAPTCHA_CATALOG_URL, catalog_check_path)
    public = _load_catalog(catalog_check_path)
    user = _load_catalog(catalog_path)

    missing_from_public = [
        f"{release}/{filename}"
        for release, filename in PINNED_FILES
        if _lookup(public, release, filename) is None
    ]
    if missing_from_public:
        raise SystemExit(
            "Pinned files are not in the DepMap catalog:\n  " + "\n  ".join(missing_from_public)
        )

    if "url" not in user.columns:
        raise SystemExit(
            f"{catalog_path} has no url column. Save "
            "https://depmap.org/portal/api/download/files from a browser after the captcha check."
        )

    for release, filename in PINNED_FILES:
        row = _lookup(user, release, filename)
        if row is None or not str(row.get("url", "")).strip():
            raise SystemExit(
                f"No download URL for {release} / {filename} in {catalog_path}. "
                "Save a fresh catalog from https://depmap.org/portal/api/download/files."
            )
        dest = raw / filename
        expected = str(row.get("md5_hash", "")).strip().lower()
        if not expected:
            public_row = _lookup(public, release, filename)
            if public_row is not None:
                expected = str(public_row.get("md5_hash", "")).strip().lower()
        if dest.is_file() and expected and _md5(dest) == expected:
            print(f"OK {filename} (md5 matches)")
            continue
        print(f"Downloading {release} / {filename}")
        _fetch(str(row["url"]).strip(), dest)
        if expected and _md5(dest) != expected:
            dest.unlink(missing_ok=True)
            raise SystemExit(f"md5 mismatch for {filename}; deleted incomplete file.")
        print(f"Saved {dest}")


def download_hrd(data_dir: Path) -> None:
    dest = _raw_dir(data_dir) / HRD_RAW_NAME
    if dest.is_file() and dest.stat().st_size > 1000:
        print(f"OK {dest.name} already present")
        return
    print(f"Downloading HRD Table S5 ({HRD_DOI})")
    try:
        _fetch(HRD_XLSX_URL, dest)
    except Exception as exc:
        print(
            f"WARNING: HRD download failed ({exc}). Save Supplementary Table S5 from {HRD_DOI} "
            f"to {dest}",
            file=sys.stderr,
        )
        return
    print(f"Saved {dest}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download pinned files for kopredict-prepare.")
    parser.add_argument(
        "--catalog",
        required=True,
        help="CSV from https://depmap.org/portal/api/download/files with a filled url column.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory for raw/ and processed files (default: KOPREDICT_DATA_DIR or ./data).",
    )
    args = parser.parse_args(argv)
    if args.data_dir:
        import os

        os.environ["KOPREDICT_DATA_DIR"] = str(Path(args.data_dir).expanduser().resolve())
        from kopredict import local_data

        local_data.clear_load_caches()
    catalog = Path(args.catalog).expanduser().resolve()
    if not catalog.is_file():
        raise SystemExit(f"Catalog not found: {catalog}")
    data_dir = get_data_dir()
    download_pinned(catalog, data_dir)
    download_hrd(data_dir)
    petljak = data_dir / "raw" / PETLJAK_RAW_NAME
    print(
        f"Petljak signature workbook is not downloaded automatically ({PETLJAK_URL}). "
        f"Save mmc1.xlsx to {petljak}"
    )


if __name__ == "__main__":
    main()
