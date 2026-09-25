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
    CATALOG_URL,
    DEFAULT_DEPMAP_RELEASE,
    DEPMAP_RELEASE_TOKEN,
    NO_CAPTCHA_CATALOG_URL,
    load_additional,
    load_depmap_files,
    metadata_dir,
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


def depmap_release_label(token: str, catalog: pd.DataFrame) -> str:
    """Map 26Q1 or 'DepMap Public 26Q1' onto a release name present in the catalog."""
    releases = sorted(
        {
            str(value)
            for value in catalog.loc[
                catalog["release"].astype(str).str.startswith("DepMap Public"), "release"
            ].unique()
        }
    )
    text = token.strip()
    if text in releases:
        return text
    quarter = text.upper().replace("DEPMAP PUBLIC", "").replace(" ", "")
    matches = [name for name in releases if name.upper().replace(" ", "").endswith(quarter)]
    if len(matches) == 1:
        return matches[0]
    if not releases:
        raise SystemExit(f"Catalog has no DepMap Public releases, so {token!r} cannot be resolved.")
    known = "\n  ".join(releases)
    raise SystemExit(f"Unknown DepMap release {token!r}. Releases in this catalog:\n  {known}")


def files_for_release(depmap_release: str) -> tuple[tuple[str, str], ...]:
    """Expand metadata/depmap_files.csv, substituting the selected DepMap quarter."""
    rows = []
    for _, row in load_depmap_files().iterrows():
        release = str(row["release"]).strip()
        if release == DEPMAP_RELEASE_TOKEN:
            release = depmap_release
        filename = str(row["filename"]).strip()
        if release and filename:
            rows.append((release, filename))
    if not rows:
        raise SystemExit("metadata/depmap_files.csv has no files to download.")
    return tuple(rows)


def require_catalog_files(catalog: pd.DataFrame, depmap_release: str) -> tuple[tuple[str, str], ...]:
    if "url" not in catalog.columns:
        raise SystemExit(
            "Catalog has no url column. Replace metadata/downloads.csv with a fresh save of "
            f"{CATALOG_URL}."
        )
    wanted = files_for_release(depmap_release)
    missing = [
        f"{release} / {filename}"
        for release, filename in wanted
        if _lookup(catalog, release, filename) is None
        or not str(_lookup(catalog, release, filename).get("url", "")).strip()
    ]
    if missing:
        raise SystemExit(
            "These files are not in the selected DepMap catalog. "
            "Older quarters use different filenames.\n  " + "\n  ".join(missing)
        )
    return wanted


def download_pinned(catalog_path: Path, data_dir: Path, depmap_release: str) -> None:
    raw = _raw_dir(data_dir)
    public = None
    print(f"Fetching filename catalog {NO_CAPTCHA_CATALOG_URL}")
    catalog_check_path = raw / "depmap_no_captcha_catalog.csv"
    try:
        _fetch(NO_CAPTCHA_CATALOG_URL, catalog_check_path)
        public = _load_catalog(catalog_check_path)
    except Exception as exc:
        print(f"Could not refresh the open filename catalog ({exc}). Using checksums from {catalog_path}.")
    user = _load_catalog(catalog_path)
    release = depmap_release_label(depmap_release, user)
    pinned = require_catalog_files(user, release)
    print(f"DepMap release: {release}")

    for release, filename in pinned:
        row = _lookup(user, release, filename)
        if row is None or not str(row.get("url", "")).strip():
            raise SystemExit(
                f"No download URL for {release} / {filename} in {catalog_path}. "
                "Save a fresh catalog from https://depmap.org/portal/api/download/files."
            )
        dest = raw / filename
        expected = str(row.get("md5_hash", "")).strip().lower()
        if not expected and public is not None:
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


def download_additional(data_dir: Path) -> None:
    """Fetch publication files listed in metadata/additional.csv."""
    raw = _raw_dir(data_dir)
    for _, row in load_additional().iterrows():
        dest = raw / str(row["local_filename"]).strip()
        citation = str(row["citation"]).strip()
        url = str(row["url"]).strip()
        auto = str(row["download"]).strip().lower() in {"yes", "true", "1", "y"}
        if not auto:
            print(
                f"{row['id']}: save the workbook from {citation} to {dest}"
            )
            continue
        if dest.is_file() and dest.stat().st_size > 1000:
            print(f"OK {dest.name} already present")
            continue
        if not url:
            print(
                f"WARNING: {row['id']} is marked for download but has no url. Save it from {citation} to {dest}",
                file=sys.stderr,
            )
            continue
        print(f"Downloading {row['id']} ({citation})")
        try:
            _fetch(url, dest)
        except Exception as exc:
            print(
                f"WARNING: {row['id']} download failed ({exc}). Save it from {url} to {dest}",
                file=sys.stderr,
            )
            continue
        head = dest.read_bytes()[:200].lstrip().lower()
        if head.startswith(b"<") or b"html" in head[:80] or not head.startswith(b"pk"):
            dest.unlink(missing_ok=True)
            print(
                f"WARNING: {row['id']} URL did not return a workbook. Save it from {url} to {dest}",
                file=sys.stderr,
            )
            continue
        print(f"Saved {dest}")


def _catalog_has_urls(path: Path) -> bool:
    head = path.read_bytes()[:400].lstrip().lower()
    if head.startswith(b"<") or b"<html" in head:
        return False
    try:
        frame = pd.read_csv(path, dtype=str, nrows=5)
    except Exception:
        return False
    return "url" in frame.columns and frame["url"].fillna("").astype(str).str.strip().ne("").any()


def fetch_url_catalog(data_dir: Path) -> Path | None:
    """Save the DepMap catalog that includes signed URLs, when the portal allows it."""
    dest = _raw_dir(data_dir) / "depmap_download_catalog.csv"
    print(f"Fetching download-URL catalog {CATALOG_URL}")
    try:
        _fetch(CATALOG_URL, dest)
    except Exception as exc:
        print(f"Could not fetch the URL catalog ({exc}).", file=sys.stderr)
        dest.unlink(missing_ok=True)
        return None
    if _catalog_has_urls(dest):
        print(f"Saved URL catalog to {dest}")
        return dest
    dest.unlink(missing_ok=True)
    return None


def bundled_catalog(data_dir: Path) -> Path | None:
    del data_dir
    candidates = [metadata_dir() / "downloads.csv"]
    for path in candidates:
        if path.is_file() and _catalog_has_urls(path):
            return path
    return None


def _catalog_instructions() -> str:
    return (
        "No catalog with download URLs was found. "
        f"Open {CATALOG_URL} in a browser, complete the check, and replace metadata/downloads.csv "
        "with that file. Signed URLs expire, so replace it with a fresh download when fetches fail."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download pinned files for kopredict-prepare.")
    parser.add_argument(
        "--depmap-release",
        default=DEFAULT_DEPMAP_RELEASE,
        metavar="QUARTER",
        help=(
            "DepMap Public quarter to download, such as 26Q1 or 25Q3 "
            f"(default: {DEFAULT_DEPMAP_RELEASE}). The command exits if that release "
            "does not contain every required filename."
        ),
    )
    parser.add_argument(
        "--catalog",
        default=None,
        help=(
            "CSV from the DepMap download-URL catalog. Default: metadata/downloads.csv. "
            "Replace that file with a fresh browser save when the signed URLs expire."
        ),
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
    data_dir = get_data_dir()
    if args.catalog:
        catalog = Path(args.catalog).expanduser().resolve()
        if not catalog.is_file():
            raise SystemExit(f"Catalog not found: {catalog}")
    else:
        catalog = bundled_catalog(data_dir)
        if catalog is None:
            catalog = fetch_url_catalog(data_dir)
        if catalog is None:
            raise SystemExit(_catalog_instructions())
        print(f"Using catalog {catalog}")
    download_pinned(catalog, data_dir, args.depmap_release)
    download_additional(data_dir)


if __name__ == "__main__":
    main()
