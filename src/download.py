"""Download pinned DepMap catalog files and the HRD / signature workbooks."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from local_data import get_data_dir
from sources import (
    ADDITIONAL_DATASETS_CSV,
    DEPMAP_DOWNLOADS_CSV,
    DEPMAP_FILES_CSV,
    VERSION,
    load_additional,
    load_depmap_files,
    metadata_dir,
)

# Portal catalog with signed URLs. A browser save replaces an expired catalog.
CATALOG_URL = "https://depmap.org/portal/api/download/files"
DEFAULT_DEPMAP_RELEASE = "26Q1"
# In the default DepMap file list this release token follows --depmap-release.
DEPMAP_RELEASE_TOKEN = "{depmap}"
KNOWN_MD5_CSV = "known_md5_hashes.csv"
BUNDLED_INPUTS_DIR = "bundled_inputs"


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


def _already_valid(dest: Path, expected_md5: str) -> bool:
    """True when dest exists and its md5 matches. Mismatches are downloaded again."""
    if not dest.is_file():
        return False
    if not expected_md5:
        return False
    actual = _md5(dest)
    if actual == expected_md5:
        print(f"OK {dest.name} (exists, md5 matches)")
        return True
    print(f"{dest.name} exists but md5 does not match; downloading again.")
    return False


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "kopredict/1.0 (scientific data download)"},
    )
    if shutil.which("wget"):
        subprocess.run(
            [
                "wget",
                "-q",
                "--user-agent=kopredict/1.0 (scientific data download)",
                "-O",
                str(tmp),
                url,
            ],
            check=True,
        )
    else:
        with urllib.request.urlopen(request) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    tmp.replace(dest)


def _load_catalog(path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"release", "filename"}
    missing = required - set(catalog.columns)
    if missing:
        raise SystemExit(f"Catalog {path} is missing columns: {sorted(missing)}")
    return catalog


def _known_md5_table() -> pd.DataFrame:
    path = metadata_dir() / KNOWN_MD5_CSV
    if not path.is_file():
        return pd.DataFrame(columns=["release", "filename", "md5_hash"])
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in ("release", "filename", "md5_hash"):
        if column not in frame.columns:
            raise SystemExit(f"metadata/{KNOWN_MD5_CSV} is missing column {column!r}")
    return frame


def fill_blank_md5(catalog: pd.DataFrame) -> pd.DataFrame:
    """Fill blank catalog md5 values from metadata/known_md5_hashes.csv.

    DepMap often leaves checksums blank for older releases such as DEMETER2.
    """
    out = catalog.copy()
    if "md5_hash" not in out.columns:
        out["md5_hash"] = ""
    known = _known_md5_table()
    if known.empty:
        return out
    lookup = {
        (str(row["release"]).strip(), str(row["filename"]).strip()): str(row["md5_hash"]).strip().lower()
        for _, row in known.iterrows()
        if str(row["md5_hash"]).strip()
    }
    by_name = {
        str(row["filename"]).strip(): str(row["md5_hash"]).strip().lower()
        for _, row in known.iterrows()
        if str(row["md5_hash"]).strip()
    }
    filled = 0
    for idx, row in out.iterrows():
        if str(row.get("md5_hash", "")).strip():
            continue
        release = str(row["release"]).strip()
        filename = str(row["filename"]).strip()
        value = lookup.get((release, filename)) or by_name.get(filename)
        if value:
            out.at[idx, "md5_hash"] = value
            filled += 1
    if filled:
        print(f"Filled {filled} blank md5_hash value(s) from metadata/{KNOWN_MD5_CSV}.")
    return out


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
    """Expand the default DepMap file list, substituting the selected DepMap quarter."""
    rows = []
    for _, row in load_depmap_files().iterrows():
        release = str(row["release"]).strip()
        if release == DEPMAP_RELEASE_TOKEN:
            release = depmap_release
        filename = str(row["filename"]).strip()
        if release and filename:
            rows.append((release, filename))
    if not rows:
        raise SystemExit(f"metadata/{DEPMAP_FILES_CSV} has no files to download.")
    return tuple(rows)


def require_catalog_files(catalog: pd.DataFrame, depmap_release: str) -> tuple[tuple[str, str], ...]:
    if "url" not in catalog.columns:
        raise SystemExit(
            f"Catalog has no url column. Replace metadata/{DEPMAP_DOWNLOADS_CSV} with a fresh save of "
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
    user = fill_blank_md5(_load_catalog(catalog_path))
    release = depmap_release_label(depmap_release, user)
    pinned = require_catalog_files(user, release)
    print(f"DepMap release: {release}")

    for release, filename in pinned:
        row = _lookup(user, release, filename)
        if row is None or not str(row.get("url", "")).strip():
            raise SystemExit(
                f"No download URL for {release} / {filename} in {catalog_path}. "
                f"Save a fresh catalog from {CATALOG_URL} and pass it with --catalog."
            )
        dest = raw / filename
        expected = str(row.get("md5_hash", "")).strip().lower()
        if expected and _already_valid(dest, expected):
            continue
        if dest.is_file() and not expected:
            print(f"OK {dest.name} (exists; no md5_hash in catalog)")
            continue
        print(f"Downloading {release} / {filename}")
        _fetch(str(row["url"]).strip(), dest)
        actual = _md5(dest)
        if expected and actual != expected:
            dest.unlink(missing_ok=True)
            raise SystemExit(f"md5 mismatch for {filename}; deleted incomplete file.")
        if not expected:
            print(f"WARNING: no md5_hash for {filename}; saved without checksum check (md5={actual}).")
        print(f"Saved {dest}")


def verify_downloaded_files(data_dir: Path, depmap_release: str, catalog_path: Path | None = None) -> None:
    """Stop if any listed input is missing or its checksum does not match."""
    catalog_path = catalog_path or resolve_catalog_path(None, data_dir, fetch=False)
    if catalog_path is None or not catalog_path.is_file():
        raise SystemExit(f"Catalog not found. Looked for metadata/{DEPMAP_DOWNLOADS_CSV}.")
    catalog = fill_blank_md5(_load_catalog(catalog_path))
    release = depmap_release_label(depmap_release, catalog)
    raw = data_dir / "raw"
    problems: list[str] = []

    for file_release, filename in files_for_release(release):
        dest = raw / filename
        row = _lookup(catalog, file_release, filename)
        if row is None:
            problems.append(f"{filename}: not listed in {catalog_path.name} for {file_release}")
            continue
        expected = str(row.get("md5_hash", "")).strip().lower()
        if not dest.is_file():
            problems.append(f"{filename}: missing ({dest}). Run kopredict-download --depmap-release {depmap_release}.")
            continue
        if expected:
            actual = _md5(dest)
            if actual != expected:
                problems.append(f"{filename}: md5 mismatch (expected {expected}, found {actual})")

    for _, row in load_additional().iterrows():
        filename = str(row["local_filename"]).strip()
        dest = raw / filename
        label = str(row["id"]).strip()
        if not dest.is_file():
            problems.append(
                f"{filename}: missing ({dest}). Listed as {label} in metadata/{ADDITIONAL_DATASETS_CSV}."
            )
            continue
        expected = str(row["md5_hash"]).strip().lower() if "md5_hash" in row.index else ""
        if expected and _md5(dest) != expected:
            problems.append(f"{filename}: md5 mismatch (expected {expected}, found {_md5(dest)})")

    if problems:
        raise SystemExit(
            "Required input files are missing or do not match their checksums:\n  "
            + "\n  ".join(problems)
        )
    print(f"All input files are present and match md5_hash for {release}.")


def _bundled_input(filename: str) -> Path | None:
    path = metadata_dir() / BUNDLED_INPUTS_DIR / filename
    return path if path.is_file() else None


def _copy_bundled(filename: str, dest: Path) -> bool:
    source = _bundled_input(filename)
    if source is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    print(f"Copied bundled {filename} to {dest}")
    return True


def download_additional(data_dir: Path) -> None:
    """Fetch publication files listed in the additional-datasets CSV."""
    raw = _raw_dir(data_dir)
    for _, row in load_additional().iterrows():
        dest = raw / str(row["local_filename"]).strip()
        citation = str(row["citation"]).strip()
        url = str(row["url"]).strip()
        expected = str(row["md5_hash"]).strip().lower() if "md5_hash" in row.index else ""
        auto = str(row["download"]).strip().lower() in {"yes", "true", "1", "y"}
        label = str(row["id"]).strip()
        if not auto:
            if dest.is_file():
                print(f"OK {dest.name} (exists)")
                continue
            if _copy_bundled(dest.name, dest):
                continue
            print(
                f"WARNING: {label}: save the workbook from {citation} to {dest}",
                file=sys.stderr,
            )
            continue
        if expected and _already_valid(dest, expected):
            continue
        if dest.is_file() and not expected:
            head = dest.read_bytes()[:4]
            if head.startswith(b"PK"):
                print(f"OK {dest.name} (exists; metadata/{ADDITIONAL_DATASETS_CSV} has no md5_hash)")
                continue
            print(f"{dest.name} exists but is not a workbook; downloading again.")
        fetched = False
        if url:
            print(f"Downloading {label} ({citation})")
            try:
                _fetch(url, dest)
                head = dest.read_bytes()[:200].lstrip().lower()
                if head.startswith(b"<") or b"html" in head[:80] or not head.startswith(b"pk"):
                    dest.unlink(missing_ok=True)
                    print(
                        f"WARNING: {label} URL did not return a workbook.",
                        file=sys.stderr,
                    )
                else:
                    fetched = True
            except Exception as exc:
                print(f"WARNING: {label} download failed ({exc}).", file=sys.stderr)
                dest.unlink(missing_ok=True)
        if not fetched:
            if _copy_bundled(dest.name, dest):
                fetched = True
            else:
                print(
                    f"WARNING: {label} is missing. Save it from {url or citation} to {dest}",
                    file=sys.stderr,
                )
                continue
        if expected and _md5(dest) != expected:
            print(
                f"WARNING: md5 mismatch for {dest.name} after fetch/copy.",
                file=sys.stderr,
            )
            continue
        if fetched:
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
    dated = metadata_dir() / f"{date.today().strftime('%Y%m%d')}_DepMap_available_downloads.csv"
    dest = _raw_dir(data_dir) / "depmap_download_catalog.csv"
    print(f"Fetching download-URL catalog {CATALOG_URL}")
    try:
        _fetch(CATALOG_URL, dest)
    except Exception as exc:
        print(f"Could not fetch the URL catalog ({exc}).", file=sys.stderr)
        dest.unlink(missing_ok=True)
        return None
    if not _catalog_has_urls(dest):
        dest.unlink(missing_ok=True)
        print(
            "The DepMap portal did not return a CSV (often a browser check). "
            f"Open {CATALOG_URL}, save the file, and pass it with --catalog PATH.",
            file=sys.stderr,
        )
        return None
    try:
        shutil.copy2(dest, dated)
        print(f"Saved URL catalog to {dated}")
        return dated
    except Exception:
        print(f"Saved URL catalog to {dest}")
        return dest


def resolve_catalog_path(explicit: str | None, data_dir: Path | None = None, *, fetch: bool = False) -> Path | None:
    """Return the catalog used for DepMap download URLs.

    Default is ``metadata/20260920_DepMap_available_downloads.csv``. The catalog is
    not fetched from DepMap unless ``fetch`` is true (unused by the CLI).
    """
    del data_dir
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Catalog not found: {path}")
        if not _catalog_has_urls(path):
            raise SystemExit(f"Catalog {path} has no usable url column.")
        return path
    path = metadata_dir() / DEPMAP_DOWNLOADS_CSV
    if path.is_file() and _catalog_has_urls(path):
        return path
    if fetch:
        return fetch_url_catalog(get_data_dir())
    return None


def _catalog_instructions() -> str:
    return (
        f"Catalog not found or has no url column: metadata/{DEPMAP_DOWNLOADS_CSV}. "
        f"Open {CATALOG_URL} in a browser, save the CSV, and either replace that file "
        "or pass it with --catalog PATH. Signed URLs expire, so use a fresh save when fetches fail."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"Version {VERSION}\n\n"
            "Download the public files used by kopredict-prepare.\n\n"
            "  kopredict-download\n"
            "  kopredict-download --depmap-release 25Q3\n"
            "  kopredict-download --catalog /path/to/DepMap_available_downloads.csv\n"
            "  kopredict-download --help\n\n"
            f"By default the catalog is metadata/{DEPMAP_DOWNLOADS_CSV}. That file supplies\n"
            "the signed URLs used to download DepMap inputs. Pass --catalog only to override\n"
            "it (for example after saving a fresh browser copy when URLs expire). Blank\n"
            f"md5_hash values for older files are filled from metadata/{KNOWN_MD5_CSV}."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
            "CSV with DepMap download URLs. Default: "
            f"metadata/{DEPMAP_DOWNLOADS_CSV}."
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
        import local_data

        local_data.clear_load_caches()
    data_dir = get_data_dir()
    catalog = resolve_catalog_path(args.catalog, data_dir, fetch=False)
    if catalog is None:
        raise SystemExit(_catalog_instructions())
    print(f"Using catalog {catalog}")
    download_pinned(catalog, data_dir, args.depmap_release)
    download_additional(data_dir)
    print("Download finished.")


if __name__ == "__main__":
    main()
