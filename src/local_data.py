"""Load kopredict inputs from local parquet files under the project ``data/`` directory.

All tables are expected to use ``cellline`` as the model / cell line identifier (same
IDs across omics, models, dependency scores, and the features matrix columns).

Set ``KOPREDICT_DATA_DIR`` to override the data directory path.

Dependency parquets may be **long** (``cellline``, gene column, ``crispr`` or ``rnai``)
or **wide** (``cellline`` + one column per gene). Parquet reads are memoized with
small ``lru_cache`` pools so repeated gene lookups in one process don't re-read disk.
"""

from __future__ import annotations

import difflib
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Default filenames (must live under the data directory)
FILE_MODELS = "Model_and_OmicsSignatures.parquet.gz"
FILE_OMICS_FULL = "OmicsFull_processed.parquet.gz"
FILE_CRISPR_GENE_EFFECT = "CRISPRGeneEffect_processed.parquet.gz"
FILE_RNAI_DEP_SCORES = "D2_combined_gene_dep_scores_processed.parquet.gz"


def get_data_dir() -> Path:
    """Directory containing local parquet inputs."""
    override = os.environ.get("KOPREDICT_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    cwd_data = Path.cwd() / "data"
    if cwd_data.is_dir():
        return cwd_data.resolve()
    # Editable install: src/local_data.py -> repository root is parents[1]
    return Path(__file__).resolve().parents[1] / "data"


def _data_path(name: str) -> Path:
    p = get_data_dir() / name
    if not p.is_file():
        raise FileNotFoundError(
            f"Expected local data file at {p}. "
            f"Place {name} there or set KOPREDICT_DATA_DIR to the folder that contains it."
        )
    return p


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.rename(columns={c: c.lower() for c in out.columns})
    return out


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names for consistent lookups (public wrapper)."""
    return _normalize_columns(df)


def _ensure_cellline(df: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """Require a ``cellline`` column after normalizing header names."""
    df = _normalize_columns(df)
    if "cellline" not in df.columns:
        raise ValueError(
            f"{context}: parquet must include a 'cellline' column (headers are lowercased). "
            f"Columns: {list(df.columns)[:50]}"
        )
    return df


@lru_cache(maxsize=4)
def _read_models_cached(path: str) -> pd.DataFrame:
    return _ensure_cellline(pd.read_parquet(path), context="Model_and_OmicsSignatures")


@lru_cache(maxsize=4)
def _read_omics_cached(path: str) -> pd.DataFrame:
    return _ensure_cellline(pd.read_parquet(path), context="OmicsFull_processed")


@lru_cache(maxsize=4)
def _read_dependency_cached(path: str) -> pd.DataFrame:
    return _ensure_cellline(pd.read_parquet(path), context="Dependency scores file")


def load_cellline_models(
    celllines: Optional[Iterable[str]] = None,
    *,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load model / lineage metadata (``Model_and_OmicsSignatures.parquet.gz``).

    Parameters
    ----------
    celllines
        If given, restrict to these ``cellline`` values.
    columns
        If given, subset to these columns after normalization (lowercase names).
    """
    path = str(_data_path(FILE_MODELS))
    df = _read_models_cached(path)
    if celllines is not None:
        cl = set(str(x) for x in celllines)
        df = df[df["cellline"].astype(str).isin(cl)]
    if columns is not None:
        cols = ["cellline"] + [c.lower() for c in columns if c.lower() != "cellline"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    return df.copy()


def load_omics_full(
    genes: Optional[Iterable[str]] = None,
    celllines: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load long-format omics (``OmicsFull_processed.parquet.gz``).

    Rows are one per (cellline, gene). Filters apply to ``gene`` and ``cellline``.
    """
    path = str(_data_path(FILE_OMICS_FULL))
    df = _read_omics_cached(path)
    if "gene" not in df.columns:
        raise ValueError(
            "OmicsFull_processed parquet must include a 'gene' column for filtering."
        )
    if genes is not None:
        gset = set(str(x) for x in genes)
        df = df[df["gene"].astype(str).isin(gset)]
    if celllines is not None:
        cset = set(str(x) for x in celllines)
        df = df[df["cellline"].astype(str).isin(cset)]
    return df.copy()


def _dependency_gene_names(df: pd.DataFrame) -> list[str]:
    for column in ("gene", "genesymbol", "hugo_symbol", "symbol"):
        if column in df.columns:
            return df[column].dropna().astype(str).unique().tolist()
    return [str(column) for column in df.columns if str(column).lower() != "cellline"]


def _unknown_gene_error(gene: str, available: list[str], dataset: str) -> ValueError:
    by_lower = {name.lower(): name for name in available}
    close = [
        by_lower[key]
        for key in difflib.get_close_matches(str(gene).lower(), list(by_lower), n=5, cutoff=0.6)
    ]
    message = f"Gene {gene!r} is not in the {dataset} dependency matrix."
    if close:
        message += " Close matches: " + ", ".join(close) + "."
    return ValueError(message)


def _wide_dependency_to_long(
    dfn: pd.DataFrame,
    gene: str,
    value_col: str,
    celllines: Optional[Iterable[str]] = None,
    dataset: str = "dependency",
) -> pd.DataFrame:
    """Wide matrix: ``cellline`` column + one column per gene → ``cellline``, ``value_col``."""
    if gene not in dfn.columns:
        lower_to_orig = {str(c).lower(): c for c in dfn.columns if str(c).lower() != "cellline"}
        if str(gene).lower() in lower_to_orig:
            gene = lower_to_orig[str(gene).lower()]
        else:
            raise _unknown_gene_error(gene, list(lower_to_orig.values()), dataset)

    wide = dfn[["cellline", gene]].copy()
    wide = wide.rename(columns={gene: value_col})
    wide["cellline"] = wide["cellline"].astype(str)
    if celllines is not None:
        cset = set(str(x) for x in celllines)
        wide = wide[wide["cellline"].isin(cset)]
    return wide


def _long_dependency_table(
    df: pd.DataFrame,
    gene: str,
    value_col: str,
    celllines: Optional[Iterable[str]] = None,
    dataset: str = "dependency",
) -> pd.DataFrame:
    """Long-format: ``cellline`` + gene column + score column ``crispr`` or ``rnai`` only."""
    df = _ensure_cellline(df, context="Dependency scores (long format)")
    if value_col not in df.columns:
        raise ValueError(
            f"Long-format dependency table must include a '{value_col}' column "
            f"(from the selected dataset crispr or rnai). Columns: {list(df.columns)}"
        )
    gcol = None
    for cand in ("gene", "genesymbol", "hugo_symbol", "symbol"):
        if cand in df.columns:
            gcol = cand
            break
    if gcol is None:
        raise ValueError(
            "Long-format dependency file needs a gene column "
            "(gene, genesymbol, hugo_symbol, or symbol)."
        )
    sub = df[df[gcol].astype(str).str.lower() == str(gene).lower()].copy()
    if sub.empty:
        raise _unknown_gene_error(gene, df[gcol].dropna().astype(str).unique().tolist(), dataset)
    sub = sub[["cellline", value_col]].copy()
    if celllines is not None:
        cset = set(str(x) for x in celllines)
        sub = sub[sub["cellline"].astype(str).isin(cset)]
    return sub


def load_gene_dependency_scores(
    dataset: str,
    gene: str,
    celllines: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load CRISPR or RNAi dependency for one gene as ``cellline``, ``crispr`` or ``rnai``."""
    dataset = dataset.lower()
    if dataset == "crispr":
        path = _data_path(FILE_CRISPR_GENE_EFFECT)
        value_col = "crispr"
    elif dataset == "rnai":
        path = _data_path(FILE_RNAI_DEP_SCORES)
        value_col = "rnai"
    else:
        raise ValueError(f"dataset must be 'crispr' or 'rnai', got {dataset!r}")

    df = _read_dependency_cached(str(path))
    gene_cols = ("gene", "genesymbol", "hugo_symbol", "symbol")
    if any(c in df.columns for c in gene_cols):
        out = _long_dependency_table(df, gene, value_col, celllines=celllines, dataset=dataset)
    else:
        out = _wide_dependency_to_long(df, gene, value_col, celllines=celllines, dataset=dataset)

    out = out.dropna(subset=[value_col])
    return out


def clear_load_caches() -> None:
    """Drop LRU caches (e.g. after replacing files on disk in the same process)."""
    _read_models_cached.cache_clear()
    _read_omics_cached.cache_clear()
    _read_dependency_cached.cache_clear()
