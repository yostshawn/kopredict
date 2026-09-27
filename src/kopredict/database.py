"""Local parquet data access (replaces Redshift-backed queries).

Builds cohorts of ``cellline`` IDs from ``Model_and_OmicsSignatures`` for:
  * CLI ``-c`` / ``--cancer-types`` (match an OncoTree column to one or more labels)
  * ``--cohort-csv`` rows (``oncotree_column`` + ``name``), including special names
    ``PANCANCER`` and ``SOLID`` (empty ``oncotree_column`` allowed for those).

All resolution funnels through ``local_data.load_cellline_models`` and exact string
matching on model fields (no case folding).
"""

import sys
from pathlib import Path

import pandas as pd

from . import local_data

_SPECIAL_PANCANCER = "PANCANCER"
_SPECIAL_SOLID = "SOLID"
# When ``SOLID`` is requested, lines with these ``oncotreeprimarydisease`` labels are
# treated as non-solid (hematologic) and excluded from the cohort.
_HEMATOLOGIC_PRIMARY_DISEASES = {
    "Acute Lymphoblastic Leukemia",
    "Acute Myeloid Leukemia",
    "Acute Myeloid Leukemia transformed from MPN",
    "B Acute Lymphoblastic Leukemia",
    "Burkitt Lymphoma",
    "Chronic Lymphocytic Leukemia",
    "Chronic Myelogenous Leukemia",
    "Diffuse Large B-Cell Lymphoma",
    "Lymphoma",
    "Mixed Phenotype Acute Leukemia",
    "Plasma Cell Myeloma",
    "T Acute Lymphoblastic Leukemia",
}


def _special_cohort_celllines(df: pd.DataFrame, name: str) -> list[str] | None:
    """Return cell lines for special cohort keywords, else ``None``."""
    if "oncotreeprimarydisease" in df.columns:
        opd = df["oncotreeprimarydisease"].astype(str).str.upper()
        cancer_mask = ~opd.str.contains("NON-CANCER", regex=False)
    else:
        cancer_mask = pd.Series(True, index=df.index)

    n = str(name).strip().upper()
    if n == _SPECIAL_PANCANCER:
        return df.loc[cancer_mask, "cellline"].astype(str).tolist()
    if n == _SPECIAL_SOLID:
        if "oncotreeprimarydisease" not in df.columns:
            raise ValueError(
                "Cannot resolve SOLID: model table is missing 'oncotreeprimarydisease'."
            )
        mask = cancer_mask & ~df["oncotreeprimarydisease"].isin(_HEMATOLOGIC_PRIMARY_DISEASES)
        return df.loc[mask, "cellline"].astype(str).tolist()
    return None


def return_celllines_matching_oncotree_filter(oncotree_column: str, values: list) -> list[str]:
    """Cell lines where ``oncotree_column`` equals one of ``values`` (exact string match; OR across values)."""
    if not values:
        sys.exit("ERROR: Provide at least one value with -c / --cancer-types (comma-separated).")
    df = local_data.load_cellline_models()
    if oncotree_column not in df.columns:
        sys.exit(
            f"ERROR: Column {oncotree_column!r} not found in model table. "
            f"Example columns: {sorted(df.columns)[:40]} ..."
        )
    out_set: set[str] = set()
    normal_values: list[str] = []
    for val in values:
        special = _special_cohort_celllines(df, str(val))
        if special is None:
            normal_values.append(val)
        else:
            out_set.update(str(x) for x in special)
    if normal_values:
        mask = df[oncotree_column].isin(normal_values)
        out_set.update(df.loc[mask, "cellline"].astype(str).tolist())
    out = list(out_set)
    if not out:
        sys.exit(
            f"ERROR: No cell lines matched filter: column={oncotree_column!r}, values={values!r}."
        )
    return out


def celllines_for_single_cohort_row(oncotree_column: str, name: str) -> list[str]:
    """Cell lines where ``oncotree_column`` equals ``name`` (exact string match)."""
    if not name:
        raise ValueError("name must be non-empty.")
    df = local_data.load_cellline_models()
    special = _special_cohort_celllines(df, name)
    if special is not None:
        out = list(set(str(x) for x in special))
        if not out:
            raise ValueError(f"No cell lines resolved for special cohort name {name!r}.")
        return out
    if not oncotree_column:
        raise ValueError(
            "oncotree_column must be non-empty unless name is one of: PANCANCER, SOLID."
        )
    if oncotree_column not in df.columns:
        raise ValueError(
            f"oncotree_column {oncotree_column!r} not in model table. "
            f"Columns include: {sorted(df.columns)[:40]} ..."
        )
    mask = df[oncotree_column] == name
    out = df.loc[mask, "cellline"].astype(str).tolist()
    if not out:
        raise ValueError(f"No cell lines for filter {oncotree_column!r} == {name!r}.")
    return out


def parse_cohort_name_list(name: str) -> list[str]:
    """One cohort name, or several names separated by semicolons.

    A semicolon-separated list is combined the same way as ``cancer_types``:
    a cell line is kept when it matches any name. Semicolons are used so a
    value that itself contains a comma, such as ``AML, NOS``, stays one name.
    """
    text = "" if name is None else str(name)
    if text.strip() == "":
        return []
    if ";" not in text:
        return [text]
    return [part.strip() for part in text.split(";") if part.strip() != ""]


def celllines_for_cohort_names(oncotree_column: str, names: list[str]) -> list[str]:
    """Cell lines matching any name in ``names`` for one OncoTree column."""
    if not names:
        raise ValueError("name must be non-empty.")
    found: set[str] = set()
    for name in names:
        found.update(celllines_for_single_cohort_row(oncotree_column, name))
    return list(found)


def validate_cohort_table(raw: pd.DataFrame) -> None:
    """Require every oncotree_column and name pair to match before any cohort runs.

    A semicolon-separated ``name`` is checked one value at a time. The process
    stops with every unknown pair listed when any value is missing.
    """
    problems: list[str] = []
    for idx, row in raw.iterrows():
        names = parse_cohort_name_list(row["name"])
        if not names:
            continue
        column = row["oncotree_column"]
        for name in names:
            try:
                celllines_for_single_cohort_row(column, name)
            except ValueError as exc:
                problems.append(f"Row {int(idx) + 2}: {exc}")
    if problems:
        raise SystemExit(
            "ERROR: Cohort CSV has oncotree_column + name combinations that do not exist. "
            "The search was not run.\n" + "\n".join(problems)
        )


def load_cohort_definition_from_csv(path: str | Path) -> tuple[list[str], list[str]]:
    """Read cohort CSV; return (cellline ids, unique ``name`` labels in first-seen order).

    CSV must include columns ``oncotree_column`` and ``name``. Any other columns are ignored.
    ``name`` may be one value or several values separated by semicolons. A list is
    combined the same way as ``cancer_types``: a cell line is kept when it matches any name.
    """
    path = Path(path)
    if not path.is_file():
        sys.exit(f"ERROR: Cohort CSV not found: {path}")
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    for req in ("oncotree_column", "name"):
        if req not in raw.columns:
            sys.exit(f"ERROR: Cohort CSV must contain column {req!r}. Found: {list(raw.columns)}")
    raw = raw[["oncotree_column", "name"]]
    validate_cohort_table(raw)
    celllines: set[str] = set()
    labels: list[str] = []
    for idx, row in raw.iterrows():
        col = row["oncotree_column"]
        name = row["name"]
        names = parse_cohort_name_list(name)
        if not names:
            continue
        try:
            matched = celllines_for_cohort_names(col, names)
        except ValueError as exc:
            sys.exit(f"ERROR: Row {idx}: {exc}")
        celllines.update(matched)
        label = ";".join(names)
        if label not in labels:
            labels.append(label)
    if not celllines:
        sys.exit("ERROR: No cell lines matched any row in the cohort CSV.")
    return list(celllines), labels


def pull_target_gene_dependency_scores(gene: str, dataset: str) -> pd.DataFrame:
    """Return ``cellline`` + ``crispr`` or ``rnai`` scores for one gene from local dependency matrices."""
    return local_data.load_gene_dependency_scores(dataset, gene)


def pull_cell_line_metadata(cl=None, full=False):
    """Load cell line / model metadata from ``Model_and_OmicsSignatures.parquet.gz``."""
    cl = cl or []
    columns = [
        "cellline",
        "oncotreelineage",
        "oncotreeprimarydisease",
        "oncotreesubtype",
        "oncotreecode",
        "rrid",
        "primaryormetastasis",
        "samplecollectionsite",
        "sourcetype",
        "growthpattern",
        "formulationid",
        "msiscore",
        "ploidy",
        "wgd",
        "lohfraction",
        "cin",
        "aneuploidy",
        "msihigh",
    ]
    df = local_data.load_cellline_models(celllines=cl if cl else None)
    if full:
        return df
    avail = [c for c in columns if c in df.columns]
    if not avail:
        return df
    return df[avail].copy()


def return_lineages_list():
    """Return distinct oncotreelineage values from the local models file."""
    cur_df = local_data.load_cellline_models()
    if "oncotreelineage" not in cur_df.columns:
        sys.exit("ERROR: Model file must contain oncotreelineage column.")
    return cur_df["oncotreelineage"].dropna().unique().tolist()


def pull_cell_line_full_genomics(genes=None, celllines=None, full=False, where_clause=""):
    """Load long-format omics from ``OmicsFull_processed.parquet.gz`` (replaces Redshift omics table)."""
    genes = genes or []
    celllines = celllines or []
    _ = where_clause  # unused; kept for API compatibility
    df = local_data.load_omics_full(
        genes=genes if genes else None,
        celllines=celllines if celllines else None,
    )
    if full:
        return df
    columns = [
        "cellline",
        "gene",
        "cnv",
        "cnv_call",
        "ko_alleles",
        "gts",
        "variantinfos",
        "proteinchanges",
        "vepimpacts",
        "mutdamaging",
        "muthotspot",
        "has_hessdriver",
        "tpmlogp1",
        "tpmlogp1_zscore",
        "altered",
        "ko",
        "genomic_status",
        "crispr",
        "crispr_zscore",
        "rnai",
        "rnai_zscore",
    ]
    avail = [c for c in columns if c in df.columns]
    return df[avail].copy() if avail else df


# --- Drug / PGX (Redshift: s3_pgx_drugs_20240809, s3_pgx_drug_sensitivities_20240809) ---
# Disabled for local runs; re-enable when PGX parquet sources are wired in.
#
# def drugs_given_target(target):
#     query = "SELECT * FROM s3_pgx_drugs_20240809 WHERE target LIKE '%" + target + "%';"
#     return run_redshift_query_return_pandas(query)
#
# def return_drug_data_given_dataset_and_drug(df):
#     ...


def drugs_given_target(target):
    raise NotImplementedError(
        "Drug-target mode is disabled in the local-only build. "
        "Use dataset 'crispr' or 'rnai'."
    )


def return_drug_data_given_dataset_and_drug(df):
    raise NotImplementedError(
        "Drug-target mode is disabled in the local-only build. "
        "Use dataset 'crispr' or 'rnai'."
    )


def return_unique_values(database="", return_column="", not_null_column=""):
    """Return distinct values from a column; ``database`` may be a parquet path or omics/models alias."""
    if not database:
        return "ERROR: Provide database"
    if not return_column:
        return "ERROR: Provide return_column"
    rc = return_column.lower()
    db_lower = str(database).lower()
    path = Path(database)
    if path.is_file():
        df = pd.read_parquet(path)
    elif db_lower in ("omics", "omicsfull", "s3_depmap_all_omics_25q2"):
        df = local_data.load_omics_full()
    elif db_lower in ("models", "model", "s3_depmap_cellline_models_25q2", "s3_depmap_cellline_models_24q2"):
        df = local_data.load_cellline_models()
    else:
        return f"ERROR: Unknown database {database!r}. Use a parquet path, 'omics', or 'models'."

    df = local_data.normalize_dataframe_columns(df)
    if rc not in df.columns:
        return f"ERROR: Column {return_column!r} not in table (have: {list(df.columns)[:40]} ...)"
    series = df[rc]
    if not_null_column:
        nn = not_null_column.lower()
        if nn not in df.columns:
            return f"ERROR: not_null_column {not_null_column!r} not found."
        series = series[df[nn].notna()]
    return pd.DataFrame({return_column: series.dropna().unique()})


def create_connection():
    raise RuntimeError("Redshift is disabled; kopredict uses local parquet files under data/.")


def run_redshift_query_return_pandas(query):
    raise RuntimeError("Redshift is disabled; kopredict uses local parquet files under data/.")
