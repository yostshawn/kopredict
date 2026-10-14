"""Build local parquet inputs and the binary feature matrix from public downloads."""

from __future__ import annotations

import argparse
import gc
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from download import DEFAULT_DEPMAP_RELEASE, verify_downloaded_files
from local_data import clear_load_caches, get_data_dir
from log import log_to_file, shape
from sources import VERSION, additional_record
from utils import hash_feature_rows

FEATURES_FILENAME = "final_depmap_features_26Q1.parquet.gz"
HRD_SCORE_CUTOFF = 42
CN_AMP_CUTOFF = 2
CN_HOMDEL_CUTOFF = 0.1
ZSCORE_CUTOFF = 1.64
LOF_CLASSES = (
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "Nonstop_Mutation",
)
PROTEIN_CODING_LOCUS = "protein-coding gene"

_FILE_MODELS = "Model_and_OmicsSignatures.parquet.gz"
_FILE_OMICS = "OmicsFull_processed.parquet.gz"
_FILE_PROTEOMICS = "Proteomics_processed.parquet.gz"
_FILE_CRISPR = "CRISPRGeneEffect_processed.parquet.gz"
_FILE_RNAI = "D2_combined_gene_dep_scores_processed.parquet.gz"

_METADATA_COLUMNS = (
    "oncotreelineage",
    "oncotreeprimarydisease",
    "oncotreesubtype",
    "oncotreecode",
    "agecategory",
    "patientrace",
    "race",
    "primaryormetastasis",
    "samplecollectionsite",
    "growthpattern",
    "wgd",
    "msihigh",
    "ploidyround",
    "aneuploidyby5bin",
    "cinquantilebins",
    "lohfractionbyp1bin",
    "hrdscorehigh",
)

_SBS_RE = re.compile(r"^sbs[\da-z]+$", re.IGNORECASE)


def _raw(data_dir: Path, name: str) -> Path:
    path = data_dir / "raw" / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run kopredict-download and place extra workbooks in raw/.")
    return path


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _symbol(value) -> str:
    text = str(value).strip()
    if text.endswith(")") and " (" in text:
        text = text.rsplit(" (", 1)[0].strip()
    return text.upper()


def protein_coding_genes(gene_table: pd.DataFrame) -> pd.DataFrame:
    """Keep Gene.csv rows whose locus_group is a protein-coding gene."""
    columns = {str(c).lower(): c for c in gene_table.columns}
    if "locus_group" not in columns:
        raise ValueError("Gene.csv must contain a locus_group column.")
    group = gene_table[columns["locus_group"]].astype(str).str.strip().str.lower()
    kept = gene_table.loc[group == PROTEIN_CODING_LOCUS].copy()
    if kept.empty:
        raise ValueError(f"Gene.csv has no rows with locus_group {PROTEIN_CODING_LOCUS!r}.")
    return kept


def build_gene_lookup(gene_table: pd.DataFrame) -> dict[str, str]:
    """Map HGNC, Entrez, Ensembl, and UniProt ids to the Gene.csv symbol."""
    columns = {str(c).lower(): c for c in gene_table.columns}
    if "symbol" not in columns:
        raise ValueError("Gene.csv must contain a symbol column.")
    symbols = gene_table[columns["symbol"]].astype(str).str.strip()
    lookup: dict[str, str] = {}

    def add(keys: pd.Series, symbol_values: pd.Series) -> None:
        for key, symbol in zip(keys, symbol_values):
            text = str(key).strip()
            if not text or text.lower() in {"nan", "none", "<na>"}:
                continue
            if not symbol or str(symbol).lower() in {"nan", "none"}:
                continue
            lookup.setdefault(text.upper(), str(symbol))

    add(symbols, symbols)
    if "entrez_id" in columns:
        entrez = pd.to_numeric(gene_table[columns["entrez_id"]], errors="coerce")
        add(entrez.dropna().astype("int64").astype(str), symbols.loc[entrez.dropna().index])
    if "ensembl_gene_id" in columns:
        ensembl = gene_table[columns["ensembl_gene_id"]].astype(str).str.split(".").str[0]
        add(ensembl, symbols)
    if "hgnc_id" in columns:
        add(gene_table[columns["hgnc_id"]].astype(str), symbols)
    if "uniprot_ids" in columns:
        parts = (
            gene_table[columns["uniprot_ids"]]
            .astype(str)
            .str.split("|")
            .explode()
            .str.strip()
        )
        add(parts, symbols.loc[parts.index])
    return lookup


def resolve_gene_label(value, lookup: dict[str, str]) -> str | None:
    """Return the Gene.csv symbol for a DepMap gene label, or None if it is not a gene."""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "unnamed: 0"}:
        return None
    left, inside = text, ""
    if text.endswith(")") and " (" in text:
        left, inside = text[:-1].rsplit(" (", 1)
        inside = inside.split(".", 1)[0].strip()
    candidates = [left, inside, text.split(".", 1)[0], text]
    for candidate in candidates:
        token = candidate.strip()
        if not token:
            continue
        hit = lookup.get(token.upper())
        if hit:
            return hit
    return None


def _apply_gene_symbols(frame: pd.DataFrame, lookup: dict[str, str] | None, label: str) -> pd.DataFrame:
    if not lookup or frame.empty:
        return frame
    mapped = [resolve_gene_label(name, lookup) for name in frame.index.astype(str)]
    kept = frame.copy()
    kept.index = mapped
    recognized = kept.index.notna()
    dropped = int((~recognized).sum())
    kept = kept.loc[recognized]
    print(
        f"    {label}: kept {len(kept)} gene symbols; "
        f"removed {dropped} genes that are not protein-coding genes in Gene.csv"
    )
    kept.attrs["genes_removed"] = dropped
    return kept


def _name_key(value) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def _collapse_duplicate_index(df: pd.DataFrame, how: str) -> pd.DataFrame:
    if not df.index.has_duplicates:
        return df
    grouped = df.groupby(level=0)
    if how == "max":
        return grouped.max()
    return grouped.mean()


def build_name_to_model(models: pd.DataFrame) -> dict[str, str]:
    """Map stripped and CCLE-style names onto DepMap ModelID."""
    mapping: dict[str, str] = {}
    id_col = "cellline" if "cellline" in models.columns else "modelid"
    for col in ("strippedcelllinename", "celllinename", "cclename", "rrid"):
        if col not in models.columns:
            continue
        for model_id, name in zip(models[id_col], models[col]):
            key = _name_key(name)
            if key and key not in mapping and pd.notna(model_id):
                mapping[key] = str(model_id)
    return mapping


def match_label_to_model(label, model_ids: set[str], name_to_model: dict[str, str]) -> str | None:
    text = str(label).strip()
    if text in model_ids:
        return text
    key = _name_key(text)
    if key in name_to_model:
        return name_to_model[key]
    if "_" in text:
        head = _name_key(text.split("_", 1)[0])
        if head in name_to_model:
            return name_to_model[head]
    return None


def _profile_frame(profiles: pd.DataFrame) -> pd.DataFrame:
    frame = profiles.copy()
    frame.columns = [str(c).lower().replace(" ", "").replace("_", "") for c in frame.columns]
    return frame


def _map_to_model(profiles: pd.DataFrame, source: str) -> dict[str, str]:
    frame = _profile_frame(profiles)
    if source not in frame.columns or "modelid" not in frame.columns:
        return {}
    mapping = {}
    for src, model in zip(frame[source], frame["modelid"]):
        if pd.notna(src) and pd.notna(model):
            mapping[str(src)] = str(model)
    return mapping


def profile_to_model_map(profiles: pd.DataFrame) -> dict[str, str]:
    return _map_to_model(profiles, "profileid")


def condition_to_model_map(profiles: pd.DataFrame) -> dict[str, str]:
    """Map OmicsProfiles ModelConditionID values onto ModelID."""
    return _map_to_model(profiles, "modelconditionid")


def orient_gene_by_model(
    df: pd.DataFrame,
    model_ids: set[str],
    name_to_model: dict[str, str],
    profile_map: dict[str, str],
    *,
    condition_map: dict[str, str] | None = None,
    binary: bool = False,
    gene_lookup: dict[str, str] | None = None,
    label: str = "matrix",
) -> pd.DataFrame:
    """Return a gene-by-ModelID numeric matrix."""
    frame = df.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    compact = {c.lower().replace(" ", "").replace("_", ""): c for c in frame.columns}
    id_name = None
    for key in ("modelid", "depmapid", "profileid", "modelconditionid", "sequencingid", "cellline"):
        if key in compact:
            id_name = compact[key]
            break
    if id_name is None:
        id_name = frame.columns[0]
    labels = frame[id_name].astype(str)
    values = frame.drop(columns=[id_name])
    values = _numeric_frame(values)
    values.index = labels

    def _to_model(text: str) -> str | None:
        if text in profile_map and profile_map[text] in model_ids:
            return profile_map[text]
        if condition_map and text in condition_map and condition_map[text] in model_ids:
            return condition_map[text]
        return match_label_to_model(text, model_ids, name_to_model)

    def _axis_models(axis_labels) -> dict:
        found = {}
        for lab in axis_labels:
            text = str(lab)
            # "SYMBOL (7157)" headers are genes, not cell lines.
            if text.endswith(")") and " (" in text:
                continue
            matched = _to_model(text)
            if matched:
                found[lab] = matched
        return found

    row_map = _axis_models(values.index)
    col_map = _axis_models(values.columns)
    sample_ids_were_columns = False
    if len(col_map) > len(row_map):
        values = values.T
        row_map = col_map
        sample_ids_were_columns = True
    raw_ids = pd.Index(values.index.astype(str)).unique()
    keep = [lab for lab in values.index if lab in row_map]
    matched_ids = {str(lab) for lab in keep}
    id_source = "column headers (cell-line name)" if sample_ids_were_columns else id_name
    print(
        f"    {label}: sample id is {id_source}; {len(raw_ids)} ids in the file; "
        f"{len(raw_ids) - len(matched_ids)} not in Model.csv; {len({row_map[lab] for lab in keep})} ModelIDs kept"
    )
    values = values.loc[keep]
    values.index = [row_map[lab] for lab in values.index]
    values = values.T
    if values.columns.duplicated().any():
        values = values.T.groupby(level=0).mean().T if not binary else values.T.groupby(level=0).max().T
    values = _collapse_duplicate_index(values, "max" if binary else "mean")
    values = _apply_gene_symbols(values, gene_lookup, label)
    genes_removed = int(values.attrs.get("genes_removed", 0))
    values = _collapse_duplicate_index(values, "max" if binary else "mean")
    values = values.dropna(axis=0, how="all").dropna(axis=1, how="all")
    values.attrs["genes_removed"] = genes_removed
    return values


def choose_hrd_score(hrd: pd.DataFrame) -> pd.Series:
    """hrdsum_summary when present, otherwise hrdsum_broad, indexed by DepMap_ID."""
    frame = hrd.copy()
    frame.columns = [str(c).strip().lower().replace(" ", "") for c in frame.columns]
    if "depmap_id" not in frame.columns:
        raise ValueError(f"HRD table has no DepMap_ID column ({additional_record('hrd')['citation']}).")
    summary = pd.to_numeric(frame["hrdsum_summary"], errors="coerce") if "hrdsum_summary" in frame.columns else None
    broad = pd.to_numeric(frame["hrdsum_broad"], errors="coerce") if "hrdsum_broad" in frame.columns else None
    if summary is None and broad is None:
        raise ValueError("HRD table has neither hrdsum_summary nor hrdsum_broad.")
    score = summary if summary is not None else broad
    if summary is not None and broad is not None:
        score = summary.where(summary.notna(), broad)
    score.index = frame["depmap_id"].astype(str)
    return score.groupby(level=0).mean()


def read_hrd_table(path: Path) -> pd.DataFrame:
    for header in (0, 1, 2):
        frame = pd.read_excel(path, sheet_name="table_s4", header=header)
        columns = {str(c).strip().lower() for c in frame.columns}
        if "depmap_id" in columns and ("hrdsum_summary" in columns or "hrdsum_broad" in columns):
            frame.columns = [str(c).strip().lower() for c in frame.columns]
            return frame
    raise ValueError(
        f"Could not find HRD columns in {path} ({additional_record('hrd')['citation']})."
    )


_SBS_SHEET = "COSMIC CellLines Signatures"


def load_sbs_exposures(path: Path) -> pd.DataFrame:
    """Cell-line SBS activities from the Petljak workbook sheet ``COSMIC CellLines Signatures``.

    The sheet has a title row, then ``Sample Name`` plus SBS columns.
    """
    book = pd.ExcelFile(path)
    sheet = next((name for name in book.sheet_names if name.strip() == _SBS_SHEET), None)
    if sheet is None:
        raise ValueError(
            f"Sheet {_SBS_SHEET!r} not found in {path}. "
            f"Sheets: {book.sheet_names}."
        )
    preview = pd.read_excel(book, sheet_name=sheet, header=None, nrows=6)
    header_row = None
    for i, row in preview.iterrows():
        values = [str(v).strip().lower() for v in row.tolist()]
        if "sample name" in values:
            header_row = int(i)
            break
    if header_row is None:
        raise ValueError(f"Sheet {_SBS_SHEET!r} in {path} has no 'Sample Name' header.")
    full = pd.read_excel(book, sheet_name=sheet, header=header_row)
    return _exposures_from_sheet(full)


def _sbs_columns(frame: pd.DataFrame) -> list:
    found = []
    for col in frame.columns:
        text = str(col).strip().lower().replace(" ", "")
        if _SBS_RE.match(text) or text.startswith("sbs"):
            found.append(col)
    return found


def _exposures_from_sheet(frame: pd.DataFrame) -> pd.DataFrame:
    sbs_cols = _sbs_columns(frame)
    if len(sbs_cols) < 3:
        raise ValueError("COSMIC CellLines Signatures sheet does not have SBS columns.")
    id_col = next((c for c in frame.columns if str(c).strip().lower() == "sample name"), frame.columns[0])
    exposures = frame.set_index(id_col)[sbs_cols]
    # Sample names such as 697 are stored as numbers in the workbook.
    exposures.index = exposures.index.map(lambda value: str(value).strip())
    exposures = exposures[~exposures.index.str.lower().isin(["nan", "sample name", ""])]
    exposures.columns = [str(c).strip().lower().replace(" ", "") for c in exposures.columns]
    return _numeric_frame(exposures)


def _combine_copy_number(wgs: pd.DataFrame, wes: pd.DataFrame) -> pd.DataFrame:
    """Use WGS for a sample when that sample has WGS data; otherwise use WES.

    Gene rows are the union of symbols kept from either file.
    """
    wgs_removed = int(wgs.attrs.get("genes_removed", 0)) if wgs is not None else 0
    wes_removed = int(wes.attrs.get("genes_removed", 0)) if wes is not None and not wes.empty else 0
    print(
        f"  Removed {wgs_removed} genes from WGS and {wes_removed} genes from WES "
        "(not protein-coding genes in Gene.csv)."
    )
    if wes is None or wes.empty:
        print("  No WES copy-number matrix; combined copy number is WGS only.")
        print(f"  Combined copy number: {shape(wgs)}")
        return wgs
    if wgs is None or wgs.empty:
        print("  No WGS copy-number matrix; combined copy number is WES only.")
        print(f"  Combined copy number: {shape(wes)}")
        return wes
    genes = wgs.index.union(wes.index)
    wgs_samples = [col for col in wgs.columns if wgs[col].notna().any()]
    wgs_sample_set = set(wgs_samples)
    wes_samples = [col for col in wes.columns if col not in wgs_sample_set]
    wgs_part = wgs.reindex(index=genes, columns=wgs_samples)
    wes_part = wes.reindex(index=genes, columns=wes_samples)
    combined = pd.concat([wgs_part, wes_part], axis=1)
    only_wgs = len(wgs.index.difference(wes.index))
    only_wes = len(wes.index.difference(wgs.index))
    both = len(wgs.index.intersection(wes.index))
    print(
        f"  Copy number samples: {len(wgs_samples)} use WGS only; "
        f"{len(wes_samples)} have no WGS data and use WES."
    )
    print(
        f"  Gene symbol union: {len(genes)} "
        f"({both} in both files, {only_wgs} only in WGS, {only_wes} only in WES)."
    )
    print(f"  Combined copy number: {shape(combined)}")
    return combined


def zscore_rows(matrix: pd.DataFrame) -> pd.DataFrame:
    std = matrix.std(axis=1, ddof=0).replace(0, np.nan)
    return matrix.sub(matrix.mean(axis=1), axis=0).div(std, axis=0)


def threshold_high_low(zscores: pd.DataFrame, high_suffix: str, low_suffix: str, cutoff: float = ZSCORE_CUTOFF):
    present = zscores.notna()
    high = pd.DataFrame(np.where(present, (zscores > cutoff).astype(float), np.nan), index=zscores.index, columns=zscores.columns)
    low = pd.DataFrame(np.where(present, (zscores < -cutoff).astype(float), np.nan), index=zscores.index, columns=zscores.columns)
    return _suffix_features(high, high_suffix), _suffix_features(low, low_suffix)


def _suffix_features(gene_by_line: pd.DataFrame, suffix: str) -> pd.DataFrame:
    out = gene_by_line.copy()
    out.index = [f"{gene}_{suffix}" for gene in out.index.astype(str)]
    return out


def _binary_where_known(mask: pd.DataFrame, known: pd.DataFrame | None = None) -> pd.DataFrame:
    if known is None:
        known = mask.notna()
    return pd.DataFrame(np.where(known, mask.fillna(False).astype(float), np.nan), index=mask.index, columns=mask.columns)


def tidy_category(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    convertible = numeric.notna() | series.isna()
    if bool(convertible.all()) and int(numeric.notna().sum()) > 0:
        delta = (numeric - numeric.round()).abs()
        if delta.dropna().empty or float(delta.dropna().max()) < 1e-6:
            text = numeric.round().astype("Int64").astype(str)
            return text.where(numeric.notna(), np.nan)
    return series.where(series.notna()).astype(str)


def one_hot(series: pd.Series, columns: list[str]) -> pd.DataFrame:
    prefix = series.name or "feature"
    cleaned = tidy_category(series)
    dummies = pd.get_dummies(cleaned.dropna()).reindex(columns).astype(float)
    missing = cleaned.reindex(columns).isna()
    dummies.loc[missing] = np.nan
    dummies.columns = [f"{prefix}_{value}" for value in dummies.columns]
    return dummies.T


def cin_decile_labels(cin: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(cin, errors="coerce")
    edges = numeric.quantile([i / 10 for i in range(10)]).to_numpy(dtype=float)

    def label(value):
        if pd.isna(value):
            return np.nan
        idx = int(np.searchsorted(edges, value, side="right") - 1)
        return str(int(min(max(idx, 0), 9)))

    return numeric.map(label)


def _find_column(columns, candidates: tuple[str, ...]) -> str | None:
    compact = {str(c).lower().replace(" ", "").replace("_", ""): c for c in columns}
    for name in candidates:
        if name in compact:
            return compact[name]
    return None


def lof_mask(mutations: pd.DataFrame, model_ids: set[str], gene_lookup: dict[str, str] | None = None) -> pd.DataFrame:
    gene_col = _find_column(
        mutations.columns, ("hugosymbol", "genesymbol", "symbol", "gene", "entrezgeneid", "ensemblgeneid")
    )
    model_col = _find_column(mutations.columns, ("modelid", "depmapid", "cellline"))
    class_col = _find_column(mutations.columns, ("variantclassification", "variantclass", "class"))
    lof_flag = _find_column(mutations.columns, ("likelylof",))
    if not gene_col or not model_col or (not class_col and not lof_flag):
        raise ValueError("OmicsSomaticMutations.csv needs gene, model, and a loss-of-function column.")
    use = mutations[[model_col, gene_col]].copy()
    if lof_flag:
        flag = mutations[lof_flag].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        flag = mutations[class_col].astype(str).isin(LOF_CLASSES)
    use["value"] = flag.astype(int)
    if gene_lookup:
        use["gene"] = use[gene_col].map(lambda value: resolve_gene_label(value, gene_lookup))
    else:
        use["gene"] = use[gene_col].map(_symbol)
    use["cellline"] = use[model_col].astype(str)
    use = use[use["cellline"].isin(model_ids) & use["gene"].notna()]
    if use.empty:
        return pd.DataFrame()
    # A gene seen only with LikelyLoF false is 0. A true call for that gene and model stays 1.
    return use.pivot_table(index="gene", columns="cellline", values="value", aggfunc="max")


def fusion_tables(
    fusions: pd.DataFrame, model_ids: set[str], gene_lookup: dict[str, str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = fusions.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    model_col = _find_column(frame.columns, ("modelid", "depmapid", "cellline"))
    left = _find_column(frame.columns, ("leftgene", "gene1", "gene1symbol", "hugosymbolleft", "5gene"))
    right = _find_column(frame.columns, ("rightgene", "gene2", "gene2symbol", "hugosymbolright", "3gene"))
    if not model_col or not left or not right:
        # Fall back to the first two columns whose names mention gene.
        gene_cols = [c for c in frame.columns if "gene" in str(c).lower()]
        if model_col and len(gene_cols) >= 2:
            left, right = gene_cols[0], gene_cols[1]
        else:
            raise ValueError("OmicsFusionFiltered.csv needs ModelID plus left and right gene columns.")
    use = frame[[model_col, left, right]].dropna()
    use["cellline"] = use[model_col].astype(str)
    use = use[use["cellline"].isin(model_ids)]
    if gene_lookup:
        use["left"] = use[left].map(lambda value: resolve_gene_label(value, gene_lookup))
        use["right"] = use[right].map(lambda value: resolve_gene_label(value, gene_lookup))
    else:
        use["left"] = use[left].map(_symbol)
        use["right"] = use[right].map(_symbol)
    use = use[use["left"].notna() & use["right"].notna()]
    use["feature"] = "FUSION_" + use["left"] + "_" + use["right"]
    pairs = use.drop_duplicates(["feature", "cellline"]).assign(value=1)
    pair_matrix = pairs.pivot_table(index="feature", columns="cellline", values="value", aggfunc="max")
    gene_hits = pd.concat(
        [
            use[["left", "cellline"]].rename(columns={"left": "gene"}),
            use[["right", "cellline"]].rename(columns={"right": "gene"}),
        ],
        ignore_index=True,
    ).drop_duplicates()
    gene_hits["value"] = 1
    gene_matrix = gene_hits.pivot_table(index="gene", columns="cellline", values="value", aggfunc="max")
    return pair_matrix, gene_matrix


def read_protein_matrix(
    path: Path,
    model_ids: set[str],
    name_to_model: dict[str, str],
    gene_lookup: dict[str, str] | None,
) -> pd.DataFrame:
    raw = _read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    id_col = raw.columns[0]
    # Proteomics matrices are model rows by UniProt columns, or the reverse.
    sample_values = raw[id_col].astype(str)
    columns_are_models = sum(match_label_to_model(col, model_ids, name_to_model) is not None for col in raw.columns[1:6]) >= 3
    rows_are_models = sample_values.head(20).map(lambda value: match_label_to_model(value, model_ids, name_to_model) is not None).mean() > 0.5
    if columns_are_models or rows_are_models:
        # Keep UniProt accessions as the feature id. Do not convert them to gene symbols.
        if rows_are_models:
            numeric = _numeric_frame(raw.iloc[:, 1:])
            numeric.columns = [str(c).strip() for c in raw.columns[1:]]
            numeric.index = sample_values.to_numpy()
            numeric = numeric.T
            sample_labels = list(numeric.columns)
            id_source = str(id_col)
        else:
            numeric = _numeric_frame(raw.iloc[:, 1:]).T
            numeric.index = [str(c).strip() for c in raw.columns[1:]]
            numeric.columns = sample_values.to_numpy()
            sample_labels = list(numeric.columns)
            id_source = "column headers"
        print(f"    {path.name}: kept {numeric.shape[0]} UniProt ids (not converted to gene symbols)")
        col_map = {}
        for col in numeric.columns:
            matched = match_label_to_model(col, model_ids, name_to_model)
            if matched:
                col_map[col] = matched
        raw_ids = pd.Index(pd.Series(sample_labels).astype(str)).unique()
        matched_ids = {str(lab) for lab in sample_labels if lab in col_map}
        print(
            f"    {path.name}: sample id is {id_source}; {len(raw_ids)} ids in the file; "
            f"{len(raw_ids) - len(matched_ids)} not in Model.csv; "
            f"{len(set(col_map.values()))} ModelIDs kept"
        )
        if not col_map:
            return pd.DataFrame()
        numeric = numeric[list(col_map)].copy()
        numeric.columns = [col_map[col] for col in numeric.columns]
        numeric = numeric.loc[:, ~numeric.columns.duplicated()]
        return _collapse_duplicate_index(numeric, "mean")
    gene_col = _find_column(raw.columns, ("genesymbol", "gene_symbol", "hugosymbol", "symbol", "gene", "approvedsymbol"))
    if gene_col is None:
        text_cols = [c for c in raw.columns if not pd.api.types.is_numeric_dtype(raw[c])]
        gene_col = text_cols[0] if text_cols else raw.columns[0]
    if gene_lookup:
        genes = raw[gene_col].map(lambda value: resolve_gene_label(value, gene_lookup))
    else:
        genes = raw[gene_col].map(_symbol)
    numeric = raw.drop(columns=[gene_col])
    numeric = numeric.loc[:, numeric.apply(lambda col: pd.to_numeric(col, errors="coerce").notna().mean() > 0.5)]
    numeric = _numeric_frame(numeric)
    numeric.index = genes
    numeric = _collapse_duplicate_index(numeric, "mean")
    col_map = {}
    for col in numeric.columns:
        matched = match_label_to_model(col, model_ids, name_to_model)
        if matched:
            col_map[col] = matched
    if len(col_map) < max(10, int(0.05 * numeric.shape[1])):
        numeric = numeric.T
        row_map = {}
        for lab in numeric.index:
            matched = match_label_to_model(lab, model_ids, name_to_model)
            if matched:
                row_map[lab] = matched
        numeric = numeric.loc[list(row_map)].copy()
        numeric.index = [row_map[lab] for lab in numeric.index]
        numeric = numeric.T
    else:
        numeric = numeric[list(col_map)].copy()
        numeric.columns = [col_map[col] for col in numeric.columns]
    numeric = numeric.loc[:, ~numeric.columns.duplicated()]
    return _collapse_duplicate_index(numeric, "mean")


def _keep_varying(frame: pd.DataFrame | None, columns: list[str]) -> tuple[pd.DataFrame | None, int]:
    """Align a 0/1/NA block to cell lines and drop rows that never take both values.

    Returns the kept float32 block and how many rows were removed.
    """
    if frame is None or frame.empty:
        return None, 0
    aligned = frame.reindex(columns=columns)
    values = aligned.to_numpy(dtype="float32", copy=True)
    ones = np.nansum(values == 1, axis=1)
    zeros = np.nansum(values == 0, axis=1)
    keep = (ones > 0) & (zeros > 0)
    dropped = int((~keep).sum())
    if not keep.any():
        return None, dropped
    kept = pd.DataFrame(values[keep], index=aligned.index[keep], columns=columns)
    return kept, dropped


def _encode_feature_rows(features: pd.DataFrame) -> np.ndarray:
    values = features.to_numpy(dtype="float32", copy=False)
    encoded = np.zeros(values.shape, dtype=np.uint8)
    encoded[np.equal(values, 1)] = 1
    encoded[np.isnan(values)] = 255
    return encoded


def dedupe_identical_rows(features: pd.DataFrame) -> pd.DataFrame:
    encoded = _encode_feature_rows(features)
    names = features.index.astype(str).tolist()
    columns = features.columns
    del features
    gc.collect()
    hashes = hash_feature_rows(encoded)
    groups: dict[int, list[int]] = {}
    name_groups: dict[int, list[str]] = {}
    reps: list[int] = []
    for position, key in enumerate(hashes.tolist()):
        row = encoded[position]
        match = None
        for rep in groups.get(int(key), ()):
            if np.array_equal(row, encoded[rep]):
                match = rep
                break
        if match is None:
            groups.setdefault(int(key), []).append(position)
            name_groups[position] = [names[position]]
            reps.append(position)
        else:
            name_groups[match].append(names[position])
    chosen = encoded[reps]
    decoded = np.full(chosen.shape, np.nan, dtype=np.float32)
    decoded[chosen == 0] = 0
    decoded[chosen == 1] = 1
    index = [":".join(dict.fromkeys(name_groups[rep])) for rep in reps]
    return pd.DataFrame(decoded, index=index, columns=columns)


def _sbs_feature_block(exposures: pd.DataFrame, columns: list[str], name_to_model: dict[str, str], model_ids: set[str]) -> pd.DataFrame:
    matched = []
    keep = []
    for sample in exposures.index:
        model = match_label_to_model(sample, model_ids, name_to_model)
        keep.append(model is not None)
        if model:
            matched.append(model)
    if not matched:
        print("WARNING: no Petljak samples matched DepMap model names.", file=sys.stderr)
        return pd.DataFrame(columns=columns)
    use = exposures.loc[keep].copy()
    use.index = matched
    use = use.groupby(level=0).max()
    blocks = []
    for signature in use.columns:
        series = pd.to_numeric(use[signature], errors="coerce")
        known = series.notna()
        positive = pd.Series(np.where(known, (series > 0).astype(float), np.nan), index=series.index)
        block = positive.reindex(columns).to_frame().T
        block.index = [f"{signature}_True"]
        blocks.append(block)
    return pd.concat(blocks) if blocks else pd.DataFrame(columns=columns)


def _engineer_models(models: pd.DataFrame, signatures: pd.DataFrame, hrd: pd.DataFrame, sbs: pd.DataFrame, name_to_model: dict[str, str]) -> pd.DataFrame:
    models = models.copy()
    models.columns = [str(c).lower().replace(" ", "") for c in models.columns]
    if "modelid" not in models.columns:
        raise ValueError("Model.csv must contain ModelID.")
    models = models.rename(columns={"modelid": "cellline"})
    models["cellline"] = models["cellline"].astype(str)
    models = models.drop_duplicates("cellline").set_index("cellline")

    sig = signatures.copy()
    sig.columns = [str(c).lower().replace(" ", "") for c in sig.columns]
    # OmicsGlobalSignatures.csv has a blank first header; pandas names it Unnamed: 0.
    sig = sig.drop(columns=[c for c in sig.columns if str(c).startswith("unnamed:")])
    sig_id = "modelid" if "modelid" in sig.columns else sig.columns[0]
    sig = sig.rename(columns={sig_id: "cellline"}).drop_duplicates("cellline").set_index("cellline")
    overlap = [c for c in sig.columns if c in models.columns]
    sig = sig.drop(columns=overlap, errors="ignore")
    models = models.join(sig, how="left")

    hrd = hrd.copy()
    hrd.columns = [str(c).lower().replace(" ", "") for c in hrd.columns]
    hrd["cellline"] = hrd["depmap_id"].astype(str)
    hrd_keep = [c for c in hrd.columns if c.startswith("hrdsum")]
    models = models.join(hrd.drop_duplicates("cellline").set_index("cellline")[hrd_keep], how="left")
    score = choose_hrd_score(hrd)
    models["hrdscore"] = score.reindex(models.index)
    high = pd.Series(np.where(models["hrdscore"].notna(), (models["hrdscore"] >= HRD_SCORE_CUTOFF).astype(float), np.nan), index=models.index)
    models["hrdscorehigh"] = high

    ploidy = pd.to_numeric(models["ploidy"], errors="coerce") if "ploidy" in models.columns else pd.Series(np.nan, index=models.index)
    models["ploidyround"] = ploidy.round()
    if "aneuploidy" in models.columns:
        models["aneuploidyby5bin"] = (pd.to_numeric(models["aneuploidy"], errors="coerce") / 5).round()
    if "lohfraction" in models.columns:
        models["lohfractionbyp1bin"] = np.floor(pd.to_numeric(models["lohfraction"], errors="coerce") * 10)
    if "cin" in models.columns:
        models["cinquantilebins"] = cin_decile_labels(models["cin"]).to_numpy()
    models["msihigh"] = _msi_high(models)

    if not sbs.empty:
        matched_rows = {}
        model_ids = set(models.index)
        for sample in sbs.index:
            matched = match_label_to_model(sample, model_ids, name_to_model)
            if matched:
                matched_rows.setdefault(matched, []).append(sample)
        for signature in sbs.columns:
            values = {}
            for model_id, samples in matched_rows.items():
                values[model_id] = pd.to_numeric(sbs.loc[samples, signature], errors="coerce").max()
            models[signature] = pd.Series(values)
    return models.reset_index()


def _msi_high(models: pd.DataFrame) -> pd.Series:
    for column in ("msihigh", "msistatus", "msicall"):
        if column not in models.columns:
            continue
        text = models[column].astype(str).str.upper()
        flag = text.isin(["1", "TRUE", "MSI-H", "MSIH", "HIGH", "MSI"])
        if flag.any():
            return flag.astype(float).where(models[column].notna(), np.nan)
    if "msiscore" in models.columns:
        score = pd.to_numeric(models["msiscore"], errors="coerce")
        return (score >= 10).astype(float).where(score.notna(), np.nan)
    return pd.Series(np.nan, index=models.index)


def _cancer_models(models: pd.DataFrame) -> list[str]:
    if "oncotreeprimarydisease" not in models.columns:
        return models["cellline"].astype(str).tolist()
    disease = models["oncotreeprimarydisease"].astype(str).str.casefold()
    keep = models.loc[disease != "non-cancerous", "cellline"].astype(str)
    return keep.tolist()


def _write_dependency(scores: pd.DataFrame, path: Path) -> None:
    # Loader expects rows = cell lines, columns = genes plus a cellline column.
    wide = scores.T.reset_index()
    first = wide.columns[0]
    wide = wide.rename(columns={first: "cellline"})
    wide.to_parquet(path, compression="gzip", index=False)


def _write_omics_long(
    path: Path,
    columns: list[str],
    frames: dict[str, pd.DataFrame],
    row_name: str = "genes",
) -> None:
    genes = sorted(set().union(*[set(frame.index.astype(str)) for frame in frames.values() if not frame.empty]))
    schema_names = ["cellline", "gene", *frames.keys()]
    string_cols = {"cnv_call"}
    writer = None
    try:
        print(f"  Writing long table for {len(genes)} {row_name} and {len(columns)} cell lines")
        print("  Progress is printed about every 10,000 genes.")
        for start in range(0, len(genes), 400):
            if start == 0 or (start // 400) % 25 == 0 or start + 400 >= len(genes):
                print(f"  Rows {start + 1}-{min(start + 400, len(genes))} of {len(genes)}")
            chunk_genes = genes[start : start + 400]
            melted = None
            for name, frame in frames.items():
                aligned = frame.copy()
                aligned.index = aligned.index.astype(str)
                aligned.index.name = "gene"
                aligned = aligned.reindex(index=chunk_genes, columns=columns)
                piece = aligned.reset_index().melt("gene", var_name="cellline", value_name=name)
                melted = piece if melted is None else melted.merge(piece, on=["gene", "cellline"], how="outer")
            value_cols = [name for name in frames if name not in string_cols]
            melted = melted.dropna(how="all", subset=value_cols)
            if melted.empty:
                continue
            melted = melted[schema_names]
            for name in frames:
                if name in string_cols:
                    melted[name] = melted[name].astype("string")
                else:
                    melted[name] = pd.to_numeric(melted[name], errors="coerce").astype("float32")
            table = pa.Table.from_pandas(melted, preserve_index=False)
            if writer is None:
                parquet = __import__("pyarrow.parquet", fromlist=["ParquetWriter"])
                writer = parquet.ParquetWriter(path, table.schema, compression="gzip")
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pd.DataFrame(columns=schema_names).to_parquet(path, compression="gzip", index=False)


def prepare(data_dir: Path | None = None, depmap_release: str = DEFAULT_DEPMAP_RELEASE) -> None:
    if data_dir is not None:
        os.environ["KOPREDICT_DATA_DIR"] = str(data_dir)
        clear_load_caches()
    data_dir = get_data_dir()
    with log_to_file(data_dir / "prepare.log"):
        _prepare_logged(data_dir, depmap_release)


def _prepare_logged(data_dir: Path, depmap_release: str) -> None:
    print(f"DepMap release: {depmap_release}")
    print(f"Step 1/6: checking raw inputs in {data_dir / 'raw'}")
    verify_downloaded_files(data_dir, depmap_release)
    print("Step 1/6: all required files are present and checksums match.")

    print("Step 2/6: reading Model.csv, Gene.csv, OmicsProfiles.csv, HRD, and SBS exposures")
    models_raw = _read_csv(_raw(data_dir, "Model.csv"))
    print(f"  Model.csv: {shape(models_raw)}")
    genes = _read_csv(_raw(data_dir, "Gene.csv"))
    coding = protein_coding_genes(genes)
    gene_lookup = build_gene_lookup(coding)
    print(
        f"  Gene.csv: {shape(genes)}; kept {len(coding)} protein-coding genes "
        f"({len(gene_lookup)} id-to-symbol keys)"
    )
    profiles = _read_csv(_raw(data_dir, "OmicsProfiles.csv"))
    print(f"  OmicsProfiles.csv: {shape(profiles)}")
    hrd = read_hrd_table(_raw(data_dir, str(additional_record("hrd")["local_filename"])))
    print(f"  HRD table: {shape(hrd)}")
    sbs = load_sbs_exposures(
        _raw(data_dir, str(additional_record("somatic_signatures")["local_filename"]))
    )
    print(f"  SBS exposures: {shape(sbs)}")

    model_col = _find_column(models_raw.columns, ("modelid", "depmapid"))
    if model_col is None:
        raise ValueError("Model.csv must contain ModelID.")
    model_ids = set(models_raw[model_col].astype(str))
    staged = models_raw.copy()
    staged.columns = [str(c).lower().replace(" ", "") for c in staged.columns]
    staged = staged.rename(columns={"modelid": "cellline"})
    name_to_model = build_name_to_model(staged)
    profile_map = profile_to_model_map(profiles)
    condition_map = condition_to_model_map(profiles)
    print(
        f"  {len(model_ids)} model IDs; {len(name_to_model)} name aliases; "
        f"{len(profile_map)} omics profiles; {len(condition_map)} model conditions."
    )

    print("Step 3/6: loading copy number, mutations, expression, dependency, and proteomics")
    print("  Reading OmicsCNGeneWGS.csv")
    wgs = orient_gene_by_model(_read_csv(_raw(data_dir, "OmicsCNGeneWGS.csv")), model_ids, name_to_model, profile_map, condition_map=condition_map, gene_lookup=gene_lookup, label="WGS copy number")
    print(f"  WGS copy number: {shape(wgs)}")
    print("  Reading OmicsCNGeneMC_WES.csv")
    wes = orient_gene_by_model(_read_csv(_raw(data_dir, "OmicsCNGeneMC_WES.csv")), model_ids, name_to_model, profile_map, condition_map=condition_map, gene_lookup=gene_lookup, label="WES copy number")
    print(f"  WES copy number: {shape(wes)}")
    cn = _combine_copy_number(wgs, wes)
    signatures = _read_csv(_raw(data_dir, "OmicsGlobalSignatures.csv"))

    print("  Reading damaging and hotspot mutation matrices")
    damaging = orient_gene_by_model(
        _read_csv(_raw(data_dir, "OmicsSomaticMutationsMatrixDamaging.csv")), model_ids, name_to_model, profile_map, condition_map=condition_map, binary=True, gene_lookup=gene_lookup, label="damaging mutations"
    )
    hotspot = orient_gene_by_model(
        _read_csv(_raw(data_dir, "OmicsSomaticMutationsMatrixHotspot.csv")), model_ids, name_to_model, profile_map, condition_map=condition_map, binary=True, gene_lookup=gene_lookup, label="hotspot mutations"
    )
    print(f"  Damaging mutations: {shape(damaging)}")
    print(f"  Hotspot mutations: {shape(hotspot)}")
    print("  Reading OmicsSomaticMutations.csv for likely loss-of-function")
    lof = lof_mask(_read_csv(_raw(data_dir, "OmicsSomaticMutations.csv")), model_ids, gene_lookup)
    print(f"  Likely LoF: {shape(lof)}")
    print("  Reading OmicsFusionFiltered.csv")
    pair_fusions, gene_fusions = fusion_tables(_read_csv(_raw(data_dir, "OmicsFusionFiltered.csv")), model_ids, gene_lookup)
    print(f"  Fusion pairs: {shape(pair_fusions)}; genes in a fusion: {shape(gene_fusions)}")
    print("  Reading expression, CRISPR, RNAi, and proteomics")
    expr = orient_gene_by_model(
        _read_csv(_raw(data_dir, "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")),
        model_ids,
        name_to_model,
        profile_map,
        condition_map=condition_map,
        gene_lookup=gene_lookup,
        label="expression",
    )
    crispr = orient_gene_by_model(_read_csv(_raw(data_dir, "CRISPRGeneEffect.csv")), model_ids, name_to_model, profile_map, condition_map=condition_map, gene_lookup=gene_lookup, label="CRISPR")
    rnai = orient_gene_by_model(
        _read_csv(_raw(data_dir, "D2_combined_gene_dep_scores.csv")), model_ids, name_to_model, profile_map, condition_map=condition_map, gene_lookup=gene_lookup, label="RNAi"
    )
    sanger = read_protein_matrix(_raw(data_dir, "harmonized_Sanger_MS_2022.csv"), model_ids, name_to_model, gene_lookup)
    gygi = read_protein_matrix(_raw(data_dir, "harmonized_MS_CCLE_Gygi.csv"), model_ids, name_to_model, gene_lookup)
    rppa = read_protein_matrix(_raw(data_dir, "harmonized_RPPA_CCLE.csv"), model_ids, name_to_model, gene_lookup)
    print(f"  Expression: {shape(expr)}")
    print(f"  CRISPR: {shape(crispr)}")
    print(f"  RNAi: {shape(rnai)}")
    print(f"  Sanger proteomics: {shape(sanger)}")
    print(f"  Gygi proteomics: {shape(gygi)}")
    print(f"  RPPA: {shape(rppa)}")

    print("Step 4/6: writing model metadata and CRISPR/RNAi dependency tables")
    model_table = _engineer_models(models_raw, signatures, hrd, sbs, name_to_model)
    model_table.columns = [str(c).lower() for c in model_table.columns]
    model_table.to_parquet(data_dir / _FILE_MODELS, compression="gzip", index=False)
    print(f"  Wrote {_FILE_MODELS}: {shape(model_table)}")

    cancer_lines = list(dict.fromkeys(_cancer_models(model_table)))
    print(f"  Cancer models kept for features: {len(cancer_lines)}")
    _write_dependency(crispr.reindex(columns=cancer_lines), data_dir / _FILE_CRISPR)
    print(f"  Wrote {_FILE_CRISPR}")
    _write_dependency(rnai.reindex(columns=cancer_lines), data_dir / _FILE_RNAI)
    print(f"  Wrote {_FILE_RNAI}")
    print("Step 5/6: building per-gene calls and the long omics table")

    amp = cn >= CN_AMP_CUTOFF
    homdel = cn < CN_HOMDEL_CUTOFF
    known_cn = cn.notna()
    amp_f = pd.DataFrame(np.where(known_cn, amp.astype(float), np.nan), index=cn.index, columns=cn.columns)
    hom_f = pd.DataFrame(np.where(known_cn, homdel.astype(float), np.nan), index=cn.index, columns=cn.columns)
    cnv_call = pd.DataFrame(index=cn.index, columns=cn.columns, dtype=object)
    cnv_call = cnv_call.mask(amp, "Amp")
    cnv_call = cnv_call.mask(homdel, "HomDel")
    cnv_call = cnv_call.where(known_cn)

    expr_z = zscore_rows(expr)
    sanger_z = zscore_rows(sanger) if not sanger.empty else sanger
    gygi_z = zscore_rows(gygi) if not gygi.empty else gygi
    rppa_z = zscore_rows(rppa) if not rppa.empty else rppa

    damaging_c = damaging.reindex(columns=cancer_lines)
    hotspot_c = hotspot.reindex(columns=cancer_lines)
    lof_c = lof.reindex(columns=cancer_lines) if not lof.empty else pd.DataFrame(index=[], columns=cancer_lines)
    gene_fusion_c = gene_fusions.reindex(columns=cancer_lines) if not gene_fusions.empty else pd.DataFrame(index=[], columns=cancer_lines)
    genes_all = sorted(set(amp_f.index).union(damaging_c.index).union(lof_c.index).union(gene_fusion_c.index))
    amp_a = amp.reindex(index=genes_all, columns=cancer_lines).fillna(False)
    hom_a = homdel.reindex(index=genes_all, columns=cancer_lines).fillna(False)
    dam_a = damaging_c.reindex(index=genes_all).fillna(0).gt(0)
    hot_a = hotspot_c.reindex(index=genes_all).fillna(0).gt(0)
    lof_a = lof_c.reindex(index=genes_all).fillna(0).gt(0)
    fus_a = gene_fusion_c.reindex(index=genes_all).fillna(0).gt(0)
    ko = (hom_a | lof_a).astype(float)
    altered = (dam_a | hot_a | amp_a | hom_a | fus_a | ko.astype(bool)).astype(float)

    omics_frames = {
        "cnv": cn,
        "mutdamaging": damaging,
        "muthotspot": hotspot,
        "tpmlogp1": expr,
        "tpmlogp1_zscore": expr_z,
        "altered": altered,
        "ko": ko,
        "crispr": crispr,
        "rnai": rnai,
        "cnv_call": cnv_call,
    }
    protein_frames = {}
    if not sanger_z.empty:
        protein_frames["sanger_proteomics_zscore"] = sanger_z
    if not gygi_z.empty:
        protein_frames["depmap_proteomics_zscore"] = gygi_z
    if not rppa_z.empty:
        protein_frames["rppa_zscore"] = rppa_z
    print(f"  Omics columns: {', '.join(omics_frames)}")
    _write_omics_long(data_dir / _FILE_OMICS, cancer_lines, omics_frames, row_name="genes")
    print(f"  Wrote {_FILE_OMICS}")
    if protein_frames:
        print(f"  Proteomics columns: {', '.join(protein_frames)}")
        _write_omics_long(
            data_dir / _FILE_PROTEOMICS,
            cancer_lines,
            protein_frames,
            row_name="UniProt ids",
        )
        print(f"  Wrote {_FILE_PROTEOMICS}")

    print("Step 6/6: building the binary feature matrix")
    blocks = []
    dropped_constant = 0

    def add(frame: pd.DataFrame | None) -> None:
        nonlocal blocks, dropped_constant
        kept, dropped = _keep_varying(frame, cancer_lines)
        dropped_constant += dropped
        if kept is not None:
            blocks.append(kept)

    model_indexed = model_table.set_index("cellline")
    for column in _METADATA_COLUMNS:
        if column not in model_indexed.columns:
            continue
        add(one_hot(model_indexed[column].rename(column), cancer_lines))
    add(_sbs_feature_block(sbs, cancer_lines, name_to_model, set(model_table["cellline"].astype(str))))
    add(_suffix_features(amp_f, "amp"))
    add(_suffix_features(hom_f, "homdel"))
    del amp_f, hom_f, amp, homdel, known_cn, cn, wgs, wes, cnv_call
    add(_suffix_features(_binary_where_known(damaging_c.gt(0), damaging_c.notna()), "mutdamaging"))
    add(_suffix_features(_binary_where_known(hotspot_c.gt(0), hotspot_c.notna()), "muthotspot"))
    if not lof_c.empty:
        add(_suffix_features(_binary_where_known(lof_c.gt(0), lof_c.notna()), "likelylof"))
    del damaging, hotspot, lof, damaging_c, hotspot_c, lof_c
    high, low = threshold_high_low(expr_z, "highexp", "lowexp")
    add(high)
    add(low)
    del high, low, expr, expr_z
    for matrix, high_name, low_name in (
        (sanger_z, "highsangerprot", "lowsangerprot"),
        (gygi_z, "highdepmapprot", "lowdepmapprot"),
        (rppa_z, "highrppa", "lowrppa"),
    ):
        if matrix.empty:
            continue
        high, low = threshold_high_low(matrix, high_name, low_name)
        add(high)
        add(low)
    del sanger, gygi, rppa, sanger_z, gygi_z, rppa_z, protein_frames
    if not pair_fusions.empty:
        aligned = pair_fusions.reindex(columns=cancer_lines)
        screened = [col for col in aligned.columns if col in pair_fusions.columns]
        aligned.loc[:, screened] = aligned.loc[:, screened].fillna(0)
        add(aligned)
        del aligned
    del pair_fusions, gene_fusions, gene_fusion_c, crispr, rnai
    add(_suffix_features(ko, "ko"))
    add(_suffix_features(altered, "altered"))
    del ko, altered, amp_a, hom_a, dam_a, hot_a, lof_a, fus_a, omics_frames, genes_all
    gc.collect()

    features = pd.concat(blocks, axis=0)
    del blocks
    gc.collect()
    print(f"  Dropped {dropped_constant} constant feature rows")
    print(f"  Feature blocks combined: {shape(features)}")
    before_dedupe = len(features)
    features = dedupe_identical_rows(features)
    print(f"  Merged identical rows: {before_dedupe} -> {len(features)}")
    features.index.name = "feature"
    out_features = data_dir / FEATURES_FILENAME
    features.to_parquet(out_features, compression="gzip", index=True)
    print(f"Wrote {out_features} with shape {features.shape}")
    clear_load_caches()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"Version {VERSION}\n\n"
            "Build the local tables and the binary feature matrix from data/raw.\n\n"
            "  kopredict-prepare\n"
            "  kopredict-prepare --depmap-release 25Q3\n"
            "  kopredict-prepare --help\n\n"
            "Use the same --depmap-release value you passed to kopredict-download."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default=None, help="Processed output directory (raw files are data/raw).")
    parser.add_argument(
        "--depmap-release",
        default=DEFAULT_DEPMAP_RELEASE,
        metavar="QUARTER",
        help=(
            "DepMap Public quarter used to check raw files against metadata/20260920_DepMap_available_downloads.csv "
            f"(default: {DEFAULT_DEPMAP_RELEASE}). Use the same value passed to kopredict-download."
        ),
    )
    args = parser.parse_args(argv)
    prepare(
        Path(args.data_dir).expanduser().resolve() if args.data_dir else None,
        depmap_release=args.depmap_release,
    )


if __name__ == "__main__":
    main()
