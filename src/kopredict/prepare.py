"""Build local parquet inputs and the binary feature matrix from public downloads."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from kopredict.local_data import clear_load_caches, get_data_dir
from kopredict.sources import (
    FEATURES_FILENAME,
    HRD_SCORE_CUTOFF,
    LOF_CLASSES,
    ZSCORE_CUTOFF,
    additional_record,
)

_FILE_MODELS = "Model_and_OmicsSignatures.parquet.gz"
_FILE_OMICS = "OmicsFull_processed.parquet.gz"
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


def _name_key(value) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _rename_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    lookup = {c.lower().replace(" ", "").replace("_", ""): c for c in out.columns}
    for key in ("modelid", "depmapid", "profileid"):
        if key in lookup:
            return out.rename(columns={lookup[key]: key})
    first = out.columns[0]
    return out.rename(columns={first: "row_id"})


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


def profile_to_model_map(profiles: pd.DataFrame) -> dict[str, str]:
    frame = _rename_id(profiles)
    frame.columns = [str(c).lower().replace(" ", "").replace("_", "") for c in frame.columns]
    if "profileid" not in frame.columns or "modelid" not in frame.columns:
        return {}
    mapping = {}
    for profile, model in zip(frame["profileid"], frame["modelid"]):
        if pd.notna(profile) and pd.notna(model):
            mapping[str(profile)] = str(model)
    return mapping


def orient_gene_by_model(
    df: pd.DataFrame,
    model_ids: set[str],
    name_to_model: dict[str, str],
    profile_map: dict[str, str],
    *,
    binary: bool = False,
) -> pd.DataFrame:
    """Return a gene-by-ModelID numeric matrix."""
    frame = df.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    compact = {c.lower().replace(" ", "").replace("_", ""): c for c in frame.columns}
    id_name = None
    for key in ("modelid", "depmapid", "profileid", "cellline"):
        if key in compact:
            id_name = compact[key]
            break
    if id_name is None:
        id_name = frame.columns[0]
    labels = frame[id_name].astype(str)
    values = frame.drop(columns=[id_name])
    values.columns = [_symbol(c) for c in values.columns]
    values = _numeric_frame(values)
    values.index = labels

    def _axis_models(axis_labels) -> dict:
        found = {}
        for lab in axis_labels:
            text = str(lab)
            if text in profile_map and profile_map[text] in model_ids:
                found[lab] = profile_map[text]
                continue
            matched = match_label_to_model(text, model_ids, name_to_model)
            if matched:
                found[lab] = matched
        return found

    row_map = _axis_models(values.index)
    col_map = _axis_models(values.columns)
    if len(col_map) > len(row_map):
        values = values.T
        row_map = col_map
    keep = [lab for lab in values.index if lab in row_map]
    values = values.loc[keep]
    values.index = [row_map[lab] for lab in values.index]
    values = values.T
    if values.columns.duplicated().any():
        values = values.T.groupby(level=0).mean().T if not binary else values.T.groupby(level=0).max().T
    values = _collapse_duplicate_index(values, "max" if binary else "mean")
    values = values.dropna(axis=0, how="all").dropna(axis=1, how="all")
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


def load_sbs_exposures(path: Path) -> pd.DataFrame:
    """Sample-by-signature numeric exposures from the Petljak workbook."""
    book = pd.ExcelFile(path)
    for sheet in book.sheet_names:
        for header in range(0, 4):
            preview = pd.read_excel(book, sheet_name=sheet, header=header, nrows=6)
            if _sbs_columns(preview):
                full = pd.read_excel(book, sheet_name=sheet, header=header)
                return _exposures_from_sheet(full)
            # Signature names may sit in the first column with cell lines as headers.
            if preview.shape[1] > 4 and preview.iloc[:, 0].astype(str).str.contains("SBS", case=False).sum() >= 3:
                full = pd.read_excel(book, sheet_name=sheet, header=header)
                return _exposures_from_sheet(full)
    raise ValueError(
        "No SBS exposure sheet in "
        f"{path}. Download supplementary data from {additional_record('somatic_signatures')['citation']}."
    )


def _sbs_columns(frame: pd.DataFrame) -> list:
    found = []
    for col in frame.columns:
        text = str(col).strip().lower().replace(" ", "")
        if _SBS_RE.match(text) or text.startswith("sbs"):
            found.append(col)
    return found


def _exposures_from_sheet(frame: pd.DataFrame) -> pd.DataFrame:
    sbs_cols = _sbs_columns(frame)
    if len(sbs_cols) >= 3:
        id_candidates = [c for c in frame.columns if c not in sbs_cols]
        id_col = id_candidates[0]
        exposures = frame.set_index(id_col)[sbs_cols]
    else:
        first = frame.iloc[:, 0].astype(str)
        if first.str.contains("SBS", case=False).sum() < 3:
            raise ValueError("Sheet does not look like SBS exposures.")
        exposures = frame.set_index(frame.columns[0]).T
    exposures.columns = [str(c).strip().lower().replace(" ", "") for c in exposures.columns]
    exposures = exposures.loc[:, [c for c in exposures.columns if c.startswith("sbs")]]
    return _numeric_frame(exposures)


def to_log2_ratio(cn: pd.DataFrame, ploidy: pd.Series | None) -> pd.DataFrame:
    values = cn.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or np.nanmedian(finite) <= 1.2:
        return cn.astype("float32")
    if ploidy is None:
        scale = pd.Series(2.0, index=cn.columns)
    else:
        scale = pd.to_numeric(ploidy.reindex(cn.columns), errors="coerce").fillna(2.0).clip(lower=0.5)
    return np.log2(cn.div(scale, axis=1).clip(lower=1e-3)).astype("float32")


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
    quantiles = numeric.quantile([i / 10 for i in range(10)])
    edges = quantiles.to_numpy(dtype=float)

    def label(value):
        if pd.isna(value):
            return np.nan
        idx = int(np.searchsorted(edges, value, side="right") - 1)
        idx = min(max(idx, 0), 9)
        return str(int(round(edges_index(idx))))

    def edges_index(idx: int) -> int:
        return idx

    return numeric.map(lambda value: str(int(min(max(np.searchsorted(edges, value, side="right") - 1, 0), 9))) if pd.notna(value) else np.nan)


def _find_column(columns, candidates: tuple[str, ...]) -> str | None:
    compact = {str(c).lower().replace(" ", "").replace("_", ""): c for c in columns}
    for name in candidates:
        if name in compact:
            return compact[name]
    return None


def lof_mask(mutations: pd.DataFrame, model_ids: set[str]) -> pd.DataFrame:
    gene_col = _find_column(mutations.columns, ("hugosymbol", "genesymbol", "symbol", "gene"))
    model_col = _find_column(mutations.columns, ("modelid", "depmapid", "cellline"))
    class_col = _find_column(mutations.columns, ("variantclassification", "variantclass", "class"))
    if not gene_col or not model_col or not class_col:
        raise ValueError("OmicsSomaticMutations.csv needs gene, model, and variant-class columns.")
    hit = mutations[mutations[class_col].astype(str).isin(LOF_CLASSES)][[model_col, gene_col]].dropna()
    hit["gene"] = hit[gene_col].map(_symbol)
    hit["cellline"] = hit[model_col].astype(str)
    hit = hit[hit["cellline"].isin(model_ids)]
    if hit.empty:
        return pd.DataFrame()
    paired = hit.drop_duplicates().assign(value=1)
    return paired.pivot_table(index="gene", columns="cellline", values="value", aggfunc="max")


def fusion_tables(fusions: pd.DataFrame, model_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    use["left"] = use[left].map(_symbol)
    use["right"] = use[right].map(_symbol)
    use = use[(use["left"] != "") & (use["right"] != "") & (use["left"] != "NAN") & (use["right"] != "NAN")]
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
    gene_table: pd.DataFrame | None,
) -> pd.DataFrame:
    raw = _read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    gene_col = _find_column(raw.columns, ("genesymbol", "gene_symbol", "hugosymbol", "symbol", "gene", "approvedsymbol"))
    if gene_col is None:
        text_cols = [c for c in raw.columns if not pd.api.types.is_numeric_dtype(raw[c])]
        gene_col = text_cols[0] if text_cols else raw.columns[0]
    genes = raw[gene_col].map(_symbol)
    if gene_table is not None and genes.str.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]").mean() > 0.5:
        uniprot_col = _find_column(gene_table.columns, ("uniprot", "uniprotswissprot", "uniprotkb"))
        symbol_col = _find_column(gene_table.columns, ("genesymbol", "symbol", "genename", "hugosymbol"))
        if uniprot_col and symbol_col:
            lookup = gene_table.set_index(gene_table[uniprot_col].astype(str).str.upper())[symbol_col]
            genes = genes.map(lambda value: _symbol(lookup.get(value, value)))
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


def _known_mask(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    aligned = frame.reindex(columns=columns)
    return aligned.notna()


def _as_01(frame: pd.DataFrame, columns: list[str], *, positive_only: bool = False) -> pd.DataFrame:
    aligned = frame.reindex(columns=columns)
    if positive_only:
        known = aligned.notna()
        data = np.where(known, (aligned.fillna(0) > 0).astype(float), np.nan)
    else:
        data = aligned.to_numpy(dtype=float)
    return pd.DataFrame(data, index=frame.index, columns=columns)


def _drop_constant_rows(features: pd.DataFrame) -> pd.DataFrame:
    ones = (features == 1).sum(axis=1)
    zeros = (features == 0).sum(axis=1)
    return features.loc[(ones > 0) & (zeros > 0)]


def dedupe_identical_rows(features: pd.DataFrame) -> pd.DataFrame:
    filled = features.fillna(-1)
    keys = pd.util.hash_pandas_object(filled, index=False).to_numpy()
    values = filled.to_numpy()
    source = features.to_numpy()
    names = features.index.astype(str).tolist()
    buckets: dict[object, list[list[int]]] = {}
    for position, key in enumerate(keys):
        groups = buckets.setdefault(key, [])
        placed = False
        row = values[position]
        for members in groups:
            if np.array_equal(row, values[members[0]], equal_nan=True):
                members.append(position)
                placed = True
                break
        if not placed:
            groups.append([position])
    rows = []
    index = []
    for groups in buckets.values():
        for members in groups:
            rows.append(source[members[0]])
            index.append(":".join(dict.fromkeys(names[i] for i in members)))
    return pd.DataFrame(rows, index=index, columns=features.columns)


def _sbs_feature_block(exposures: pd.DataFrame, columns: list[str], name_to_model: dict[str, str], model_ids: set[str]) -> pd.DataFrame:
    rows = []
    index = []
    for sample, matched in ((str(sample), match_label_to_model(sample, model_ids, name_to_model)) for sample in exposures.index):
        if matched:
            rows.append(matched)
            index.append(sample)
    if not rows:
        print("WARNING: no Petljak samples matched DepMap model names.", file=sys.stderr)
        return pd.DataFrame(columns=columns)
    use = exposures.loc[index].copy()
    use.index = rows
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


def _dependency_feature_blocks(scores: pd.DataFrame, columns: list[str], sensitive: str, resistant: str) -> list[pd.DataFrame]:
    aligned = scores.reindex(columns=columns)
    zscores = zscore_rows(aligned)
    return list(threshold_high_low(zscores, resistant, sensitive))


def _write_dependency(scores: pd.DataFrame, path: Path) -> None:
    # Loader expects rows = cell lines, columns = genes plus a cellline column.
    wide = scores.T.reset_index()
    first = wide.columns[0]
    wide = wide.rename(columns={first: "cellline"})
    wide.to_parquet(path, compression="gzip", index=False)


def _write_omics_long(path: Path, columns: list[str], frames: dict[str, pd.DataFrame]) -> None:
    genes = sorted(set().union(*[set(frame.index.astype(str)) for frame in frames.values() if not frame.empty]))
    schema_names = ["cellline", "gene", *frames.keys()]
    string_cols = {"cnv_call"}
    writer = None
    try:
        for start in range(0, len(genes), 400):
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
                writer = pq.ParquetWriter(path, table.schema, compression="gzip")
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pd.DataFrame(columns=schema_names).to_parquet(path, compression="gzip", index=False)


def prepare(data_dir: Path | None = None) -> None:
    if data_dir is not None:
        os.environ["KOPREDICT_DATA_DIR"] = str(data_dir)
        clear_load_caches()
    data_dir = get_data_dir()
    print(f"Preparing inputs in {data_dir}")

    models_raw = _read_csv(_raw(data_dir, "Model.csv"))
    genes = _read_csv(_raw(data_dir, "Gene.csv"))
    profiles = _read_csv(_raw(data_dir, "OmicsProfiles.csv"))
    hrd = read_hrd_table(_raw(data_dir, str(additional_record("hrd")["local_filename"])))
    sbs = load_sbs_exposures(
        _raw(data_dir, str(additional_record("somatic_signatures")["local_filename"]))
    )

    model_col = _find_column(models_raw.columns, ("modelid", "depmapid"))
    if model_col is None:
        raise ValueError("Model.csv must contain ModelID.")
    model_ids = set(models_raw[model_col].astype(str))
    staged = models_raw.copy()
    staged.columns = [str(c).lower().replace(" ", "") for c in staged.columns]
    staged = staged.rename(columns={"modelid": "cellline"})
    name_to_model = build_name_to_model(staged)
    profile_map = profile_to_model_map(profiles)

    print("Loading copy-number, mutations, expression, and dependency matrices...")
    wgs = orient_gene_by_model(_read_csv(_raw(data_dir, "OmicsCNGeneWGS.csv")), model_ids, name_to_model, profile_map)
    wes = orient_gene_by_model(_read_csv(_raw(data_dir, "OmicsCNGeneMC_WES.csv")), model_ids, name_to_model, profile_map)
    cn = wgs.combine_first(wes) if not wes.empty else wgs
    signatures = _read_csv(_raw(data_dir, "OmicsGlobalSignatures.csv"))
    sig_preview = signatures.copy()
    sig_preview.columns = [str(c).lower().replace(" ", "") for c in sig_preview.columns]
    ploidy = None
    if "ploidy" in sig_preview.columns:
        sid = "modelid" if "modelid" in sig_preview.columns else sig_preview.columns[0]
        ploidy = pd.to_numeric(sig_preview.set_index(sid)["ploidy"], errors="coerce")
        ploidy.index = ploidy.index.astype(str)
    cn_log = to_log2_ratio(cn, ploidy)

    damaging = orient_gene_by_model(
        _read_csv(_raw(data_dir, "OmicsSomaticMutationsMatrixDamaging.csv")), model_ids, name_to_model, profile_map, binary=True
    )
    hotspot = orient_gene_by_model(
        _read_csv(_raw(data_dir, "OmicsSomaticMutationsMatrixHotspot.csv")), model_ids, name_to_model, profile_map, binary=True
    )
    lof = lof_mask(_read_csv(_raw(data_dir, "OmicsSomaticMutations.csv")), model_ids)
    pair_fusions, gene_fusions = fusion_tables(_read_csv(_raw(data_dir, "OmicsFusionFiltered.csv")), model_ids)
    expr = orient_gene_by_model(
        _read_csv(_raw(data_dir, "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")),
        model_ids,
        name_to_model,
        profile_map,
    )
    crispr = orient_gene_by_model(_read_csv(_raw(data_dir, "CRISPRGeneEffect.csv")), model_ids, name_to_model, profile_map)
    rnai = orient_gene_by_model(
        _read_csv(_raw(data_dir, "D2_combined_gene_dep_scores.csv")), model_ids, name_to_model, profile_map
    )
    sanger = read_protein_matrix(_raw(data_dir, "harmonized_Sanger_MS_2022.csv"), model_ids, name_to_model, genes)
    gygi = read_protein_matrix(_raw(data_dir, "harmonized_MS_CCLE_Gygi.csv"), model_ids, name_to_model, genes)
    rppa = read_protein_matrix(_raw(data_dir, "harmonized_RPPA_CCLE.csv"), model_ids, name_to_model, genes)

    print("Writing model, dependency, and omics tables...")
    model_table = _engineer_models(models_raw, signatures, hrd, sbs, name_to_model)
    model_table.columns = [str(c).lower() for c in model_table.columns]
    model_table.to_parquet(data_dir / _FILE_MODELS, compression="gzip", index=False)

    cancer_lines = list(dict.fromkeys(_cancer_models(model_table)))
    _write_dependency(crispr.reindex(columns=cancer_lines), data_dir / _FILE_CRISPR)
    _write_dependency(rnai.reindex(columns=cancer_lines), data_dir / _FILE_RNAI)

    amp = cn_log >= 1
    homdel = cn_log <= -1
    known_cn = cn_log.notna()
    amp_f = pd.DataFrame(np.where(known_cn, amp.astype(float), np.nan), index=cn_log.index, columns=cn_log.columns)
    hom_f = pd.DataFrame(np.where(known_cn, homdel.astype(float), np.nan), index=cn_log.index, columns=cn_log.columns)
    cnv_call = pd.DataFrame(index=cn_log.index, columns=cn_log.columns, dtype=object)
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
        "cnv": cn_log,
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
    if not sanger_z.empty:
        omics_frames["sanger_proteomics_zscore"] = sanger_z
    if not gygi_z.empty:
        omics_frames["depmap_proteomics_zscore"] = gygi_z
    if not rppa_z.empty:
        omics_frames["rppa_zscore"] = rppa_z
    _write_omics_long(data_dir / _FILE_OMICS, cancer_lines, omics_frames)

    print("Building the binary feature matrix...")
    blocks = []
    model_indexed = model_table.set_index("cellline")
    for column in _METADATA_COLUMNS:
        if column not in model_indexed.columns:
            continue
        blocks.append(one_hot(model_indexed[column].rename(column), cancer_lines))
    blocks.append(_sbs_feature_block(sbs, cancer_lines, name_to_model, set(model_table["cellline"].astype(str))))
    blocks.append(_suffix_features(amp_f.reindex(columns=cancer_lines), "amp"))
    blocks.append(_suffix_features(hom_f.reindex(columns=cancer_lines), "homdel"))
    blocks.append(_suffix_features(_binary_where_known(damaging_c.gt(0), damaging_c.notna()), "mutdamaging"))
    blocks.append(_suffix_features(_binary_where_known(hotspot_c.gt(0), hotspot_c.notna()), "muthotspot"))
    if not lof_c.empty:
        blocks.append(_suffix_features(_binary_where_known(lof_c.gt(0), lof_c.notna()), "likelylof"))
    high, low = threshold_high_low(expr_z.reindex(columns=cancer_lines), "highexp", "lowexp")
    blocks.extend([high, low])
    for matrix, high_name, low_name in (
        (sanger_z, "highsangerprot", "lowsangerprot"),
        (gygi_z, "highdepmapprot", "lowdepmapprot"),
        (rppa_z, "highrppa", "lowrppa"),
    ):
        if matrix.empty:
            continue
        blocks.extend(threshold_high_low(matrix.reindex(columns=cancer_lines), high_name, low_name))
    if not pair_fusions.empty:
        aligned = pair_fusions.reindex(columns=cancer_lines)
        screened = [col for col in aligned.columns if col in pair_fusions.columns]
        aligned.loc[:, screened] = aligned.loc[:, screened].fillna(0)
        blocks.append(aligned)
    blocks.extend(_dependency_feature_blocks(crispr, cancer_lines, "sensitivecrispr", "resistantcrispr"))
    blocks.extend(_dependency_feature_blocks(rnai, cancer_lines, "sensitivernai", "resistantrnai"))
    blocks.append(_suffix_features(ko.reindex(columns=cancer_lines), "ko"))
    blocks.append(_suffix_features(altered.reindex(columns=cancer_lines), "altered"))

    features = pd.concat([block for block in blocks if block is not None and not block.empty], axis=0)
    features = features.loc[:, [c for c in cancer_lines if c in features.columns]]
    features = _drop_constant_rows(features)
    features = dedupe_identical_rows(features)
    features.index.name = "feature"
    out_features = data_dir / FEATURES_FILENAME
    features.to_parquet(out_features, compression="gzip", index=True)
    print(f"Wrote {out_features} with shape {features.shape}")
    clear_load_caches()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build kopredict parquet inputs from data/raw.")
    parser.add_argument("--data-dir", default=None, help="Processed output directory (raw files are data/raw).")
    args = parser.parse_args(argv)
    prepare(Path(args.data_dir).expanduser().resolve() if args.data_dir else None)


if __name__ == "__main__":
    main()
