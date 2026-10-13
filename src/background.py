"""Select background cell lines from OmicsFull_processed.parquet.gz."""

import pandas as pd
from scipy import stats

from utils import try_float_conversion

# Value columns in data/OmicsFull_processed.parquet.gz. cellline and gene identify the row.
OMICS_COLUMNS = (
    "cnv",
    "cnv_call",
    "mutdamaging",
    "muthotspot",
    "tpmlogp1",
    "tpmlogp1_zscore",
    "altered",
    "ko",
    "crispr",
    "rnai",
)
_COMPARISONS = ("greater", "less", "equal")


def _parse_bool(value: str, label: str) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise SystemExit(f"{label} must be true or false (got {value!r}).")


def _gene_rows(df: pd.DataFrame, gene: str, column: str) -> pd.DataFrame:
    return df[df["gene"] == gene].dropna(how="any", subset=[column]).copy()


def _keep(current: list, selected: pd.Series) -> list:
    return list(set(current).intersection(set(selected.tolist())))


class Background:
    """Keep cohort cell lines that pass every OmicsFull rule."""

    def __init__(self, background_string, background_celllines):
        self.background_string = background_string
        self.background_celllines = background_celllines
        self.background_df = pd.DataFrame()
        self.gene_list = []
        self.cancer_types = ["PANCANCER"]

    def __str__(self):
        lines = []
        for _, row in self.background_df.iterrows():
            lines.append(
                f"{row['GENE']} {row['COLUMN_NAME']} {row['GLE']} {row['VALUE']} zscore={row['ZSCORE_FLAG']}"
            )
        return "\n".join(lines)

    def create_background_df(self):
        """Parse ``GENE,omics,COLUMN,greater|less|equal,VALUE,true|false`` rules."""
        rows = []
        allowed = ", ".join(OMICS_COLUMNS)
        for raw in str(self.background_string).split(";"):
            text = raw.strip()
            if not text:
                continue
            parts = [part.strip() for part in text.split(",")]
            if len(parts) != 6:
                raise SystemExit(
                    "A background rule is GENE,omics,COLUMN,greater|less|equal,VALUE,true|false "
                    f"(got {text!r})."
                )
            gene, dataset, column, comparison, value, zscore_flag = parts
            if dataset != "omics":
                raise SystemExit(
                    "Background rules read OmicsFull_processed.parquet.gz only. "
                    f"The dataset field must be omics (got {dataset!r})."
                )
            if column not in OMICS_COLUMNS:
                raise SystemExit(
                    f"Column {column!r} is not in OmicsFull_processed.parquet.gz. "
                    f"Allowed columns: {allowed}."
                )
            if comparison not in _COMPARISONS:
                raise SystemExit(
                    f"Comparison must be greater, less, or equal (got {comparison!r})."
                )
            rows.append(
                {
                    "GENE": gene,
                    "DATASET": "omics",
                    "COLUMN_NAME": column,
                    "GLE": comparison,
                    "VALUE": value,
                    "ZSCORE_FLAG": _parse_bool(zscore_flag, "Z-score flag"),
                }
            )
        if not rows:
            raise SystemExit("group_options has no background rules.")
        self.background_df = pd.DataFrame(rows)

    def fill_in_gene_list(self):
        if not self.background_df.empty:
            self.gene_list = self.background_df["GENE"].tolist()

    def update_celllines_given_options(self, df):
        """Drop cohort lines that have no value for a rule's gene and column."""
        self._require_columns(df)
        for _, row in self.background_df.iterrows():
            tmp_df = _gene_rows(df, row["GENE"], row["COLUMN_NAME"])
            self.background_celllines = _keep(self.background_celllines, tmp_df["cellline"])

    def check_background_df(self, input_df, perc=25):
        self._require_columns(input_df)
        self.check_distribution(input_df, perc)

    def _require_columns(self, df: pd.DataFrame) -> None:
        allowed = ", ".join(OMICS_COLUMNS)
        for column in self.background_df["COLUMN_NAME"]:
            if column not in OMICS_COLUMNS or column not in df.columns:
                raise SystemExit(
                    f"Column {column!r} is not in OmicsFull_processed.parquet.gz. "
                    f"Allowed columns: {allowed}."
                )

    def check_distribution(self, input_df, perc=25):
        for index, row in self.background_df.iterrows():
            if not row["ZSCORE_FLAG"] or len(self.background_celllines) >= 50:
                continue
            tmp_df = _gene_rows(input_df, row["GENE"], row["COLUMN_NAME"])
            print(
                f"WARNING: {len(self.background_celllines)} cell lines is below 50, "
                f"so {row['COLUMN_NAME']} uses the {perc}% quantile instead of a z-score."
            )
            self.background_df.loc[index, "ZSCORE_FLAG"] = False
            if row["GLE"] == "greater":
                cutoff = tmp_df[row["COLUMN_NAME"]].quantile(1 - perc / 100)
            elif row["GLE"] == "less":
                cutoff = tmp_df[row["COLUMN_NAME"]].quantile(perc / 100)
            else:
                continue
            self.background_df.loc[index, "VALUE"] = cutoff
            print(f"New cutoff for {row['COLUMN_NAME']} is: {cutoff:.4f}")

    def update_background_celllines(self, df):
        self._require_columns(df)
        for _, row in self.background_df.iterrows():
            val = row["VALUE"]
            if val == "True" or val == "False":
                val = val == "True"
            else:
                val = try_float_conversion(val)
            tmp_df = _gene_rows(df, row["GENE"], row["COLUMN_NAME"])
            if row["ZSCORE_FLAG"]:
                tmp_df["zscore"] = stats.zscore(tmp_df[row["COLUMN_NAME"]])
                series = tmp_df["zscore"]
            else:
                series = tmp_df[row["COLUMN_NAME"]]
            if row["GLE"] == "equal":
                chosen = tmp_df.loc[series == val, "cellline"]
            elif row["GLE"] == "greater":
                chosen = tmp_df.loc[series > float(val), "cellline"]
            else:
                chosen = tmp_df.loc[series < float(val), "cellline"]
            self.background_celllines = _keep(self.background_celllines, chosen)
