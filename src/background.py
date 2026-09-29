"""Background filtering logic (from 02_CHARGR_functions)."""

import io
import pandas as pd
from scipy import stats

from utils import try_float_conversion


class Background:
    """Background cell line filtering based on genomic/omics criteria."""

    def __init__(self, background_string, background_celllines):
        self.background_string = background_string
        self.background_celllines = background_celllines
        self.background_df = pd.DataFrame()
        self.gene_list = []
        self.cancer_types = ["PANCANCER"]

    def __str__(self):
        return_string = []
        for _, row in self.background_df.iterrows():
            return_string.append(
                " ".join(row.drop("DATASET").to_string(index=False, header=False).strip().split())
            )
        return "\n".join(return_string)

    def create_background_df(self):
        self.background_df = pd.read_csv(
            io.StringIO(self.background_string.strip().replace(";", "\n")),
            header=None,
            names=["GENE", "DATASET", "COLUMN_NAME", "GLE", "VALUE", "ZSCORE_FLAG"],
        )
        self.background_df["VALUE"] = self.background_df["VALUE"].astype(object)

    def fill_in_gene_list(self):
        if not self.background_df.empty:
            self.gene_list = self.background_df["GENE"].values.tolist()

    def update_celllines_given_options(self, df):
        for _, row in self.background_df.iterrows():
            if row["DATASET"] == "omics":
                tmp_df = df[df["gene"] == row["GENE"]].dropna(how="any", subset=[row["COLUMN_NAME"]]).copy()
            else:
                tmp_df = df.dropna(how="any", subset=[row["COLUMN_NAME"]]).copy()
            self.background_celllines = list(
                set(self.background_celllines).intersection(set(tmp_df["cellline"].values.tolist()))
            )

    def check_background_df(self, input_df, perc=25):
        if not self.background_df["GLE"].isin(["greater", "less", "equal"]).all():
            print(
                "ERROR: GLE column contains a string that isn't 'greater', 'less', or 'equal'. "
                "Please update the input file"
            )
        for _, row in self.background_df.iterrows():
            if row["COLUMN_NAME"] not in input_df.columns:
                print(
                    f"ERROR: Column {row['COLUMN_NAME']} is not in the inputted dataframe. "
                    "Please check spelling and data"
                )
        self.check_distribution(input_df, perc)

    def check_distribution(self, input_df, perc=25):
        for index, row in self.background_df.iterrows():
            if row["DATASET"] == "omics":
                tmp_df = (
                    input_df[input_df["gene"] == row["GENE"]]
                    .dropna(how="any", subset=[row["COLUMN_NAME"]])
                    .copy()
                )
            else:
                tmp_df = input_df.dropna(how="any", subset=[row["COLUMN_NAME"]]).copy()

            if row["ZSCORE_FLAG"]:
                if len(self.background_celllines) < 50:
                    print(
                        f"WARNING: Given inputted data and cancer types, the samples size is less than 50 "
                        f"({len(self.background_celllines)}).\nSo instead of a Zscore we will select the "
                        f"{perc}% of samples.\nYou can adjust this by specifying the threshold you wish "
                        "to use as 'perc = X' in the function"
                    )
                    self.background_df.loc[index, "ZSCORE_FLAG"] = False
                    if row["GLE"] == "greater":
                        self.background_df.loc[index, "VALUE"] = tmp_df[row["COLUMN_NAME"]].quantile(
                            1 - perc / 100
                        )
                        print(
                            f"New cutoff for {row['COLUMN_NAME']} is: "
                            f"{self.background_df.loc[index, 'VALUE']:.4f}\n\n"
                        )
                    elif row["GLE"] == "less":
                        self.background_df.loc[index, "VALUE"] = tmp_df[row["COLUMN_NAME"]].quantile(
                            perc / 100
                        )
                        print(
                            f"New cutoff for {row['COLUMN_NAME']} is: "
                            f"{self.background_df.loc[index, 'VALUE']:.4f}\n\n"
                        )

    def update_background_celllines(self, df):
        for _, row in self.background_df.iterrows():
            val = row["VALUE"]
            if val == "True" or val == "False":
                val = eval(val)
            else:
                val = try_float_conversion(val)

            if row["DATASET"] == "omics":
                tmp_df = df[df["gene"] == row["GENE"]].dropna(how="any", subset=[row["COLUMN_NAME"]]).copy()
            else:
                tmp_df = df.dropna(how="any", subset=[row["COLUMN_NAME"]]).copy()

            if row["ZSCORE_FLAG"]:
                tmp_df = tmp_df.copy()
                tmp_df["zscore"] = stats.zscore(tmp_df[row["COLUMN_NAME"]])
                if row["GLE"] == "equal":
                    self.background_celllines = list(
                        set(self.background_celllines).intersection(
                            set(tmp_df.loc[(tmp_df["zscore"] == val), "cellline"].values.tolist())
                        )
                    )
                elif row["GLE"] == "greater":
                    self.background_celllines = list(
                        set(self.background_celllines).intersection(
                            set(tmp_df.loc[(tmp_df["zscore"] > float(val)), "cellline"].values.tolist())
                        )
                    )
                elif row["GLE"] == "less":
                    self.background_celllines = list(
                        set(self.background_celllines).intersection(
                            set(tmp_df.loc[(tmp_df["zscore"] < float(val)), "cellline"].values.tolist())
                        )
                    )
            else:
                if row["GLE"] == "equal":
                    self.background_celllines = list(
                        set(self.background_celllines).intersection(
                            set(tmp_df.loc[(tmp_df[row["COLUMN_NAME"]] == val), "cellline"].values.tolist())
                        )
                    )
                elif row["GLE"] == "greater":
                    self.background_celllines = list(
                        set(self.background_celllines).intersection(
                            set(tmp_df.loc[(tmp_df[row["COLUMN_NAME"]] > float(val)), "cellline"].values.tolist())
                        )
                    )
                elif row["GLE"] == "less":
                    self.background_celllines = list(
                        set(self.background_celllines).intersection(
                            set(tmp_df.loc[(tmp_df[row["COLUMN_NAME"]] < float(val)), "cellline"].values.tolist())
                        )
                    )
