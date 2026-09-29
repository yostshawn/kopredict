"""Summary rows, the summary filename, and the shuffle cutoff."""

from datetime import date

import numpy as np
import pandas as pd

from cli_impl import load_run_config
from cohort_summary import build_cohort_summary_rows
from features import FeatureNode, directional_wilcoxon_table
from permutation import permutation_cutoffs_from_frames
from plotting import summary_csv_path


def test_default_yaml_sets_effect_size_and_null_percentile():
    config = load_run_config(None)
    assert config.min_effect_size == -0.2
    assert config.null_percentile == 5


def test_summary_rows_count_the_and_mask_and_flag_an_empty_tree():
    celllines = ["c1", "c2", "c3", "c4"]
    features = pd.DataFrame([[1, 1, 0, 0]], index=["CDKN2A_ko"], columns=celllines)
    scores = pd.DataFrame({"cellline": celllines, "crispr": [-2.0, -1.0, 1.0, 2.0]})
    root = FeatureNode("PPM1D")
    root.add_child(FeatureNode("CDKN2A_ko"))
    rows = build_cohort_summary_rows(
        feature_tree=root,
        fdf=features,
        tdf_filtered=scores,
        target="PPM1D",
        dataset="crispr",
        cancertype="SOLID",
        oncotree_column="oncotreeprimarydisease",
        logic_type="AND",
        ko_datatype="crispr",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["sens_res"] == "sensitive"
    assert row["Number_mutant"] == 2
    assert row["Number_WT"] == 2
    assert row["Group"] == "CDKN2A_ko"
    assert row["Wilcoxon_pval"] is not None

    flagged = build_cohort_summary_rows(
        feature_tree=None,
        fdf=features,
        tdf_filtered=scores,
        target="PPM1D",
        dataset="crispr",
        cancertype="SOLID",
        oncotree_column="oncotreeprimarydisease",
        logic_type="AND",
        ko_datatype="crispr",
    )
    assert flagged[0]["Group"] == "FLAG - No Significant Features"
    assert flagged[0]["sens_res"] == "sensitive"


def test_summary_csv_path_matches_the_boxplot_name(tmp_path):
    path = summary_csv_path("PPM1D", "crispr", "AND", ["SOLID"], tmp_path)
    today = date.today().strftime("%Y-%m-%d")
    assert path.endswith(f"{today}_PPM1D_crispr_AND_SOLID_summary.csv")
    assert "_boxplots" not in path


def test_permutation_cutoff_is_the_percentile_of_the_best_p_value():
    celllines = [f"c{i}" for i in range(8)]
    scores = np.array([-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0])
    features = pd.DataFrame(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ],
        index=["ko", "other"],
        columns=celllines,
    )
    zscores = (scores - scores.mean()) / scores.std(ddof=0)
    tdf = pd.DataFrame({"cellline": celllines, "crispr": scores, "crispr_zscore": zscores})
    unshuffled = directional_wilcoxon_table(features, tdf, "crispr_zscore", "crispr")
    assert "ko" in set(unshuffled["feature"])
    assert "other_inverse" in set(unshuffled["feature"])

    result = permutation_cutoffs_from_frames(
        tdf=tdf,
        fdf=features,
        filter_score_column="crispr_zscore",
        wilcox_score_column="crispr",
        n_permutations=8,
        percentile=5,
        n_jobs=1,
    )
    assert result["pvalue_cutoff"] == np.percentile(result["min_pvals_per_permutation"], 5)
    assert result["qvalue_cutoff"] == np.percentile(result["min_qvals_per_permutation"], 5)

    flipped = tdf.copy()
    flipped["cellline"] = ["c4", "c5", "c6", "c7", "c0", "c1", "c2", "c3"]
    flipped_table = directional_wilcoxon_table(features, flipped, "crispr_zscore", "crispr")
    flipped_names = set(flipped_table["feature"])
    assert "ko_inverse" in flipped_names
    assert flipped_table.loc[flipped_table["feature"] == "ko_inverse", "p_value"].iloc[0] == min(
        flipped_table["p_value"]
    )
