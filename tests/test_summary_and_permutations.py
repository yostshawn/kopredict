"""Summary rows, the summary filename, and the shuffle cutoff."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from cli_impl import load_run_config
from cohort_summary import build_cohort_summary_rows
from features import (
    FeatureNode,
    directional_wilcoxon_table,
    permutation_null_for_seed,
    prepare_permutation_layout,
)
from pipeline import _keep_smaller_direction_pair, _keep_smaller_root_pair
from permutation import clear_cutoff_cache, permutation_cutoffs_from_frames, smoothed_null_cutoff
from plotting import summary_csv_path


def test_default_yaml_sets_effect_size_and_null_percentile():
    config = load_run_config(None)
    assert config.min_effect_size == -0.2
    assert config.resistance_min_effect_size == 0.2
    assert config.strong_min_effect_size == -0.2
    assert config.null_percentile == 5
    assert config.min_group_count == 3
    assert config.jaccard_threshold == 0.9


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


def test_resistant_child_row_is_marked_resistant():
    celllines = ["c1", "c2", "c3", "c4"]
    features = pd.DataFrame([[1, 1, 0, 0]], index=["CDKN2A_ko"], columns=celllines)
    scores = pd.DataFrame({"cellline": celllines, "crispr": [2.0, 1.0, -1.0, -2.0]})
    root = FeatureNode("PPM1D")
    root.add_child(FeatureNode("CDKN2A_ko", sens_res="resistant"))
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
    assert rows[0]["sens_res"] == "resistant"
    assert rows[0]["Wilcoxon_pval"] is not None


def test_root_keeps_the_smaller_feature_when_both_directions_pass():
    columns = [f"c{i}" for i in range(8)]
    features = pd.DataFrame([[1, 1, 0, 0, 0, 0, 0, 0]], index=["TP53_ko"], columns=columns)
    sensitive = pd.DataFrame({"feature": ["TP53_ko_inverse"], "p_value": [0.01]})
    resistant = pd.DataFrame({"feature": ["TP53_ko"], "p_value": [0.01]})
    kept_sensitive, kept_resistant = _keep_smaller_root_pair(sensitive, resistant, features)
    assert kept_sensitive.empty
    assert kept_resistant["feature"].tolist() == ["TP53_ko"]

    common = pd.DataFrame([[1, 1, 1, 1, 1, 1, 0, 0]], index=["B_cell"], columns=columns)
    sensitive = pd.DataFrame({"feature": ["B_cell_inverse"], "p_value": [0.01]})
    resistant = pd.DataFrame({"feature": ["B_cell"], "p_value": [0.01]})
    kept_sensitive, kept_resistant = _keep_smaller_root_pair(sensitive, resistant, common)
    assert kept_resistant.empty
    assert kept_sensitive["feature"].tolist() == ["B_cell_inverse"]


def test_direction_pair_keeps_the_smaller_selected_group():
    columns = [f"c{i}" for i in range(8)]
    features = pd.DataFrame([[1, 1, 0, 0, 0, 0, 0, 0]], index=["TP53_ko"], columns=columns)
    scored = pd.DataFrame(
        {
            "feature": ["TP53_ko", "TP53_ko_inverse"],
            "p_value": [0.01, 0.02],
        }
    )
    kept = _keep_smaller_direction_pair(scored, features)
    assert kept["feature"].tolist() == ["TP53_ko"]


def test_summary_uses_discovery_p_when_present():
    celllines = ["c1", "c2", "c3", "c4"]
    features = pd.DataFrame([[1, 1, 0, 0]], index=["CDKN2A_ko"], columns=celllines)
    scores = pd.DataFrame({"cellline": celllines, "crispr": [-2.0, -1.0, 1.0, 2.0]})
    root = FeatureNode("PPM1D")
    root.add_child(
        FeatureNode(
            "CDKN2A_ko",
            discovery_p=1.23e-5,
            discovery_feature="CDKN2A_ko",
            n_mutant=2,
            n_reference=2,
        )
    )
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
    assert rows[0]["Wilcoxon_pval"] == 1.23e-5
    assert rows[0]["Number_mutant"] == 2
    assert rows[0]["Number_WT"] == 2


def test_summary_csv_path_omits_logic_type(tmp_path):
    path = summary_csv_path("PPM1D", "crispr", ["SOLID"], tmp_path)
    today = date.today().strftime("%Y-%m-%d")
    assert path.endswith(f"{today}_PPM1D_crispr_SOLID_summary.csv")
    assert "_AND_" not in path
    assert "_OR_" not in path
    assert "_boxplots" not in path


def test_smoothed_null_cutoff_means_the_lowest_k_minima():
    values = [0.04, 0.01, 0.03, 0.02, 0.05]
    # percentile 40 -> k = ceil(5 * 0.4) = 2 -> mean of 0.01, 0.02
    assert smoothed_null_cutoff(values, percentile=40) == pytest.approx(0.015)


def test_permutation_cutoff_is_the_smoothed_mean_of_the_best_p_values():
    clear_cutoff_cache()
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
        use_cache=False,
    )
    assert result["pvalue_cutoff"] == smoothed_null_cutoff(result["min_pvals_per_permutation"], 5)
    assert result["qvalue_cutoff"] == smoothed_null_cutoff(result["min_qvals_per_permutation"], 5)

    clear_cutoff_cache()
    stored = permutation_cutoffs_from_frames(
        tdf=tdf,
        fdf=features,
        filter_score_column="crispr_zscore",
        wilcox_score_column="crispr",
        n_permutations=8,
        percentile=5,
        n_jobs=1,
        use_cache=True,
    )
    # Same n_cells reuses the cached cutoff without reshuffling.
    again = permutation_cutoffs_from_frames(
        tdf=tdf,
        fdf=features,
        filter_score_column="crispr_zscore",
        wilcox_score_column="crispr",
        n_permutations=8,
        percentile=5,
        n_jobs=1,
        use_cache=True,
    )
    assert again["pvalue_cutoff"] == stored["pvalue_cutoff"]
    assert again["min_pvals_per_permutation"] == stored["min_pvals_per_permutation"]

    flipped = tdf.copy()
    flipped["cellline"] = ["c4", "c5", "c6", "c7", "c0", "c1", "c2", "c3"]
    flipped_table = directional_wilcoxon_table(features, flipped, "crispr_zscore", "crispr")
    flipped_names = set(flipped_table["feature"])
    assert "ko_inverse" in flipped_names
    assert flipped_table.loc[flipped_table["feature"] == "ko_inverse", "p_value"].iloc[0] == min(
        flipped_table["p_value"]
    )


def test_prepared_shuffle_matches_shuffling_the_cellline_labels():
    celllines = [f"c{i}" for i in range(10)]
    features = pd.DataFrame(
        [
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, np.nan],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        ],
        index=["separated", "missing_cell", "alternating"],
        columns=celllines,
    )
    scores = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.4, 0.8, 1.2, 1.6, np.nan])
    tdf = pd.DataFrame({"cellline": celllines, "crispr": scores})
    layout = prepare_permutation_layout(features, tdf, "crispr")
    matched = 0
    for seed in range(12):
        pvals, qvals = permutation_null_for_seed(layout, seed, min_effect_size=-0.2)
        rng = np.random.default_rng(seed)
        shuffled = tdf.copy()
        shuffled["cellline"] = rng.permutation(tdf["cellline"].to_numpy())
        table = directional_wilcoxon_table(features, shuffled, "crispr", "crispr")
        kept = table[table["p_value"] < 1]
        assert pvals == kept["p_value"].astype(float).tolist()
        assert qvals == kept["q_value"].astype(float).tolist()
        matched += int(len(pvals) > 0)
    assert matched > 0

    finite_scores = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
    finite_tdf = pd.DataFrame({"cellline": celllines, "crispr": finite_scores})
    finite_layout = prepare_permutation_layout(features.fillna(0), finite_tdf, "crispr")
    finite_hits = 0
    for seed in range(12):
        pvals, qvals = permutation_null_for_seed(finite_layout, seed, min_effect_size=-0.2)
        rng = np.random.default_rng(seed)
        shuffled = finite_tdf.copy()
        shuffled["cellline"] = rng.permutation(finite_tdf["cellline"].to_numpy())
        table = directional_wilcoxon_table(features.fillna(0), shuffled, "crispr", "crispr")
        kept = table[table["p_value"] < 1]
        assert pvals == kept["p_value"].astype(float).tolist()
        assert qvals == kept["q_value"].astype(float).tolist()
        finite_hits += int(len(pvals) > 0)
    assert finite_hits > 0
