"""Feature-construction rules used while preparing the binary matrix."""

import numpy as np
import pandas as pd

from prepare import _combine_copy_number, _keep_varying, dedupe_identical_rows, lof_mask


def test_lof_mask_sets_false_to_zero_and_leaves_absent_pairs_missing():
    mutations = pd.DataFrame(
        {
            "HugoSymbol": ["TP53", "TP53", "TP53", "CDKN2A"],
            "ModelID": ["ACH-1", "ACH-1", "ACH-2", "ACH-1"],
            "LikelyLoF": ["false", "true", "false", "FALSE"],
        }
    )
    calls = lof_mask(mutations, {"ACH-1", "ACH-2", "ACH-3"})
    assert calls.loc["TP53", "ACH-1"] == 1
    assert calls.loc["TP53", "ACH-2"] == 0
    assert calls.loc["CDKN2A", "ACH-1"] == 0
    assert "ACH-3" not in calls.columns
    assert np.isnan(calls.loc["CDKN2A", "ACH-2"])


def test_keep_varying_and_dedupe_identical_rows():
    columns = ["c1", "c2", "c3"]
    frame = pd.DataFrame(
        [
            [1, 1, 1],
            [0, 0, np.nan],
            [np.nan, np.nan, np.nan],
            [1, 0, np.nan],
            [1, 0, 0],
            [1, 0, np.nan],
        ],
        index=["all1", "zeros_and_missing", "all_missing", "same_a", "zero_not_missing", "same_b"],
        columns=columns,
    )
    kept, dropped = _keep_varying(frame, columns)
    assert dropped == 3
    assert set(kept.index) == {"same_a", "zero_not_missing", "same_b"}
    merged = dedupe_identical_rows(kept)
    assert "same_a:same_b" in merged.index
    assert "zero_not_missing" in merged.index
    assert list(merged.loc["zero_not_missing"]) == [1.0, 0.0, 0.0]
    assert np.isnan(merged.loc["same_a:same_b", "c3"])


def test_combine_copy_number_prefers_wgs_when_a_model_has_any_wgs_call():
    wgs = pd.DataFrame({"ACH-1": [1.5, np.nan]}, index=["MTAP", "CDKN2A"])
    wes = pd.DataFrame(
        {"ACH-1": [0.01, 0.2], "ACH-2": [0.5, 1.1]},
        index=["MTAP", "CDKN2A"],
    )
    combined = _combine_copy_number(wgs, wes)
    assert combined.loc["MTAP", "ACH-1"] == 1.5
    assert np.isnan(combined.loc["CDKN2A", "ACH-1"])
    assert combined.loc["MTAP", "ACH-2"] == 0.5
    assert "ACH-2" in combined.columns
