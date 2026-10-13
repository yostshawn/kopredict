# kopredict

kopredict finds genomic features linked to CRISPR or RNAi sensitivity. The search uses a Wilcoxon test, Jaccard grouping, and a feature tree. `kopredict-download` and `kopredict-prepare` rebuild the local inputs from public files so a clone can run without a private database.

## Install

```bash
pip install -e .
```

Python 3.9 or newer. The same commands work inside the Docker image.

## Data

Processed files go in `data/` (override with `KOPREDICT_DATA_DIR`). Raw downloads go in `data/raw/`. Cell-line ids are DepMap `ModelID` values such as `ACH-000001`, stored in a column named `cellline`.

### 1. DepMap catalog

`metadata/20260920_DepMap_available_downloads.csv` is the DepMap file list, including signed download URLs. Those URLs expire. Download a fresh copy from [https://depmap.org/portal/api/download/files](https://depmap.org/portal/api/download/files) in a browser and replace that file before running the download, and again whenever a link starts failing. The date in the filename is the day that catalog was saved. Three older files in that catalog have no `md5_hash` from DepMap (`D2_combined_gene_dep_scores.csv`, `sanger-dose-response.csv`, and `sanger-viability.csv`). Those three checksums were computed locally and written into the catalog. If you replace the catalog with a newer download, add `md5_hash` values for any row that is blank, or prepare will stop on those files.

`metadata/default_depmap_files_to_download_and_process.csv` lists which rows of that catalog to download. A release value of `{depmap}` follows `--depmap-release`. Other rows, such as DEMETER2, GDSC, and proteomics, keep the release name written in the CSV. Add or remove files there rather than in the Python code.

`metadata/additional_datasets_to_download.csv` lists workbooks that are not in the DepMap catalog. Each row has a local filename, a citation, a file URL, and whether `kopredict-download` should fetch it. The HRD table is downloaded from its URL in that CSV. Somatic signature activities come from Petljak et al., Cell 2019, supplementary file [mmc3.xlsx](https://www.cell.com/cms/10.1016/j.cell.2019.02.012/attachment/4e70a2ab-6758-47e5-a843-802a5474bc82/mmc3.xlsx) ([paper](https://www.cell.com/cell/fulltext/S0092-8674(19)30161-8)). Save that file as `data/raw/mmc3.xlsx`. Prepare reads the sheet `COSMIC CellLines Signatures`. Cell.com often blocks a scripted download, so if the fetch fails, save that same URL by hand.

```bash
kopredict-download
kopredict-download --depmap-release 25Q3
```

`--depmap-release` selects the DepMap Public quarter (default `26Q1`). Pass a label such as `26Q1` or `25Q3`. The command checks that every `{depmap}` filename exists in that release and stops with an error when one does not. Older quarters often use different copy-number and expression filenames.

The HRD score used is `hrdsum_summary` when it is present, otherwise `hrdsum_broad`. A score of 42 or higher is the high-HRD feature. Copy number uses the raw ratio from the WGS or WES file: amplification is a value of at least 2 (`CN_AMP_CUTOFF`), and homozygous deletion is a value below 0.1 (`CN_HOMDEL_CUTOFF`). On the `COSMIC CellLines Signatures` sheet, an exposure greater than 0 becomes a `{signature}_True` feature. Sample names are matched to DepMap models by stripped cell-line name.

GDSC dose-response and viability are downloaded and left in `data/raw/`. Drug-response search is not implemented yet.

### 2. Build the matrices

```bash
kopredict-prepare
kopredict-prepare --depmap-release 25Q3
```

This first checks every file in `metadata/default_depmap_files_to_download_and_process.csv` and `metadata/additional_datasets_to_download.csv`. A missing file, a missing `md5_hash` for a catalog file, or a checksum that does not match stops the command and lists each problem. Use the same `--depmap-release` value you used for `kopredict-download`.

This writes:

| File | Role |
| --- | --- |
| `Model_and_OmicsSignatures.parquet.gz` | Model metadata, global signatures, HRD, and SBS exposures |
| `OmicsFull_processed.parquet.gz` | Long gene-symbol table used by the `group_options` background filter (copy number, mutations, expression, CRISPR, RNAi). Proteomics are not in this file. |
| `Proteomics_processed.parquet.gz` | Long UniProt table of Sanger, Gygi, and RPPA z-scores. The `gene` column holds accessions. |
| `CRISPRGeneEffect_processed.parquet.gz` | CRISPR gene effect |
| `D2_combined_gene_dep_scores_processed.parquet.gz` | DEMETER2 RNAi scores |
| `final_depmap_features_26Q1.parquet.gz` | Binary features (rows) by cell line (columns) |

Non-cancer models (`OncotreePrimaryDisease` of `Non-Cancerous`) are omitted from the feature matrix. How each raw file becomes a feature is described below.

These old DataHub feature families are not rebuilt, because the pinned files do not contain them: `has_hessdriver`, `has_oncogenehighimpact`, `has_tumorsuppressorhighimpact`.

### Feature construction

`kopredict-prepare` writes `final_depmap_features_26Q1.parquet.gz` for the sensitivity search. Each row is one feature and each column is a cancer cell line (`ModelID`). A cell is 1, 0, or missing. Sample identifiers in each file are mapped to `ModelID` with `Model.csv`, `OmicsProfiles.csv` (including `ModelConditionID` for WES copy number), and stripped cell-line name aliases.

Gene features use only `Gene.csv` rows whose `locus_group` is `protein-coding gene`. DepMap headers such as `SYMBOL (NCBI)` are mapped to the HGNC symbol in that table. Proteomics features keep the UniProt accession and are not mapped to symbols. A discretized row is dropped when it is only 1s, only 0s, or only missing, because the search needs both values. Identical remaining rows are collapsed into one row and their names are joined with `:`.

The search tests these features against the continuous CRISPR or RNAi score of the target gene. Those scores are not converted into features. By default the search drops feature names containing `lowexp`, `highexp`, `depmapprot`, `sangerprot`, or `rppa`.

Expression and protein z-scores are computed across every model in that assay. A z-score uses the population standard deviation. The cutoff is 1.64 (`ZSCORE_CUTOFF`).

| Feature | Raw source | Gene symbols | Manipulation | Discretization |
| --- | --- | --- | --- | --- |
| Copy number | `OmicsCNGeneWGS.csv`, otherwise `OmicsCNGeneMC_WES.csv` | Y | If a model has any WGS value, its WES values are not used. Genes are the union of the two files. | `{gene}_amp` is 1 when the copy-number ratio is at least 2. `{gene}_homdel` is 1 when the ratio is below 0.1. Otherwise 0 when the ratio is present, and missing when it is not. |
| Damaging mutation | `OmicsSomaticMutationsMatrixDamaging.csv` | Y | Counts are aligned to cancer models. | `{gene}_mutdamaging` is 1 when the count is greater than 0, else 0. Missing when the cell is missing. |
| Hotspot mutation | `OmicsSomaticMutationsMatrixHotspot.csv` | Y | Counts are aligned to cancer models. | `{gene}_muthotspot` is 1 when the count is greater than 0, else 0. Missing when the cell is missing. |
| Likely loss of function | `OmicsSomaticMutations.csv` | Y | If a `LikelyLoF` column is present, a true flag is a loss-of-function call. Otherwise the variant class must be Nonsense_Mutation, Frame_Shift_Del, Frame_Shift_Ins, Splice_Site, or Nonstop_Mutation. A gene and model with any true call is 1. A gene and model present in the file with only false calls is 0. | `{gene}_likelylof` is 1 or 0 for gene–model pairs in the mutation file. Pairs that are not in the file stay missing. |
| Gene fusion | `OmicsFusionFiltered.csv` | Y | Left and right genes are mapped to symbols. A model that appears in the fusion file and lacks a given pair is set to 0. | `FUSION_{left}_{right}` is 1 when that pair is reported. |
| Expression | `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` | Y | Per-gene z-score of log TPM. | `{gene}_highexp` is 1 when z is greater than 1.64. `{gene}_lowexp` is 1 when z is less than -1.64. Otherwise 0 when expression is present, and missing when it is not. |
| Sanger proteomics | `harmonized_Sanger_MS_2022.csv` | N | Per-protein z-score. The feature id stays the UniProt accession. | `{accession}_highsangerprot` is 1 when z is greater than 1.64. `{accession}_lowsangerprot` is 1 when z is less than -1.64. Otherwise 0 when measured, and missing when not. |
| Gygi proteomics | `harmonized_MS_CCLE_Gygi.csv` | N | Per-protein z-score. The feature id stays the UniProt accession. | `{accession}_highdepmapprot` is 1 when z is greater than 1.64. `{accession}_lowdepmapprot` is 1 when z is less than -1.64. Otherwise 0 when measured, and missing when not. |
| RPPA | `harmonized_RPPA_CCLE.csv` | N | Per-protein z-score. The feature id stays the antibody or UniProt id stored in the file. | `{id}_highrppa` is 1 when z is greater than 1.64. `{id}_lowrppa` is 1 when z is less than -1.64. Otherwise 0 when measured, and missing when not. |
| SBS signatures | `mmc3.xlsx`, sheet `COSMIC CellLines Signatures` | N | Sample names are matched to `ModelID`. If several rows match one model, the maximum exposure is kept. | `{signature}_True` is 1 when the exposure is greater than 0, else 0 when the exposure is present. Models not in the sheet are missing. |
| HRD | `crc-24-0316_table_s5_suppst5.xlsx`, sheet `table_s4` | N | Use `hrdsum_summary` when present, otherwise `hrdsum_broad`. | `hrdscorehigh_1` is the high-HRD state when the score is at least 42. `hrdscorehigh_0` is the other state when a score is present. Missing scores stay missing. |
| MSI | `OmicsGlobalSignatures.csv` | N | Use an MSI-high status flag when one is present. Otherwise a score of at least 10 is high. | `msihigh_1` and `msihigh_0` are the one-hot states. Missing status stays missing. |
| Global signatures | `OmicsGlobalSignatures.csv` | N | Ploidy is rounded to the nearest integer. Aneuploidy is divided by 5 and rounded. CIN is cut into deciles 0–9. LOH fraction is multiplied by 10 and truncated. WGD is used as recorded. | Each value is one-hot encoded as `{measure}_{value}`. Missing values stay missing. The feature names are `ploidyround`, `aneuploidyby5bin`, `cinquantilebins`, `lohfractionbyp1bin`, and `wgd`. |
| Cell-line annotations | `Model.csv` | N | Whole-number categories are stored as integers. | One-hot. Prefixes are `oncotreelineage`, `oncotreeprimarydisease`, `oncotreesubtype`, `oncotreecode`, `agecategory`, `patientrace`, `race`, `primaryormetastasis`, `samplecollectionsite`, and `growthpattern`. A missing annotation stays missing. |
| Knockout | Derived from homozygous deletion and likely loss of function | Y | A missing call is treated as absent. | `{gene}_ko` is 1 when the gene is homozygously deleted or likely loss of function, else 0. |
| Altered | Derived from damaging mutation, hotspot mutation, amplification, homozygous deletion, fusion, and knockout | Y | A missing call is treated as absent. | `{gene}_altered` is 1 when any of those calls is 1, else 0. |

### Inverse features

Each feature is tested in two directions against the raw CRISPR or RNAi score.

The forward test asks whether cell lines with a `1` have lower scores than cell lines with a `0`. A forward hit keeps its name. The selected cell lines are the `1`s.

The inverse test uses the cell lines with a `0` as the selected group. An inverse hit swaps that row: every `0` becomes `1`, every `1` becomes `0`, and missing cells stay missing. The new name adds `_inverse` to every piece split on `;` and `:`. Later steps use this swapped row, so a `1` on an inverse feature is a cell line that was `0` on the original feature. A merged name such as `MTAP_homdel:MTAP_ko` becomes `MTAP_homdel_inverse:MTAP_ko_inverse`.

A sensitive search uses the lower tail, so the selected group has lower scores. A resistant search uses the upper tail, so the selected group has higher scores. When the resistant group is the original `0`s, the name still gets `_inverse` on every piece, and those summary rows are `resistant`.

When a feature and its inverse both pass the same step, the search keeps the direction whose selected group has fewer cell lines. A tie keeps the forward name. Choosing one direction excludes the other for the rest of that branch.

### P-value cutoffs

A step tests the features that pass its effect-size filter and have at least `min_group_count` cell lines on each side (default `3`). The test is a Wilcoxon rank-sum test. Sensitivity uses the lower tail. Resistance uses the upper tail. q-values are Benjamini–Hochberg values computed from the p-values at that step.

The p-value cutoff and the q-value cutoff come from shuffling the dependency-score labels `n_permutations` times (default `50`). Each shuffle uses the same effect-size filter, the same tail, and the same comparison group as the real test, then records its smallest p-value and its smallest q-value.

The cutoff is the mean of the lowest `ceil(n_permutations * null_percentile / 100)` of those minima. With the defaults, `ceil(50 * 5 / 100)` is `3`, so each cutoff is the mean of the three smallest null minima. A smaller `null_percentile` is stricter. A feature is kept when its p-value is below the p-value cutoff and its q-value is below the q-value cutoff.

The full cohort is shuffled for the root. Each later cell-line subset is shuffled again, separately for sensitivity and for resistance, and separately in the AND tree and the OR tree. A subset reuses an earlier cutoff when the cell-line count is within 5% and the direction, effect-size comparison, and effect-size threshold match. A child is kept when it passes that subset's own cutoff.

### Grouping features

Features that pass a step are placed in groups before they become tree nodes. Sensitivity and resistance are grouped separately. Grouping does not change a feature's direction.

The similarity is the Jaccard overlap of cell lines called `1`. It is the number of cell lines called `1` in both features divided by the number called `1` in either feature. A missing cell stays out of both the overlap and the union.

Features are considered from the smallest Wilcoxon p-value to the largest. The first feature starts a group and is that group's seed. Each later feature joins the first existing group whose seed has Jaccard similarity at or above `jaccard_threshold` (default `0.8`). A feature that matches no seed starts a new group.

Within one scored table, if a feature and its inverse both passed, only the direction with fewer selected cell lines is sent to grouping.

### How features are added

The first marker can be sensitive or resistant. The root runs both searches on the full cohort. A sensitive root marker must fall `min_effect_size` below the cohort mean (default `-0.2`). A resistant root marker must rise `resistance_min_effect_size` above it (default `0.2`).

A change of direction uses 0.2. A resistant child under a sensitive marker must rise `resistance_min_effect_size` above that parent. A later step that stays sensitive must fall `child_min_effect_size` below its parent (default `-0.1`). A later step that stays resistant must rise `child_resistance_min_effect_size` above its parent (default `0.1`).

A sensitive node can have both a sensitive child and a resistant child. A child is tested in the cell lines that already carry every marker on the path to its parent. A resistant child is added under a sensitive parent when that parent's effect size is at or below `strong_min_effect_size` (default `-0.2`, the same cutoff the parent already passed). A resistant parent can have another resistant child in that same subset. Each child uses shuffles in its own direction.

Root scoring and grouping run once. From those same first-level markers the search grows an AND tree and an OR tree. AND continues inside the cell lines that carry the parent marker and uses parent-relative child cutoffs (`child_min_effect_size` / `child_resistance_min_effect_size`, default ±0.1). OR continues in the remaining cell lines. An OR same-direction sensitive child must fall `min_effect_size` (default `-0.2`) below its WT group. An OR direction switch uses ±0.2 versus the parent mean. Summary rows keep `Logic_type` as `AND` or `OR`. Each logic writes its own boxplot PDF.

The diagrams below show the rule for adding each feature. `X` is that step's permutation p-value cutoff.

![AND tree. Child nodes contain both the parent feature and the child feature.](example/and_tree.svg)

![OR tree. Child nodes do not contain the parent feature.](example/or_tree.svg)

The tree expands siblings best first. At every row, the search under a sibling cannot see that sibling or any earlier sibling on the row (sensitive and resistant children alike), and features excluded by a parent stay excluded for the rest of that branch. A chosen feature also excludes its other direction. Boxplots and summary Wilcoxon p-values use the discovery stats stored on each tree node.

## Run a search

```bash
kopredict
kopredict path/to/run.yaml
kopredict --help
```

Search settings live in [`metadata/kopredict_default.yaml`](metadata/kopredict_default.yaml). That file runs target `PRMT5`, dataset `rnai`, and the cohorts in [`metadata/oncotree_cohorts_solid.csv`](metadata/oncotree_cohorts_solid.csv) (`cancer_types` is null). Copy it and pass the copy as the argument above. Set either `cancer_types` or `cohort_csv`, not both. A copied file can omit keys; missing keys are filled from the default file.

Changing direction uses `min_effect_size` (default `-0.2`) and `resistance_min_effect_size` (default `0.2`). That includes the first marker, which can be sensitive or resistant, and a resistant child under a sensitive marker. Continuing in the same direction uses `child_min_effect_size` (default `-0.1`) and `child_resistance_min_effect_size` (default `0.1`). `strong_min_effect_size` is how negative a sensitive parent's effect must be before a resistant child is added (default `-0.2`). `null_percentile` sets how many of the lowest null minima are averaged for the cutoff (default `5`). `n_permutations` is how many shuffles are run for the full cohort and again for each later cell-line subset (default `50`). `min_group_count` (default `3`) is the minimum mutant and WT count after the effect filter. `jaccard_threshold` (default `0.8`) controls how tightly features are grouped.

Which cohort strings are valid is listed in [`metadata/oncotree.yaml`](metadata/oncotree.yaml). That file is from DepMap Public 26Q1 `Model.csv`. The `oncotree_column` list names the four columns the run file accepts: `oncotreelineage`, `oncotreeprimarydisease`, `oncotreesubtype`, and `oncotreecode`. Under each column, every value is paired with the number of cell lines that have that exact string.

To run one cohort, set `oncotree_column` and `cancer_types` in the run YAML. `cancer_types` can be one value or a list. The match is exact, and a list keeps a cell line when it matches any value. `SOLID` and `PANCANCER` do not use `oncotree_column`. `PANCANCER` is every model whose primary disease does not contain `NON-CANCER` (2,003 models). `SOLID` is that same set with the hematologic primary diseases listed in [`metadata/oncotree.yaml`](metadata/oncotree.yaml) removed (1,939 models).

To run several cohorts in one command, set `cancer_types` to null and set `cohort_csv` to a CSV path. A relative path such as `metadata/oncotree_cohorts_solid.csv` is resolved from the source directory, so it still works after the package is installed. An absolute path is used as given. Each row is one analysis. The required columns are `oncotree_column` and `name`. When `output_name` is present, it is the result-folder name and the summary label; otherwise the folder and label use `name`. Other columns are ignored. A `name` may be one value or several values separated by semicolons. A semicolon list is combined the same way as a `cancer_types` list. Every `oncotree_column` and `name` pair must exist. If any name in a list does not, the error lists the missing pairs and the search does not run. For `SOLID` or `PANCANCER`, leave `oncotree_column` blank. Values that contain a comma must be quoted.

[`metadata/oncotree_cohorts_solid.csv`](metadata/oncotree_cohorts_solid.csv) is the cohort file used by the default run. The first row is `PANCANCER`. The other rows are solid primary diseases and subtypes from DepMap Public 26Q1. `depmap_26q1_cell_lines` and `cohort_kind` are not used by the search. A semicolon list looks like this:

```csv
oncotree_column,name,output_name
oncotreeprimarydisease,Non-Small Cell Lung Cancer,NSCLC
oncotreecode,"LUCA;GCLC;LCLC;LUMEC",Lung codes
```

`kopredict --help`, `kopredict-download --help`, and `kopredict-prepare --help` print a short how-to and the settings or flags each command accepts.

By default the search drops feature names containing `absolute`, `lowexp`, `highexp`, `depmapprot`, `sangerprot`, or `rppa`. Set `keep_expression_features` or `keep_proteiomic_features` to true in the YAML file to keep those families (`rppa` follows the protein flag).

Plots and a summary CSV are written under `data/output/{target}/{dataset}/` unless `output_dir` is set in the YAML file. That folder sits inside `data/`, which git ignores. Each run writes AND and OR boxplot PDFs and one combined summary CSV. Boxplot names include `AND` or `OR`. The summary name omits that token and puts both trees in one file, with `Logic_type` set per row. `sens_res` is `sensitive` on sensitivity rows and `resistant` on resistance rows.

## Example outputs

Run the examples from the repository root after `pip install -e .` and `kopredict-prepare`. The prepared matrices must already be in `data/`. Both files use the cohorts in [`metadata/oncotree_cohorts_solid.csv`](metadata/oncotree_cohorts_solid.csv) and the same search settings as [`metadata/kopredict_default.yaml`](metadata/kopredict_default.yaml). `n_jobs` is 16 so the shuffles run together. The saved results under `example/` are from these commands.

```bash
kopredict example/prmt5_rnai.yaml
kopredict example/prmt5_rnai_expr_prot.yaml
```

`example/prmt5_rnai.yaml` leaves expression and proteomics off and writes to `example/prmt5_rnai/`. `example/prmt5_rnai_expr_prot.yaml` sets `keep_expression_features` and `keep_proteiomic_features` to true and writes to `example/prmt5_rnai_expr_prot/`. Each cohort with a marker tree gets an AND boxplot PDF and an OR boxplot PDF in a folder named with that cohort's `output_name`. A cohort with no passing marker, or too few essential cell lines, is a `FLAG` row in the summary and has no boxplot.

The genomic summary is [`example/prmt5_rnai/PRMT5/rnai/261012_PRMT5_rnai_cohort_prediction_summary.csv`](example/prmt5_rnai/PRMT5/rnai/261012_PRMT5_rnai_cohort_prediction_summary.csv). The expression-and-proteomics summary is [`example/prmt5_rnai_expr_prot/PRMT5/rnai/261012_PRMT5_rnai_cohort_prediction_summary.csv`](example/prmt5_rnai_expr_prot/PRMT5/rnai/261012_PRMT5_rnai_cohort_prediction_summary.csv).

### What the PRMT5 examples show

Chromosome bands below are the HGNC locations in DepMap `Gene.csv`. MTAP, CDKN2A, and CDKN2B are all at 9p21.3. PRMT5 is at 14q11.2. Both runs use the 693 solid-cancer lines with a PRMT5 RNAi score. A more negative score is more sensitive.

The figure is the top PANCANCER result from the expression run: lines with low MTAP expression versus the other lines.

![PRMT5 RNAi in PANCANCER. Lines with low MTAP expression are more sensitive than the other lines.](example/prmt5_mtap_lowexp.png)

**MTAP low expression is a stronger marker than MTAP copy number.** With expression features on, `MTAP_lowexp` is the top PANCANCER marker: 105 lines, median score −0.76 versus −0.27, Wilcoxon p = 5.22e-40. `MTAP_homdel` is 141 lines, median −0.71 versus −0.27, p = 1.2e-36. Of the 105 low-expression lines, 102 are also homozygously deleted. The 39 deleted lines that still express MTAP have a median of −0.39, so the sensitive deletions are the ones that lose expression. Low MTAP protein in the Gygi table (`Q13126`, the MTAP UniProt accession) agrees: 25 lines, median −0.77, p = 1.84e-9. The Sanger low-protein call does not separate the scores.

In the genomic summary, the top row is the group `MTAP_homdel;MTAP_ko;MTAP_altered`. The p-value on that row (1.2e-36) and the count (141) are the homozygous-deletion feature. The median on that same row (−0.60) is the combined call, 163 lines with any of the three features.

**An MTAP call without loss of the neighboring CDKN2A or CDKN2B locus is not sensitive.** `MTAP_ko` is a homozygous deletion or a likely loss-of-function mutation. It matches `MTAP_homdel` in 141 of 143 lines. The other two lines are likely loss-of-function, are wild type for CDKN2A, and have PRMT5 scores of −0.11 and −0.22. Those two calls do not behave like MTAP loss. Inside the broader altered group, the AND tree adds three resistant branches: 12 lines with no CDKN2A homozygous deletion (median −0.23 versus −0.67, p = 9.76e-6), 31 lines with no CDKN2B deletion or alteration (median −0.26 versus −0.72, p = 3.75e-8), and 17 lines with a CDKN2A–MTAP fusion (median −0.17 versus −0.69, p = 8.54e-8).

**The other strong sensitive markers sit in the same 9p21 deletion.** PANCANCER roots in both runs are genes at 9p21.3 (CDKN2B, the interferon cluster, DMRTA1, KLHL9, FOCAD, ELAVL2, MLLT3) or the neighboring bands (TUSC1, CAAP1, and LINGO2 at 9p21.2; SLC24A2 at 9p22.1). Cohort trees that pass follow the same region: MTAP in NSCLC and LUAD, the interferon cluster or `MTAP_lowexp` in glioma, and ELAVL2 in breast.

**Markers outside 9p are weaker.** `CRIP2` low expression (14q32.33) is still associated in lines that are not MTAP-deleted: 15 lines, median −0.44 versus −0.26, p = 2.89e-6. `DDX56` alteration (7p13) shrinks once those deleted lines are removed: 14 lines, median difference −0.18, p = 2.66e-4. In colorectal models the only passing marker is `OR10A2` high expression at 11p15.4 (7 lines, median −0.51 versus −0.21, p = 1.01e-4).

Summaries and the plots behind this reading:

| Run | Summary | Example plot |
| --- | --- | --- |
| Genomic features | [summary CSV](example/prmt5_rnai/PRMT5/rnai/261012_PRMT5_rnai_cohort_prediction_summary.csv) | [PANCANCER AND PDF](example/prmt5_rnai/PRMT5/rnai/PANCANCER/2026-10-12_PRMT5_rnai_AND_PANCANCER_boxplots.pdf) |
| Expression and proteomics | [summary CSV](example/prmt5_rnai_expr_prot/PRMT5/rnai/261012_PRMT5_rnai_cohort_prediction_summary.csv) | [PANCANCER AND PDF](example/prmt5_rnai_expr_prot/PRMT5/rnai/PANCANCER/2026-10-12_PRMT5_rnai_AND_PANCANCER_boxplots.pdf) |
| Colorectal, expression on | same expression summary | [COAD AND PDF](example/prmt5_rnai_expr_prot/PRMT5/rnai/COAD/2026-10-12_PRMT5_rnai_AND_COAD_boxplots.pdf) |

## Docker

The image does not contain DepMap files. Mount a `data/` directory:

```bash
docker build -t kopredict .
docker run --rm --entrypoint kopredict-download -v "$PWD/data:/data" kopredict --depmap-release 26Q1
docker run --rm --entrypoint kopredict-prepare -v "$PWD/data:/data" kopredict
docker run --rm -v "$PWD/data:/data" kopredict
```

`KOPREDICT_DATA_DIR` inside the image is `/data`.

## Future updates

1. Add PRISM and GDSC as target datasets, alongside CRISPR and RNAi.
2. Accept a user-supplied knockout dataset as the target scores.
3. Score and prioritize the sensitive and resistant markers using gene function, pathway membership, TCGA, and other datasets as evidence that a hit is biological.
