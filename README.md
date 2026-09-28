# kopredict

kopredict finds genomic features linked to CRISPR or RNAi sensitivity. The search uses a Wilcoxon test, a hypergeometric overlap step, and a feature tree. `kopredict-download` and `kopredict-prepare` rebuild the local inputs from public files so a clone can run without a private database.

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

### 3. Build the matrices

```bash
kopredict-prepare
kopredict-prepare --depmap-release 25Q3
```

This first checks every file in `metadata/default_depmap_files_to_download_and_process.csv` and `metadata/additional_datasets_to_download.csv`. A missing file, a missing `md5_hash` for a catalog file, or a checksum that does not match stops the command and lists each problem. Use the same `--depmap-release` value you used for `kopredict-download`.

This writes:

| File | Role |
| --- | --- |
| `Model_and_OmicsSignatures.parquet.gz` | Model metadata, global signatures, HRD, and SBS exposures |
| `OmicsFull_processed.parquet.gz` | Long gene-symbol table used by `-g` background filters (copy number, mutations, expression, CRISPR, RNAi). Proteomics are not in this file. |
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

Features that pass the Wilcoxon cutoff are grouped by a hypergeometric overlap test. The first cutoff comes from the label shuffles in `n_permutations`. Each later cell-line subset repeats that same number of shuffles after features that are constant in the selected lines are removed and identical rows are merged. The tree expands siblings best first. At every row, the search under a sibling cannot see that sibling or any earlier sibling on the row, and features excluded by a parent stay excluded for the rest of that branch.

## Run a search

```bash
kopredict
kopredict path/to/run.yaml
kopredict --help
```

Search settings live in [`metadata/kopredict_default.yaml`](metadata/kopredict_default.yaml). That file stays short: it is the example for target `IPPK`, dataset `crispr`, and cohort `SOLID`. Copy it, edit `target`, `dataset`, and `cancer_types`, and pass the copy as the argument above. A copied file can omit keys; missing keys are filled from the default file.

Which cohort strings are valid is listed in [`metadata/oncotree.yaml`](metadata/oncotree.yaml). That file is from DepMap Public 26Q1 `Model.csv`. The `oncotree_column` list names the four columns the run file accepts: `oncotreelineage`, `oncotreeprimarydisease`, `oncotreesubtype`, and `oncotreecode`. Under each column, every value is paired with the number of cell lines that have that exact string.

To run one cohort, set `oncotree_column` and `cancer_types` in the run YAML. `cancer_types` can be one value or a list. The match is exact, and a list keeps a cell line when it matches any value. `SOLID` and `PANCANCER` do not use `oncotree_column`. `PANCANCER` is every model whose primary disease does not contain `NON-CANCER` (2,003 models). `SOLID` is that same set with the hematologic primary diseases listed in [`metadata/oncotree.yaml`](metadata/oncotree.yaml) removed (1,939 models).

To run several cohorts in one command, set `cancer_types` to null and set `cohort_csv` to a CSV path. The search reads only `oncotree_column` and `name`. Other columns are ignored. Each row is one analysis. A `name` may be one value or several values separated by semicolons. A semicolon list is combined the same way as a `cancer_types` list. Every `oncotree_column` and `name` pair must exist. If any name in a list does not, the error lists the missing pairs and the search does not run. For `SOLID` or `PANCANCER`, leave `oncotree_column` blank. Values that contain a comma must be quoted.

[`metadata/example_lung_cohort.csv`](metadata/example_lung_cohort.csv) is a cohort file for every cancer Lung cohort in DepMap Public 26Q1: the `Lung` lineage, plus each primary disease, subtype, and code recorded under that lineage. `depmap_26q1_cell_lines` is the number of `Model.csv` cell lines with that exact value, or the union when `name` contains a semicolon list. The search does not use that column. The last row combines four codes:

```csv
oncotree_column,name,depmap_26q1_cell_lines
oncotreelineage,Lung,261
oncotreeprimarydisease,Non-Small Cell Lung Cancer,165
oncotreesubtype,Lung Adenocarcinoma,91
oncotreecode,LUAD,91
oncotreecode,LUCA;GCLC;LCLC;LUMEC,23
```

`kopredict --help`, `kopredict-download --help`, and `kopredict-prepare --help` print a short how-to and the settings or flags each command accepts.

By default the search drops feature names containing `absolute`, `lowexp`, `highexp`, `depmapprot`, `sangerprot`, or `rppa`. Set `keep_expression_features` or `keep_proteiomic_features` to true in the YAML file to keep those families (`rppa` follows the protein flag).

Plots and the prediction summary are written under `data/output/{target}/{dataset}/` unless `output_dir` is set in the YAML file. That folder sits inside `data/`, which git ignores.

## Docker

The image does not contain DepMap files. Mount a `data/` directory:

```bash
docker build -t kopredict .
docker run --rm --entrypoint kopredict-download -v "$PWD/data:/data" kopredict --depmap-release 26Q1
docker run --rm --entrypoint kopredict-prepare -v "$PWD/data:/data" kopredict
docker run --rm -v "$PWD/data:/data" kopredict
```

`KOPREDICT_DATA_DIR` inside the image is `/data`.
