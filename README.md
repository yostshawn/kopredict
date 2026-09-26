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

`metadata/downloads.csv` is the DepMap file list, including signed download URLs. Those URLs expire. Download a fresh copy from [https://depmap.org/portal/api/download/files](https://depmap.org/portal/api/download/files) in a browser and replace `metadata/downloads.csv` with that file before running the download, and again whenever a link starts failing. Three older files in that catalog have no `md5_hash` from DepMap (`D2_combined_gene_dep_scores.csv`, `sanger-dose-response.csv`, and `sanger-viability.csv`). Those three checksums were computed locally and written into `metadata/downloads.csv`. If you replace the catalog with a newer download, add `md5_hash` values for any row that is blank, or prepare will stop on those files.

`metadata/depmap_files.csv` lists which rows of that catalog to download. A release value of `{depmap}` follows `--depmap-release`. Other rows, such as DEMETER2, GDSC, and proteomics, keep the release name written in the CSV. Add or remove files there rather than in the Python code.

`metadata/additional.csv` lists workbooks that are not in the DepMap catalog. Each row has a local filename, a citation, a file URL, and whether `kopredict-download` should fetch it. The HRD table is downloaded from its URL in that CSV. Somatic signature activities come from Petljak et al., Cell 2019, supplementary file [mmc3.xlsx](https://www.cell.com/cms/10.1016/j.cell.2019.02.012/attachment/4e70a2ab-6758-47e5-a843-802a5474bc82/mmc3.xlsx) ([paper](https://www.cell.com/cell/fulltext/S0092-8674(19)30161-8)). Save that file as `data/raw/mmc3.xlsx`. Prepare reads the sheet `COSMIC CellLines Signatures`. Cell.com often blocks a scripted download, so if the fetch fails, save that same URL by hand.

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

This first checks every file in `metadata/depmap_files.csv` and `metadata/additional.csv`. A missing file, a missing `md5_hash` for a catalog file, or a checksum that does not match stops the command and lists each problem. Use the same `--depmap-release` value you used for `kopredict-download`.

This writes:

| File | Role |
| --- | --- |
| `Model_and_OmicsSignatures.parquet.gz` | Model metadata, global signatures, HRD, and SBS exposures |
| `OmicsFull_processed.parquet.gz` | Long gene-symbol table used by `-g` background filters (copy number, mutations, expression, CRISPR, RNAi). Proteomics are not in this file. |
| `Proteomics_processed.parquet.gz` | Long UniProt table of Sanger, Gygi, and RPPA z-scores. The `gene` column holds accessions. |
| `CRISPRGeneEffect_processed.parquet.gz` | CRISPR gene effect |
| `D2_combined_gene_dep_scores_processed.parquet.gz` | DEMETER2 RNAi scores |
| `final_depmap_features_26Q1.parquet.gz` | Binary features (rows) by cell line (columns) |

Non-cancer models (`OncotreePrimaryDisease` of `Non-Cancerous`) are omitted from the feature matrix. Gene features use only `Gene.csv` rows whose `locus_group` is `protein-coding gene`. Pseudogenes, non-coding RNAs, and other locus groups are dropped from copy number, mutations, expression, CRISPR, RNAi, and fusions. Proteomics rows stay as UniProt accessions and are not converted to gene symbols. Copy-number amplification is a copy-number value of at least 2 (`CN_AMP_CUTOFF`), and homozygous deletion is a copy-number value below 0.1 (`CN_HOMDEL_CUTOFF`). Expression, Sanger MS, Gygi MS, and RPPA use a per-gene z-score cutoff of 1.64. Likely loss-of-function uses the variant classes Nonsense_Mutation, Frame_Shift_Del, Frame_Shift_Ins, Splice_Site, and Nonstop_Mutation.

These old DataHub feature families are not rebuilt, because the pinned files do not contain them: `has_hessdriver`, `has_oncogenehighimpact`, `has_tumorsuppressorhighimpact`.

## Run a search

```bash
kopredict -t IPPK -d crispr -c SOLID -O oncotreeprimarydisease
```

`-d` is `crispr` or `rnai`. `-p AND|OR` chooses how feature blocks narrow the cohort (default `AND`). `-O` is the model column matched by `-c` (default `oncotreeprimarydisease`). `SOLID` and `PANCANCER` are special cohort names. `-f` defaults to `data/final_depmap_features_26Q1.parquet.gz`.

By default the search drops feature names containing `absolute`, `sensitive`, `resistant`, `lowexp`, `highexp`, `depmapprot`, `sangerprot`, or `rppa`. `--keep-lowexp-highexp` and `--keep-depmapprot-sangerprot` keep those families (`rppa` is kept with the protein flag).

Plots and the prediction summary are written under `data/output/{target}/{dataset}/` unless you pass `--output-dir`. That folder sits inside `data/`, which git ignores.

## Docker

The image does not contain DepMap files. Mount a `data/` directory:

```bash
docker build -t kopredict .
docker run --rm --entrypoint kopredict-download -v "$PWD/data:/data" kopredict --depmap-release 26Q1
docker run --rm --entrypoint kopredict-prepare -v "$PWD/data:/data" kopredict
docker run --rm -v "$PWD/data:/data" -v "$PWD/results:/results" kopredict -t IPPK -d crispr -c SOLID -O oncotreeprimarydisease --output-dir /results
```

`KOPREDICT_DATA_DIR` inside the image is `/data`.
