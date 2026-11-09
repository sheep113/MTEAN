# Private black carp input contract

The black carp dataset remains with the research group. The release includes its final configuration and executable workflow.

The data custodian supplies the following paths or changes them in `configs/blackcarp/experiment.json`.

```text
data/blackcarp/
  processed/blackcarp_allele10_trainval.h5
  processed/blackcarp_allele10_test.h5
  cv_splits_1.csv
  cv_splits_2.csv
  cv_splits_3.csv
  geno_fixed_for_python.csv
  phenotypes.csv
  test_samples.txt
  pca/seed_<seed>/fold_<fold>/pca_scores.csv
  pca/seed_<seed>/fold_<fold>/metadata.json
  pca/seed_<seed>/fold_<fold>/fitted_preprocessing.joblib
  pca_holdout/seed_<seed>/fold_<fold>/pca_scores.csv
```

The development and holdout HDF5 files contain 424 and 75 individuals, respectively, with the original 28,804-SNP order and 10-channel encoding. Required datasets are `sample_ids`, `snp_ids`, `phenotype_names`, `phenotypes` and `features/genotype_features`. The genotype axes are SNP × individual × channel. Target columns are BW and LE.

Each CV CSV has `sample_id`, `fold` and `split` columns, with `train` and `val` split values. The supplied fold-specific PCA20 assets must retain the original fitting membership and CV checksum. Holdout PCA scores are produced with the corresponding training-fold transformation.

The baseline genotype CSV contains `sample_id` followed by the original 0/1/2 marker dosages. The phenotype CSV contains `sample_id`, BW and LE in their original numeric precision and units. `test_samples.txt` lists the original holdout IDs in evaluation order.

The preprocessing reference scripts document the original import, candidate-marker selection and PCA procedures. The complete historical data and fitted assets are required to reproduce the historical evaluation identity.
