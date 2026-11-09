# Fixed 500-individual analysis inputs

Each dataset contains the 500 individuals used in the formal analysis, together with their marker annotations, fixed CV memberships and fold-fitted PCA assets.

## Common carp

The common carp subset was derived from the supplementary data of Palaiokostas et al. (2019), *Optimizing Genomic Prediction of Host Resistance to Koi Herpesvirus Disease in Carp*, Frontiers in Genetics 10, 543.

Publication and supplementary files

https://doi.org/10.3389/fgene.2019.00543

The publication is distributed under the Creative Commons Attribution license. Copyright remains with Palaiokostas, Vesely, Kocour, Prchal, Pokorova, Piackova, Pojezdal and Houston. The original source files can be obtained from the publication's supplementary material. SL is the configured prediction target.

## Rainbow trout

The rainbow trout subset was derived from the data of Barría et al. (2019), *Single-Step Genome-Wide Association Study for Resistance to Piscirickettsia salmonis in Rainbow Trout*, G3 9, 3833–3841.

Publication

https://doi.org/10.1534/g3.119.400204

Source dataset

https://doi.org/10.6084/m9.figshare.8146076

The source repository specifies CC BY 4.0 and provides the original genotype, map and phenotype files. Copyright remains with the original contributors. End_Weight is the configured prediction target.

## Derived inputs

For each public dataset, `processed/all_500_allele10.h5` is the portable materialization of the exact selected 500-individual analysis input. The genotype array has axes SNP × individual × channel. Channels are AA, AT, AC, AG, TT, TC, TG, CC, CG and GG. Missing genotypes are encoded as all-zero vectors.

`cv_splits_*.csv` preserves the accepted fold memberships. `selection/` records the selected individual IDs. `pca/seed_<seed>/fold_<fold>/` contains the original fitted preprocessing, PCA scores, retained-marker list and binding metadata. Derived material is supplied with the original-data attribution above.

The black carp configuration uses private data supplied by its owner. Its input contract is described in `../docs/private_data.md`.
