# METAN

Configuration-driven genomic prediction of fish growth traits using multi-scale SNP embedding, contextual attention, expert-choice routing, coverage-aware aggregation and PCA-assisted feature fusion.

## Repository contents

- `metan/` contains the shared model, training, inference, validation and baseline entry points.
- `configs/` contains the three final model configurations and their optimization and evaluation settings.
- `data/carp/` and `data/trout/` contain the fixed 500-individual inputs, marker annotations, CV indices and fold-fitted PCA assets.
- `baselines/` contains the GBLUP implementation and reference scripts. SVR and LightGBM are implemented in `metan/baselines.py`.
- `scripts/preprocessing/` contains source-format and private-data preparation references.
- `tests/` contains numerical and configuration contract checks.

Black carp data are private. Their owner supplies the genotype, phenotype, CV and PCA files described in `docs/private_data.md` before running that configuration.

## Final configurations

| Configuration | Target traits | Input SNPs | PCA components | Initial LR | Weight decay | Output dropout | Parameters |
|---|---|---:|---:|---:|---:|---:|---:|
| `configs/blackcarp/experiment.json` | BW, LE | 28,804 | 20 | 8e-5 | 0.03 | 0.2 | 422,486 |
| `configs/carp/experiment.json` | SL | 15,615 | 8 | 5e-5 | 0.1 | 0.6 | 506,179 |
| `configs/trout/experiment.json` | End_Weight | 26,068 | 40 | 3.5e-3 | 0.1 | 0.6 | 545,411 |

Each configuration defines its trait order, input layout, dimensions, PCA assets, optimizer, scheduler, early-stopping rule and evaluation protocol. The same commands are used for every configuration. BW and LE share one joint training trajectory, with a separate best checkpoint selected for each trait.

## Environment

The reference environment uses Linux, Python 3.10 and an NVIDIA CUDA GPU with BF16 support. Install PyTorch and the core dependencies first, then compile the attention extension against that environment.

```bash
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install lightgbm==4.7.0
```

GBLUP requires R, a valid ASReml-R license, ASRgenomics, data.table and jsonlite. The optional preprocessing dependencies are listed in `requirements-optional.txt`.

## Verify the supplied inputs

Run these commands from the repository root.

```bash
python -B -m unittest discover -s tests -v
python -B -m metan check --config configs/carp/experiment.json
python -B -m metan check --config configs/trout/experiment.json
```

The input check verifies original CV membership, SNP axes, sample IDs, PCA training/validation binding and file checksums. The supplied PCA transformations were fitted on their corresponding training folds.

## Reproduce an experiment

The complete workflow runs the configured 15 training folds, evaluates the selected checkpoints, fits all three baselines and generates original-unit predictions and metric summaries.

```bash
python -B -m metan reproduce --config configs/carp/experiment.json --out runs/carp
python -B -m metan reproduce --config configs/trout/experiment.json --out runs/trout
```

After the private inputs have been supplied, the corresponding command is

```bash
python -B -m metan reproduce --config configs/blackcarp/experiment.json --out runs/blackcarp
```

For a detached server run, select the desired configuration and a fresh output directory.

```bash
mkdir -p run_logs
nohup python -u -B -m metan reproduce --config configs/carp/experiment.json --out runs/carp > run_logs/carp.log 2>&1 &
```

Progress is recorded in `runs/carp/status.json` and `runs/carp/METAN/status.json`. Each task has its own log, predictions, checkpoint and status record. A failed task stops the queue and retains its error and exit status. New runs require fresh output directories.

## Individual workflow stages

The bundled 500-individual HDF5 files are ready for the training and baseline commands above. Optional source-data rebuilding requires the original genotype and phenotype files obtained from the publications linked in `data/README.md`, placed in the `raw_input.root` directory specified by the configuration. The rebuilding command retains the fixed individual list and copies the fixed CV and fitted PCA assets into a fresh preparation directory.

```bash
python -B -m metan rebuild --config configs/carp/experiment.json --out prepared/carp
python -B -m metan rebuild --config configs/trout/experiment.json --out prepared/trout
```

```bash
python -B -m metan cv --config configs/carp/experiment.json --out runs/carp_METAN
python -B -m metan prepare --config configs/carp/experiment.json --out runs/carp_baseline_inputs
python -B -m metan baselines --config configs/carp/experiment.json --genotype runs/carp_baseline_inputs/dosage.csv --phenotype runs/carp_baseline_inputs/phenotypes.csv --out runs/carp_baselines
```

`python -m metan <command> --help` describes each entry point. `reproduce`, `train`, `cv` and `baselines` also accept `--describe` to print their planned settings.

## Output interpretation

The workflow reports Pearson correlation, MSE, RMSE, MAE, R² and the prediction-to-observation SD ratio. Metrics are evaluated after inverse phenotype standardization. Summaries use the arithmetic mean and sample SD across evaluations. RMSE is calculated within each evaluation before averaging.

Carp and trout use repeated five-fold CV on their fixed 500-individual subsets. The black carp configuration evaluates each trait-selected fold model on the same 75-individual historical holdout cohort. Its upstream MIC candidate set was selected using the 424-individual development pool. The evaluation scopes are kept separate in generated outputs.

The package has been checked for parameter-count identity, identical initialized tensors, strict loading of representative accepted weights and identical BF16 forward outputs after renaming. Full historical training has not been rerun for this packaging operation. GPU training may retain hardware-dependent numerical variation.

## Data sources and licensing

The 500-individual data subsets retain their original source attributions in `data/README.md`. Code is distributed under the Apache License 2.0 provided in `LICENSE`.
