# Whisper of DNA

This project is a deep learning framework for genomic prediction, with a model architecture inspired by the Whisper model. It provides a complete pipeline, including data preprocessing, model training (pre-training and fine-tuning), and visualization of results and model interpretability (attention weights).

## Framework Overview

### Model Architecture

The Whisper of DNA framework employs a hierarchical transformer architecture designed specifically for genomic sequence analysis:

<div align="center">
  <img src="assets/Architecture.png" alt="Whisper of DNA Architecture" width="800"/>
  <p><em>Figure 1: Overall architecture of the Whisper of DNA framework. The model processes genomic sequences through embedding layers, multi-scale transformer blocks, and specialized attention mechanisms for genomic prediction tasks.</em></p>
</div>

### Attention Weight Aggregation

The framework provides sophisticated attention weight analysis to identify important genomic regions:

<div align="center">
  <img src="assets/Attention Aggregation.png" alt="Attention Weight Aggregation" width="800"/>
  <p><em>Figure 2: Attention weight aggregation mechanism. The system aggregates attention weights across multiple transformer layers and heads to compute SNP importance scores, enabling interpretation of genomic contributions to phenotypic predictions.</em></p>
</div>

## Environment Setup

This project uses **Conda** for environment management and is tested with **Python 3.10.15**.

### Prerequisites

- Anaconda or Miniconda
- PLINK 1.9 (for SNP quality control). Ensure it is installed and accessible in your system's PATH.

### Quick Environment Setup

We provide two separate environments for convenience:

1. **Training** (`mlenv.yml`): For model training with PyTorch, GPU support, and R packages
2. **Visualization** (`notebooks/plotenv.yaml`): For Jupyter notebooks and visualization

#### One-Command Setup

```bash
cd /path/to/Whisper_of_DNA_pl
bash setup_environments.sh
```

#### Manual Setup

```bash
# Training environment
conda env create -f mlenv.yml
conda activate mlenv

# Visualization environment  
conda env create -f notebooks/plotenv.yaml
conda activate plotenv
```

### Viewing Results with HTML Reports

For reviewers: view all notebook figures in one click without running code:

```bash
# Setup visualization environment
conda activate plotenv

# Generate and view HTML reports (with all embedded figures)
python generate_report.py
```

This will:
1. Convert all notebooks in `notebooks/` to HTML
2. Embed all figures directly in HTML files  
3. Automatically open in your browser

You can also view individual notebooks:
```bash
python generate_report.py notebooks/drawattentionweights.ipynb
```

Generated reports are saved in `reports/` directory.

### Install PLINK (if not already installed)

```bash
# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install plink1.9

# On CentOS/RHEL
sudo yum install plink

# Verify PLINK installation
plink --version
```

## Framework Overview

### Training Data Types

This framework supports two types of training data:

1. **Binary Format Training** (`training/train.py`):
   - For genomic data that has been preprocessed into HDF5 format
   - Requires data preprocessing pipeline
   - Supports SNP quality control, filtering, and genome partitioning
   - Recommended for large-scale genomic datasets

2. **CSV Format Training** (`training/traincsvdata.py`):
   - For direct training on CSV format data (e.g., wheat 599, wheat 2000 datasets)
   - No preprocessing required
   - Suitable for small to medium-sized datasets in tabular format

### Pipeline Components

1. **Data Preprocessing** (For Binary Format Training):
   - Set preprocessing parameters in: [`config/preprocessing_config.json`](config/preprocessing_config.json)
   - Run preprocessing pipeline: [`scripts/preprocess_pipeline.py`](scripts/preprocess_pipeline.py)
   - Converts raw genomic data into HDF5 format suitable for model training

2. **Model Training**:
   - **Pre-training**:
     - Configure model architecture in [`training/config/model_config_pretrain.json`](training/config/model_config_pretrain.json)
     - Configure training parameters in [`training/config/training_config_pretrain.yml`](training/config/training_config_pretrain.yml)
     - Run pre-training with [`training/train.py`](training/train.py) or [`training/traincsvdata.py`](training/traincsvdata.py)
   - **Fine-tuning**:
     - Configure model architecture in [`training/config/model_config.json`](training/config/model_config.json)
     - Configure training parameters in [`training/config/training_config.yml`](training/config/training_config.yml)
     - Run fine-tuning with [`training/train.py`](training/train.py) or [`training/traincsvdata.py`](training/traincsvdata.py)

3. **Evaluation & Visualization**:
   - **Model Interpretability**: Use [`notebooks/drawattentionweights.ipynb`](notebooks/drawattentionweights.ipynb) to analyze attention weights
   - **Prediction Results**: Use [`notebooks/drawpreandtrue.ipynb`](notebooks/drawpreandtrue.ipynb) to visualize predictions vs. true values

## Quick Start

### Option A: Binary Format Training (Recommended for Large Datasets)

#### Step 1: Data Preprocessing

1. Edit the preprocessing configuration:
   ```bash
   # Edit preprocessing configuration
   nano config/preprocessing_config.json
   ```

2. Run the preprocessing pipeline:
   ```bash
   # Basic preprocessing (without quality control)
   python scripts/preprocess_pipeline.py
   
   # With quality control enabled
   python scripts/preprocess_pipeline.py --enable-qc
   
   # With both quality control and MIC analysis
   python scripts/preprocess_pipeline.py --enable-qc --enable-mic
   
   # Using custom config directory
   python scripts/preprocess_pipeline.py --config /path/to/config --enable-qc
   ```

#### Step 2: Pre-training (Optional but Recommended)

1. Configure pre-training settings:
   ```bash
   # Edit model architecture for pre-training
   nano training/config/model_config_pretrain.json
   
   # Edit training parameters for pre-training
   nano training/config/training_config_pretrain.yml
   ```

2. Start pre-training:
   ```bash
   # Run pre-training
   python training/train.py \
     --model-config training/config/model_config_pretrain.json \
     --training-config training/config/training_config_pretrain.yml
   ```

#### Step 3: Fine-tuning

1. Configure fine-tuning settings:
   ```bash
   # Edit model architecture for fine-tuning
   nano training/config/model_config.json
   
   # Edit training parameters (ensure to set resume_from_checkpoint path)
   nano training/config/training_config.yml
   ```

2. Start fine-tuning:
   ```bash
   # Run fine-tuning with pre-trained checkpoint
   python training/train.py \
     --model-config training/config/model_config.json \
     --training-config training/config/training_config.yml \
     --checkpoint /path/to/pretrained/checkpoint.ckpt
   
   # Or fine-tuning from scratch
   python training/train.py \
     --model-config training/config/model_config.json \
     --training-config training/config/training_config.yml
   ```

### Option B: CSV Format Training (For Tabular Data)

#### Step 1: Prepare CSV Data

Ensure your CSV data follows the expected format with features and target columns.

#### Step 2: Configure CSV Training

1. Configure model and training settings:
   ```bash
   # Edit model configuration for CSV data
   nano training/config/model_config.json
   
   # Edit training configuration for CSV data
   nano training/config/training_config.yml
   ```

#### Step 3: Run CSV Training

```bash
# Pre-training with CSV data
python training/traincsvdata.py \
  --model-config training/config/model_config_pretrain.json \
  --training-config training/config/training_config_pretrain.yml

# Fine-tuning with CSV data
python training/traincsvdata.py \
  --model-config training/config/model_config.json \
  --training-config training/config/training_config.yml \
  --checkpoint /path/to/pretrained/checkpoint.ckpt

# Training from scratch with CSV data
python training/traincsvdata.py \
  --model-config training/config/model_config.json \
  --training-config training/config/training_config.yml
```

### Step 4: Visualization and Analysis

#### Analyze Attention Weights

```bash
# Open Jupyter notebook for attention analysis
jupyter notebook notebooks/drawattentionweights.ipynb

# Or use Jupyter Lab
jupyter lab notebooks/drawattentionweights.ipynb
```

In the notebook:
- Modify path variables to match your training output paths
- Run cells sequentially to generate SNP importance visualizations

#### View Prediction Results

```bash
# Open Jupyter notebook for prediction visualization
jupyter notebook notebooks/drawpreandtrue.ipynb

# Or use Jupyter Lab
jupyter lab notebooks/drawpreandtrue.ipynb
```

In the notebook:
- Update log directory path to match your training output
- Run the notebook to generate prediction vs. true value plots

## Training Examples

### Example 1: Complete Binary Format Pipeline

```bash
# 1. Preprocess genomic data with quality control
python scripts/preprocess_pipeline.py --enable-qc --enable-mic

# 2. Pre-train the model
python training/train.py \
  --model-config training/config/model_config_pretrain.json \
  --training-config training/config/training_config_pretrain.yml

# 3. Fine-tune with the pre-trained model
python training/train.py \
  --model-config training/config/model_config.json \
  --training-config training/config/training_config.yml \
  --checkpoint logs/DNAWhisper/pretrain_run/checkpoints/epoch-99-val_loss-0.1234.ckpt

# 4. Analyze results
jupyter notebook notebooks/drawattentionweights.ipynb
jupyter notebook notebooks/drawpreandtrue.ipynb
```

### Example 2: CSV Format Training (e.g., Wheat Datasets)

```bash
# 1. Direct training on CSV data (wheat 599 or wheat 2000)
python training/traincsvdata.py \
  --model-config training/config/model_config.json \
  --training-config training/config/training_config.yml

# 2. Analyze results
jupyter notebook notebooks/drawpreandtrue.ipynb
```

### Example 3: Cross-Validation Training

```bash
# Enable K-fold cross-validation in training config
# Set use_cv_folds: true and cv_n_splits: 5 in training_config.yml

# Run cross-validation training
python training/train.py \
  --model-config training/config/model_config.json \
  --training-config training/config/training_config.yml
```

## Configuration Tips

### For Binary Format Training:
- Ensure `data.data_path` in `training_config.yml` points to the HDF5 file generated by preprocessing
- Set appropriate `batch_size` based on your GPU memory
- Configure `phenotype_name` in `model_config.json` to match your target variables

### For CSV Format Training:
- Ensure your CSV file path is correctly set in the training configuration
- Verify feature and target column names match the model expectations
- Adjust model input dimensions based on your CSV data structure

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce `batch_size` in training config
2. **File not found**: Check all file paths in configuration files
3. **Module import errors**: Ensure conda environment is activated
4. **PLINK errors**: Verify PLINK installation and PATH configuration

### Log Files:
- Training logs: `logs/DNAWhisper/[experiment_name]/`
- TensorBoard logs: `tensorboard --logdir logs/DNAWhisper/`
- Model checkpoints: `checkpoints/[experiment_name]/`

## Advanced Usage

### Custom Data Preprocessing:
```bash
# Use custom configuration directory
python scripts/preprocess_pipeline.py --config /custom/config/path --enable-qc
```

### Resume Training:
```bash
# Resume from specific checkpoint
python training/train.py \
  --model-config training/config/model_config.json \
  --training-config training/config/training_config.yml \
  --checkpoint /path/to/checkpoint.ckpt
```

### Distributed Training:
```bash
# Multi-GPU training (configure devices in training_config.yml)
python training/train.py \
  --model-config training/config/model_config.json \
  --training-config training/config/training_config.yml
```