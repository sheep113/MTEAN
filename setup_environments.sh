#!/bin/bash

# Simple setup script for Whisper of DNA environments
# Just create environments from YAML files

cd "$(dirname "$0")"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        CONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
        CONDA_URL="https://repo.anaconda.com/miniconda/$CONDA_INSTALLER"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if [[ $(uname -m) == "arm64" ]]; then
            # Apple Silicon
            CONDA_INSTALLER="Miniconda3-latest-MacOSX-arm64.sh"
        else
            # Intel
            CONDA_INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
        fi
        CONDA_URL="https://repo.anaconda.com/miniconda/$CONDA_INSTALLER"
    else
        echo "Error: Unsupported OS. Please install Miniconda manually:"
        echo "https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html"
        exit 1
    fi
    
    # Download and install
    echo "Downloading: $CONDA_URL"
    wget -q "$CONDA_URL" -O miniconda_installer.sh
    bash miniconda_installer.sh -b -p "$HOME/miniconda3"
    rm miniconda_installer.sh
    
    # Initialize conda
    "$HOME/miniconda3/bin/conda" init
    
    echo "Miniconda installed. Please run: source ~/.bashrc"
    echo "Then run this script again."
    exit 0
fi

echo "Conda found. Creating environments..."
echo ""

echo "Creating mlenv (Training environment)..."
conda env create -f mlenv.yml -y

echo ""
echo "Creating plotenv (Visualization environment)..."
conda env create -f notebooks/plotenv.yaml -y

echo ""
echo "Done! Use:"
echo "  conda activate mlenv    # for training"
echo "  conda activate plotenv  # for visualization"
