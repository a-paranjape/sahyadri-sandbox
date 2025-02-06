#!/bin/bash
set -euo pipefail

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Conda is installed
if ! command_exists conda; then
    echo "Conda is not installed. Please install Conda and try again."
    exit 1
fi

# Create and activate the environment
#This is needed only for postprocessing
ENV_NAME="sahyadri_pp"
#PYTHON_VERSION="3.8.8"
PYTHON_VERSION="3.12.0"

if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Creating conda environment $ENV_NAME..."
    conda create -n "$ENV_NAME" python=="$PYTHON_VERSION" -y -c conda-forge
else
    echo "Conda environment $ENV_NAME already exists."
fi

echo "Activating conda environment $ENV_NAME..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Define the packages to install
packages=(
    "boost"
    "cython"
    "fitsio"
    "gsl"
    "numpy"
    "matplotlib"
    "joblib"
    "pandas"
    "h5py"
    "scipy"
    "pip"
    "git"
    "ipython"
    "jupyter"
    "ipykernel"
)

# Install all packages at once
echo "Installing packages from conda-forge..."
if ! conda install -y -c conda-forge "${packages[@]}"; then
    echo "Error: Failed to install one or more packages. Please check the package names and try again."
    echo "You can also try installing packages individually to identify which one is causing the issue."
    exit 1
fi

# Install the IPython kernel
python -m ipykernel install --user --name="$ENV_NAME" --display-name="$ENV_NAME"


# Clone the repository
REPO_URL="https://gitlab.com/shadaba/CorrelationFunction.git"
REPO_DIR="CorrelationFunction"

if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository..."
    git clone "$REPO_URL"
else
    echo "Repository already exists. Updating..."
    cd "$REPO_DIR"
    git pull
    cd ..
fi

# Build the extension
cd "$REPO_DIR"
python setup.py clean
rm -f calc.c
python setup.py build_ext --inplace

echo "Postbuild script completed successfully."
