#!/bin/bash
set -euo pipefail

# Sahyadri Sandbox Environment Setup Script
# This script sets up a complete conda environment for sahyadri-sandbox

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print colored output
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Check if Conda is installed
if ! command_exists conda; then
    print_error "Conda is not installed. Please install Conda and try again."
    echo "You can download it from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Configuration
ENV_NAME="sahyadri_pp"
PYTHON_VERSION="3.12.0"

print_info "Setting up Sahyadri Sandbox environment..."
print_info "Environment name: $ENV_NAME"
print_info "Python version: $PYTHON_VERSION"

# Create and activate the environment
if ! conda env list | grep -q "$ENV_NAME"; then
    print_info "Creating conda environment $ENV_NAME..."
    conda create -n "$ENV_NAME" python=="$PYTHON_VERSION" -y -c conda-forge
    print_success "Environment $ENV_NAME created successfully!"
else
    print_warning "Conda environment $ENV_NAME already exists."
    read -p "Do you want to update the existing environment? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Updating existing environment..."
    else
        print_info "Using existing environment..."
    fi
fi

print_info "Activating conda environment $ENV_NAME..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Define the packages to install via conda (for optimal performance)
conda_packages=(
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
    "pytest"
    "black"
    "flake8"
)

# Install conda packages
print_info "Installing packages from conda-forge..."
if ! conda install -y -c conda-forge "${conda_packages[@]}"; then
    print_error "Failed to install one or more conda packages."
    print_info "You can try installing packages individually to identify issues."
    exit 1
fi

print_success "Conda packages installed successfully!"

# Install the sahyadri-sandbox package in development mode
print_info "Installing sahyadri-sandbox in development mode..."
if [ -f "setup.py" ]; then
    pip install -e .[dev]
    print_success "Sahyadri-sandbox installed in development mode!"
else
    print_warning "setup.py not found. Installing dependencies only."
fi

# Install the IPython kernel
print_info "Installing IPython kernel..."
python -m ipykernel install --user --name="$ENV_NAME" --display-name="Sahyadri Sandbox ($ENV_NAME)"
print_success "IPython kernel installed!"

# Verify installation
print_info "Verifying installation..."
python -c "
import sys
print(f'Python version: {sys.version}')

# Test key imports
try:
    import numpy as np
    print(f'NumPy: {np.__version__}')
except ImportError as e:
    print(f'NumPy import failed: {e}')

try:
    import matplotlib
    print(f'Matplotlib: {matplotlib.__version__}')
except ImportError as e:
    print(f'Matplotlib import failed: {e}')

try:
    import h5py
    print(f'h5py: {h5py.__version__}')
except ImportError as e:
    print(f'h5py import failed: {e}')

try:
    import fitsio
    print(f'fitsio: {fitsio.__version__}')
except ImportError as e:
    print(f'fitsio import failed: {e}')

# Try to import sahyadri-sandbox if available
try:
    import sahyadri_sandbox
    print(f'Sahyadri Sandbox: {sahyadri_sandbox.__version__}')
    print(f'Git info: {sahyadri_sandbox.__git_info__[\"version_string\"]}')
except ImportError:
    print('Sahyadri Sandbox: Not installed as package (this is OK for script-only usage)')
"

print_success "Environment setup completed successfully!"

echo ""
echo "================================================================"
echo "SAHYADRI SANDBOX ENVIRONMENT READY"
echo "================================================================"
echo "Environment: $ENV_NAME"
echo "Python: $PYTHON_VERSION"
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To test the installation:"
echo "  python -c \"import sahyadri_sandbox; sahyadri_sandbox.version_info()\""
echo ""
echo "To run Jupyter with this environment:"
echo "  jupyter lab"
echo "  # Then select kernel: Sahyadri Sandbox ($ENV_NAME)"
echo ""
echo "To deactivate:"
echo "  conda deactivate"
echo "================================================================"
