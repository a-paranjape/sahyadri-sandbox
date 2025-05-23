#!/usr/bin/env python
"""
Setup script for sahyadri-sandbox
"""

from setuptools import setup, find_packages
import os
import sys
import subprocess

# Read README for long description
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "A comprehensive toolkit for N-body simulation post-processing and analysis"

# Get version from git or fallback
def get_version():
    try:
        # Try to get version from git tag
        version = subprocess.check_output(
            ['git', 'describe', '--tags', '--always'], 
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        # If we have a proper tag (starts with v), clean it up
        if version.startswith('v'):
            version = version[1:]  # Remove 'v' prefix
        
        return version
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback version if git is not available or no tags
        return "0.1.0-dev"

# Define core dependencies that are available via pip
install_requires = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "matplotlib>=3.3.0",
    "pandas>=1.3.0",
    "h5py>=3.1.0",
    "joblib>=1.0.0",
    "ipython>=7.0.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
]

# Dependencies that need conda or special handling
conda_only_deps = [
    "fitsio",  # astronomy-specific, better from conda-forge
    "gsl",     # GNU Scientific Library
    "boost",   # C++ libraries
    "cython",  # for compilation
]

# Development dependencies
dev_requires = [
    "pytest>=6.0.0",
    "pytest-cov>=2.10.0",
    "black>=21.0.0",
    "flake8>=3.8.0",
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=0.5.0",
]

setup(
    name="sahyadri-sandbox",
    version=get_version(),
    author="Aseem, Saee, Shadab",
    author_email="shadab.alam@tifr.res.in",
    description="A comprehensive toolkit for N-body simulation post-processing and analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/a-paranjape/sahyadri-sandbox",
    project_urls={
        "Bug Tracker": "https://github.com/a-paranjape/sahyadri-sandbox/issues",
        "Documentation": "https://github.com/a-paranjape/sahyadri-sandbox/docs",
        "Source Code": "https://github.com/a-paranjape/sahyadri-sandbox",
    },
    packages=find_packages(where=".", exclude=["tests*", "docs*"]),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "docs": ["sphinx>=4.0.0", "sphinx-rtd-theme>=0.5.0"],
        "test": ["pytest>=6.0.0", "pytest-cov>=2.10.0"],
    },
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "sahyadri-postprocess=scripts.post_process.postprocess:main",
            "sahyadri-compress=scripts.post_process.compress_id_test:main",
            "sahyadri-correlations=scripts.post_process.correlations:main",
        ],
    },
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.cfg", "*.ini"],
        "scripts": ["*.sh", "*.pbs"],
        "config": ["*.yaml", "*.yml", "*.json"],
        "docs": ["*.rst", "*.md"],
    },
    
    # Data files
    data_files=[
        ("scripts", ["postBuild_modern.sh"]),
    ],
    
    # Zip safe
    zip_safe=False,
    
    # Keywords
    keywords=[
        "astronomy", "astrophysics", "cosmology", "n-body", "simulation", 
        "post-processing", "data-analysis", "compression", "visualization"
    ],
)

# Post-installation message
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SAHYADRI-SANDBOX INSTALLATION")
    print("="*60)
    print(f"Version: {get_version()}")
    print("\nIMPORTANT: Some dependencies require conda for optimal performance:")
    print("Conda-only dependencies:")
    for dep in conda_only_deps:
        print(f"  - {dep}")
    
    print("\nTo install all dependencies optimally, run:")
    print("  conda install -c conda-forge fitsio gsl boost cython")
    print("\nOr use the provided environment setup:")
    print("  bash postBuild_modern.sh")
    print("\nFor development installation:")
    print("  pip install -e .[dev]")
    print("="*60)
