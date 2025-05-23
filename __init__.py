"""
Sahyadri Sandbox: A comprehensive toolkit for N-body simulation post-processing and analysis

This package provides tools for:
- N-body simulation data compression and decompression
- Post-processing of cosmological simulation outputs
- Correlation function analysis
- Visualization and plotting utilities
- Data format conversion (e.g., to FITS)

Modules:
--------
- scripts.post_process: Core post-processing functionality
- code: Main simulation and analysis code
- config: Configuration files and utilities
- docs: Documentation and examples
"""

import subprocess
import os

def get_version():
    """Get version from git or fallback to default"""
    try:
        if os.path.exists('.git'):
            version = subprocess.check_output(
                ['git', 'describe', '--tags', '--always'], 
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            
            if version.startswith('v'):
                version = version[1:]
            return version
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return "0.1.0-dev"

def get_git_info():
    """Get detailed git information for version tracking"""
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short=8', 'HEAD'], 
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        try:
            tag = subprocess.check_output(
                ['git', 'describe', '--exact-match', '--tags'], 
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
        except subprocess.CalledProcessError:
            tag = None
        
        # Check if working directory is clean
        try:
            subprocess.check_output(
                ['git', 'diff-index', '--quiet', 'HEAD'], 
                stderr=subprocess.DEVNULL
            )
            is_dirty = False
        except subprocess.CalledProcessError:
            is_dirty = True
        
        return {
            'hash': commit_hash,
            'branch': branch,
            'tag': tag,
            'is_dirty': is_dirty,
            'version_string': f"{commit_hash} ({branch})" + (" [Modified]" if is_dirty else "")
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            'hash': 'unknown',
            'branch': 'unknown',
            'tag': None,
            'is_dirty': False,
            'version_string': 'unknown'
        }

# Package metadata
__version__ = get_version()
__git_info__ = get_git_info()
__author__ = "Aseem, Saee, Shadab"
__email__ = "shadab.alam@tifr.res.in"
__license__ = "MIT"
__url__ = "https://github.com/a-paranjape/sahyadri-sandbox"

# Make key functionality easily accessible
try:
    # Import main modules if they exist
    from scripts.post_process import postprocess
    from scripts.post_process import utilities
    from scripts.post_process import readers
except ImportError:
    # Modules might not be available in all installations
    pass

def version_info():
    """Print detailed version information"""
    print(f"Sahyadri Sandbox v{__version__}")
    print(f"Git info: {__git_info__['version_string']}")
    print(f"Python package location: {os.path.dirname(__file__)}")
    print(f"Author: {__author__} <{__email__}>")
    print(f"URL: {__url__}")

def show_dependencies():
    """Show information about required dependencies"""
    conda_deps = ["fitsio", "gsl", "boost", "cython"]
    pip_deps = ["numpy", "scipy", "matplotlib", "pandas", "h5py", "joblib"]
    
    print("Required dependencies:")
    print("Via conda (recommended):", ", ".join(conda_deps))
    print("Via pip:", ", ".join(pip_deps))
    print("\nFor full setup, run: bash postBuild_modern.sh")

# Convenience functions for users
__all__ = [
    '__version__', '__git_info__', '__author__', '__email__', 
    'version_info', 'show_dependencies', 'get_version', 'get_git_info'
]
