"""
setup.py: Configuration file for packaging and distributing the MLOps project.

This script uses setuptools to define the package metadata and dependencies required 
to run the project.
"""
from setuptools import setup, find_packages

setup(
    name="mlops_assignment",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
