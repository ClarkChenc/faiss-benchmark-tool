#!/usr/bin/env python3
"""
FAISS Benchmark Framework Setup Script
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="faiss-benchmark",
    version="1.0.0",
    author="FAISS Benchmark Team",
    author_email="your-email@example.com",
    description="A comprehensive benchmarking framework for FAISS library",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/faiss-benchmark",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "gpu": ["faiss-gpu>=1.7.4"],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "faiss-benchmark=faiss_benchmark.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "faiss_benchmark": ["config/*.yaml", "examples/*.py"],
    },
    keywords="faiss, benchmark, vector search, similarity search, machine learning",
    project_urls={
        "Bug Reports": "https://github.com/your-repo/faiss-benchmark/issues",
        "Source": "https://github.com/your-repo/faiss-benchmark",
        "Documentation": "https://github.com/your-repo/faiss-benchmark/wiki",
    },
)