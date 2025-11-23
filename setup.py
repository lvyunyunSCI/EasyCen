#!/usr/bin/env python3
"""
EasyCen - Genome-wide k-mer analysis for centromere detection and visualization
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="easycen",
    version="1.0.0",
    author="Yunyun Lv",
    author_email="lvyunyun_sci@foxmail.com",
    description="Genome-wide k-mer analysis for centromere detection and visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lyyunyun/EasyCen",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.6.0",
        "matplotlib>=3.3.0",
        "biopython>=1.78",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0",
        "cooler>=0.8.0",
        "trackc",
        "multiprocess>=0.70.0",
        "psutil",
        "numba"
    ],
    extras_require={
        "accel": ["numba>=0.53.0"],
    },
    entry_points={
        "console_scripts": [
            "easycen=easycen.__main__:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
