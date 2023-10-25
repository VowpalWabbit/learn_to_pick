from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

with open("version.txt", "r", encoding="utf-8") as version_file:
    version = version_file.read().strip()

setup(
    name="learn_to_pick",
    version=version,
    install_requires=[
        "numpy>=1.24.4",
        "pandas>=2.0.3",
        "vowpal-wabbit-next==0.7.0",
        "sentence-transformers>=2.2.2",
        "torch",
        "pyskiplist",
        "parameterfree",
    ],
    extras_require={"dev": ["pytest", "black==23.10.0"]},
    author="VowpalWabbit",
    description="a python library for online learning RL loops, specialized for Contextual Bandit scenarios.",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    url="https://github.com/VowpalWabbit/learn_to_pick",
    python_requires=">=3.8",
)
