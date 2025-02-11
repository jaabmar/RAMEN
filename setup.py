from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="RAMEN",
    version="0.1",
    author="Javier Abad & Piersilvio de Bartolomeis",
    author_email="javier.abadmartinez@ai.ethz.ch",
    description="Python implementation of the methods introduced in the paper: Doubly robust identification of treatment effects from multiple environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaabmar/RAMEN",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="machine learning, causal inference, identifiability, observational studies, treatment effect, double robustness",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "xgboost",
        "scikit-learn",
        "scipy",
        "tqdm",
    ],
    python_requires=">=3.12",
)
