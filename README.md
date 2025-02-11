# RAMEN: Robust ATE identification from Multiple ENvironments

<!-- [![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2502.04262-B31B1B.svg)](https://arxiv.org/abs/2502.04262) -->
[![Python 3.13.2](https://img.shields.io/badge/python-3.13.2-blue.svg)](https://python.org/downloads/release/python-3132/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Pytorch 2.6.0](https://img.shields.io/badge/pytorch-2.6.0-green.svg)](https://pytorch.org/)

## Table of Contents
- [Overview](#overview)
- [Contents](#contents)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact](#contact)
<!-- - [Citation](#citation) -->

---

## Overview

![HAIPW Diagram](diagram.jpg)

This repository contains the Python implementation of **Robust ATE identification from Multiple ENvironments (RAMEN)**, as introduced in the paper *"Doubly robust identification of treatment effects from multiple environments"*.

<!-- For more details, see our [research paper](https://arxiv.org/abs/2502.04262). -->

---

## Contents

The `RAMEN` folder contains the core package:

```bash
RAMEN/
│── data/                              # Data loading and preprocessing
│   ├── datasets/                      # Dataset files (e.g., ihdp_obs.csv)
│   ├── data.py                        # Data processing functions
│
│── evaluations/                       # Evaluation scripts and utilities
│   ├── evaluate_semi_synthetic.py     # Evaluation on semi-synthetic datasets
│   ├── evaluate_synthetic.py          # Evaluation on synthetic datasets
│   ├── evaluations_utils.py           # Evaluation helper functions
│
│── models/                            # Model implementations
│   ├── instant_ramen.py               # Instant RAMEN model
│   ├── IRM.py                         # Invariant Risk Minimization (IRM) model
│   ├── ramen.py                       # RAMEN model
│   ├── models_utils.py                # Model utility functions
```

---

## Getting Started

### **Dependencies**

This package is set up with **Python 3.13.2** and the following libraries. These versions represent a possible implementation:

```txt
numpy==2.2.2
pandas==2.2.3
torch==2.6.0
scikit-learn==1.6.1
scipy==1.15.1
tqdm==4.67.1
xgboost==2.1.4
```

### Installation

#### Step 1: Create and Activate a Conda Environment

```bash
conda create -n ramen_env python -y
conda activate ramen_env
```
#### Step 2: Install the Package

Start by cloning the repository from GitHub. Then, upgrade `pip` to its latest version and use the local setup files to install the package.

```bash
git clone https://github.com/jaabmar/RAMEN.git
cd RAMEN
pip install --upgrade pip
pip install -e .
```

---

## Usage

### Running Synthetic Experiments

To run synthetic experiments, navigate to the `RAMEN/evaluations` folder and execute:

```bash
python evaluate_synthetic.py --n_env 5 --n 1000 --n_features 2 --invariance Y --post_treatment collider --n_post 2 --ate 3.0 --seed 1
```

#### Arguments

- `--n_env`: Number of environments
- `--n`: Number of samples for each environment
- `--n_features`: Dimension of pre-treatment features
- `--invariance`: Invariance setting (`Y`, `T`, `TY`)
- `--post_treatment`: Type of post-treatment features (`collider`, `descendant`, `noise`)
- `--n_post`: Number of post-treatment features
- `--ate`: Average treatment effect
- `--seed`: Random seed for reproducibility

---

## Contributing

We welcome contributions to improve this project. Here's how you can contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -m "Description of change"`)
4. Push to your branch (`git push origin feature-branch`)
5. Open a Pull Request

---

## Contact

For questions or collaborations, feel free to reach out:

- Javier Abad Martinez - [javier.abadmartinez@ai.ethz.ch](mailto:javier.abadmartinez@ai.ethz.ch)
- Piersilvio de Bartolomeis - [piersilvio.debartolomeis@inf.ethz.ch](mailto:piersilvio.debartolomeis@inf.ethz.ch)
- Julia Kostin - [jkostin@ethz.ch](jkostin@ethz.ch)

---

## Citation

If you find this code useful, please consider citing our paper:
 ```
@article{debartolomeis2025doubly,
      title={Doubly robust identification of treatment effects from multiple environments}, 
      author={Piersilvio De Bartolomeis and Julia Kostin and Javier Abad and Yixin Wang and Fanny Yang},
      journal={International Conference on Learning Representations},
      year={2025},
}
```