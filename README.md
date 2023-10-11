# ecnn4klm: Equivariant Convolutional Neural Network for Kondo Lattice Model

`ecnn4klm` aims to perform fast and large-scale simulations of localized spin dynamics in the Kondo lattice model using an equivariant convolutional neural network (ECNN). For details on the ECNN model, please refer to the paper: [Y. Miyazaki, <i>Mach. Learn.: Sci. Technol.</i> <b>4</b> 045006 (2023)](https://dx.doi.org/10.1088/2632-2153/acffa2).

## Installation

`ecnn4klm` is based on PyTorch and e3nn. To install, you need PyTorch version 1.8 or higher. I also highly recommend running it on a GPU to maximize performance.

```bash
pip install git+https://github.com/Miyazaki-Yu/ecnn4klm.git
```

## Brief Introduction with Colab

You can easily try it out on your browser using Google Colab.

<!-- <a href="http://colab.research.google.com/github/Miyazaki-Yu/blob/master/notebook/ecnn_test.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg"/>
</a> -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/Miyazaki-Yu/ecnn4klm/blob/master/notebook/ecnn_test.ipynb)

## Data

The models and the datasets for both the square lattice and the triangular lattice cases, which are discussed in the paper, can be found in the [`data/`](https://github.com/Miyazaki-Yu/ecnn4klm/tree/main/data) directory.

## How to Cite

```bibtex
@article{Miyazaki2023_ecnn4klm,
doi = {10.1088/2632-2153/acffa2},
url = {https://dx.doi.org/10.1088/2632-2153/acffa2},
year = {2023},
month = {oct},
publisher = {IOP Publishing},
volume = {4},
number = {4},
pages = {045006},
author = {Yu Miyazaki},
title = {Equivariant neural networks for spin dynamics simulations of itinerant magnets},
journal = {Machine Learning: Science and Technology},
}
```
