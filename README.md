# ecnn4klm: Equivariant Convolutional Neural Network for Kondo Lattice Model

`ecnn4klm` aims to perform fast and large-scale simulations of localized spin dynamics in the Kondo lattice model using an equivariant convolutional neural network (ECNN). For details on the ECNN model, please refer to the paper: [arvix:2305.03804](https://arxiv.org/abs/2305.03804).

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
@misc{miyazaki2023equivariant,
      title={Equivariant Neural Networks for Spin Dynamics Simulations of Itinerant Magnets}, 
      author={Yu Miyazaki},
      year={2023},
      eprint={2305.03804},
      archivePrefix={arXiv},
      primaryClass={cond-mat.str-el}
}
```
