# Wasserstein Embedding for Graph Learning (WEGL)

This reposority contains a sample implementation code pertaining to our recent work, [Wasserstein Embedding for Graph Learning (WEGL)](https://arxiv.org/abs/2006.09430), in which we leverage the linear optimal transport (LOT) theory to introduce a novel and fast framework for embedding entire graphs in a vector space that can then be used for graph classification tasks.

The Jupyter Notebook included in this repository runs WEGL on the OGBG-molhiv dataset from the [Open Graph Benchmark](https://ogb.stanford.edu/), achieving state-of-the-art results on molecular property prediction using a downstream random forest classifier with much-reduced training complexity compared to the existing graph neural network (GNN) approaches.

For further details on the approach and more comprehensive evaluation results, please visit [our paper webpage](https://skolouri.github.io/wegl/).

## Requirements

* [OGB](https://ogb.stanford.edu/): The Open Graph Benchmark.
* [POT](https://pythonot.github.io/): A Python library for Optimal Transport.
* [PyTorch](https://pytorch.org/)
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [NumPy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [tqdm](https://tqdm.github.io/)
* [PrettyTable](https://pypi.org/project/PrettyTable/)


##

If you use WEGL in your work, please cite our paper using the BibTeX citation below:

```
@article{kolouri2020wasserstein,
  title={Wasserstein Embedding for Graph Learning},
  author={Kolouri, Soheil and Naderializadeh, Navid and Rohde, Gustavo K and Hoffmann, Heiko},
  journal={arXiv preprint arXiv:2006.09430},
  year={2020}
}
```
