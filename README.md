# Multi-Graph Graph Attention Network (MG-GAT)

This repository holds the Tensorflow based implementation of Multi-Graph Graph Attention Network (MG-GAT) proposed in the 
[Interpretable Recommender System With Heterogeneous Information: A Geometric Deep Learning Perspective](http://dx.doi.org/10.2139/ssrn.3696092). 

## Getting started

We recommend using a conda virtual environment:
```
conda create -n mggat_env python=3.7
conda activate mggat_env
```
Install TensorFlow (your installation may vary):
```
conda install tensorflow-gpu==2.4.1
```
Pip install packages:
```
pip install ray==0.8.7 ray[tune] hyperopt pandas scikit-learn
```
To train our model on the MovieLens100K dataset, run:
```
python models.py
```
Check models.py to change arguments for model, dataset, etc.

## Code

A. datasets.py - Preprocessing for each dataset.

B. layers.py - Definitions of neural network classes, including GAT and GCN.

C. metrics.py - Definitions of metrics used to evaluate recommender systems.

D. models.py - Definitions of recommender systems and code to tune/test them. We include our model as well as some of the benchmarks we used ([SVD++](https://people.engr.tamu.edu/huangrh/Spring16/papers_course/matrix_factorization.pdf), [GRALS](https://arxiv.org/pdf/1908.09393v2.pdf), and [MGCNN](https://papers.nips.cc/paper/2017/file/2eace51d8f796d04991c831a07059758-Paper.pdf)). For other benchmarks, we refer you to their github implementations: [IGMC](https://github.com/muhanzhang/IGMC), [GraphRec](https://github.com/wenqifan03/GraphRec-WWW19), [NGCF](https://github.com/xiangwang1223/neural_graph_collaborative_filtering), [F-EAE](https://github.com/mravanba/deep_exchangeable_tensors), [GC-MC](https://github.com/riannevdberg/gc-mc), [NNMF](https://github.com/jstol/neural-net-matrix-factorization).

E. results.py - Print table of metrics after running models.py.

## Data
We release the processed dataset in our paper from the [Yelp data challenge](https://www.yelp.com/dataset). 

data/datasets - Standardized datasets.

data/raw_data - Unprocessed datasets.

data/results - Saved metrics, hyperparameters, and models.

## Reference
If you use this code as part of your research, please cite the following paper: 
```
@article{leng2020interpretable,
  title={Interpretable recommender system with heterogeneous information: A geometric deep learning perspective},
  author={Leng, Yan and Ruiz, Rodrigo and Dong, Xiaowen and Pentland, Alex}
}
```
