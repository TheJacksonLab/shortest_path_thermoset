# shortest_path_thermoset
Repository for shortest path analysis in thermosets

## 1. Introduction

Understanding fracture nucleation in thermoset polymers under deformation, i.e., how the initial bond scissions occur, remains a fundamental challenge due to the interwine of complex network topology and amorphous polymeric structure. This project introduces a theoretical approach based on the shortest path of the network topology to predict where and when the first bond breaks in network.  

This repo provides tools for computing the shortest path in bulk thermoset network with periodic boundaries, and sample simulation data for testing. More information can be found in our preprint paper, "Shortest paths govern fracture nucleation in thermoset networks" available [here](https://arxiv.org/abs/2405.03551).

## 2. Network analyses

As detailed in the paper, our shortest path involves three steps:

1. Create a series of multi-image networks. This is necessary because shortest paths (normalized) in periodic systems can have wavelengths of more than one image.

2. Compute shortest paths using Dijkstra’s algorithm. The source and sink nodes are selected to be the identical atoms but in different periodic images. As a consequence, these shortest paths are intrisincally constrained by the periodic boundaries and . This step is conducted for each multi-image network created in Step 1.

3. Delete redundant shortest paths.

### 2.1 Coarse-grained thermoset models



### 2.2 All-atom thermoset models

### 2.3 Simple network models

## 3. Simulations

This folder contains sample simulation inputs and outputs that are used for shortest path analyses. 

- The all-atom model setup is provided in [Machine Learning Bond Breaking](https://github.com/TheJacksonLab/QM_ML_BondBreaking).

- We performed CG simulations with three types of bonded interactions: harmonic, FENE, and quartic bonds. Both the CG network and AA network are crosslinked dynamically through MD simulations.

- The network model is a static network model, where the edges are formed between randomly distributed nodes based on a Gaussian probability distribution.
