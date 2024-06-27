# shortest_path_thermoset
Repository for shortest path analysis in thermosets

## 1. Introduction

Understanding the microscopic mechanicsm of initial bond scissions in polymer networks under deformation remains a challenge due to the interwine of complex network topology and amorphous polymeric structure. This project introduces a theoretical approach based on the shortest path of the network topology to predict where and when the first bond breaks in thermosets.  

This repo provides tools for computing the shortest path in bulk thermoset network with periodic boundaries, and sample simulation data for testing. More information can be found in our preprint paper, "Shortest paths govern fracture nucleation in thermoset networks" available [here](https://arxiv.org/abs/2405.03551).

## 2. Network Analyses and Simulations

As detailed in the paper, our shortest path involves three steps:

1. Create a series of multi-image networks. This is necessary because shortest paths (normalized) in periodic systems can have wavelengths of more than one image.

2. Compute shortest paths using Dijkstra’s algorithm. The source and sink nodes are selected to be the identical atoms but in different periodic images. This step is conducted for each multi-image network created in Step 1.

3. Delete redundant shortest paths.

This folder also contains sample simulation inputs and outputs that are used for shortest path analyses.

### 2.1 Coarse-grained thermoset models

We performed CG simulations with three types of bonded interactions: harmonic, FENE, and quartic bonds. Both the CG network and AA network are crosslinked dynamically through MD simulations.

`simulations/CG/SP_CG0.py` provides simple examples to compute the minimum shortest path and to visualize its relationship with the strain for the first bond breaking.

`simulations/CG/length_evolution.py` shows how to analyze the evolution of shortest path length evolution under defromations.

### 2.2 All-atom thermoset models

The all-atom model setup is provided in [Machine Learning Bond Breaking](https://github.com/TheJacksonLab/QM_ML_BondBreaking).

`simulations/AA/SP_dist.py` shows how to convert the AA configurations to crosslinker-based network (for efficiency) and computes the shortest path distributions for various degrees of crosslinking.

### 2.3 Simple network models

The network model is a static network model, where the edges are formed between randomly distributed nodes based on a Gaussian probability distribution.

`simulations/network/generate_network.py` provides tools to create the network model.

`simulations/run.py` creates multiple random network models and computes their shortest paths.