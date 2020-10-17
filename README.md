# Taxonomic Parameter Sharing

This repository contains a PyTorch implementation of the taxonomic parameter sharing approach introduced in
[_Towards a Visipedia: Combining Computer Vision and Communities of Experts_](https://thesis.library.caltech.edu/11502/1/Towards_a_Visipedia__Tools_and_Techniques_for_Computer_Vision_Dataset_Collection%20%284%29.pdf) by Grant Van Horn.

It can be used to perform taxonomic classification, while saving a large amount of parameters by sharing 
parameters among nodes of the same layer of the taxonomic tree. For each layer, it creates a distinct new classification 
problem, that is independent of other layers, by assigning each node/class in the current layer to a bucket 
(or super class). For details, please refer to the original work linked above.

## Prerequisites

Apart from PyTorch, you will need to install the _treelib_ package, for example via `pip install treelib`. It is used
to construct the taxonomic class trees. 

## File Structure

**tps.py** - Contains the PyTorch module that classifies data taxonomically. It outputs the layer-wise classification 
into buckets.

**data_processing.py** - Contains the code to create buckets from the taxonomic class tree and to convert between the 
original classes and their layer-wise bucket representation. 

**create_dataset.py** - Creates a dummy taxonomic dataset.

**train.py** - Trains a TPS-model using the created dummy dataset.   

## Running the example code

To try the example, first run `python train.py`, to create a dummy dataset. For example, the setting
`classes_per_level = [3, 4, 2]` leads to the following taxonomic tree.

```
0
├── 1
│   ├── 4
│   │   ├── 16
│   │   └── 17
│   ├── 5
│   │   ├── 18
│   │   └── 19
│   ├── 6
│   │   ├── 20
│   │   └── 21
│   └── 7
│       ├── 22
│       └── 23
├── 2
│   ├── 8
│   │   ├── 24
│   │   └── 25
│   ├── 9
│   │   ├── 26
│   │   └── 27
│   ├── 10
│   │   ├── 28
│   │   └── 29
│   └── 11
│       ├── 30
│       └── 31
└── 3
    ├── 12
    │   ├── 32
    │   └── 33
    ├── 13
    │   ├── 34
    │   └── 35
    ├── 14
    │   ├── 36
    │   └── 37
    └── 15
        ├── 38
        └── 39
```

Then, run `python train.py` to train a model on this dataset. It will continually output the losses at each of the 
three layers.