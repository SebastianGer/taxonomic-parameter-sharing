"""
This script creates a dummy taxonomic dataset, based on a full tree, indicating the class structure.
The dataset will be balanced, i.e. there will be the same amount of points for each combination of classes among
taxonomic layrs. Each data point has d features, where d is the number of layers in the tree.
The i-th feature is equal to the i-th class of the data point. If is_perfectly_solvable is False, the i-th feature
will be drawn from a normal distribution with the i-th class of the data point as its mean value and a small
standard distribution, to add some noise to the data.
"""
import itertools
import math
import pickle
from pathlib import Path
from random import normalvariate
from random import seed
from typing import List

import numpy as np
from treelib import Tree

seed(123)


def create_datapoint(class_vector: List[int], is_perfectly_solvable: bool = False) -> List[float]:
    datapoint = []
    if is_perfectly_solvable:
        sigma = 0.0
    else:
        sigma = 0.25
    for class_index in class_vector:
        feature = normalvariate(mu=class_index, sigma=sigma)
        datapoint.append(feature)
    return datapoint


def create_dataset(min_points: int, classes_per_layer: List[int], class_tree: Tree, is_perfectly_solvable=False) -> (
        np.array, np.array):

    class_value_ranges = []
    for classes_in_current_level in classes_per_layer:
        class_value_ranges.append(list(range(classes_in_current_level)))

    # Local classes are unique within one tree layer, but not across layers. They are used to choose the next branch
    # while traversing the class_tree, to find the global classes. Global classes are those given in the class_tree.
    # They are unique among all nodes of the class_tree.
    local_class_vectors = list(itertools.product(*class_value_ranges))
    n_repetitions = math.ceil(min_points / len(local_class_vectors))

    X = []
    Y = []

    for rep in range(n_repetitions):
        for class_vector in local_class_vectors:
            global_class_vector = []
            current_node_id = class_tree.root
            for local_class in class_vector:
                possible_next_branches = class_tree.get_node(current_node_id).successors(0)
                next_node_id = possible_next_branches[local_class]
                global_class_vector.append(next_node_id)
                current_node_id = next_node_id
            datapoint = create_datapoint(global_class_vector, is_perfectly_solvable)
            X.append(datapoint)
            Y.append(global_class_vector)

    return np.array(X), np.array(Y)


def create_tree_structure(classes_per_level: List[int]) -> Tree:
    # Root should always be initialized with 0, otherwise all calls to successor must be changed
    tree = Tree(identifier=0)
    tree.create_node(0, 0)

    nodes_to_populate = [0]
    nodes_to_populate_next = []
    new_node_id = 1

    # For each layer i in the tree, create classes_per_level[i] child nodes to existing parents on the previous layer.
    # The root is considered to be on layer -1 for this. The new nodes are consecutively numbered.
    for layer_number, classes_at_current_level in enumerate(classes_per_level):
        for parent_id in nodes_to_populate:
            for child_id in range(new_node_id, new_node_id + classes_per_level[layer_number]):
                tree.create_node(child_id, child_id, parent=parent_id)
                nodes_to_populate_next.append(child_id)
            new_node_id += classes_per_level[layer_number]
        nodes_to_populate = nodes_to_populate_next
        nodes_to_populate_next = []

    return tree


def write_output(output_dir: str, classes_per_level: List[int], class_tree: Tree, X: np.array, Y: np.array):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    with open(str(output_path / "classes_per_level.txt"), 'w') as file:
        str_representation = " ".join(map(str, classes_per_level))
        file.write(str_representation)

    with open(str(output_path / "class_tree.pickle"), "wb") as f:
        pickle.dump(class_tree, f)

    np.save(str(output_path / "X.npy"), X)
    np.save(str(output_path / "Y.npy"), Y)


def main():
    min_points = 1000
    classes_per_level = [3, 4, 2]
    output_dir = "./data"
    problem_is_perfectly_solvable = False

    class_tree = create_tree_structure(classes_per_level=classes_per_level)

    X, Y = create_dataset(min_points=min_points, classes_per_layer=classes_per_level, class_tree=class_tree,
                          is_perfectly_solvable=problem_is_perfectly_solvable)

    write_output(output_dir, classes_per_level, class_tree, X, Y)


if __name__ == "__main__":
    main()
