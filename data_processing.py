from typing import List, Callable, Dict

from numpy.random import shuffle
from treelib import Tree


def compute_class_buckets(class_tree: Tree) -> List[Dict]:
    """
    Given the taxonomic class structure, this method distributes classes layer-wise into buckets (or superclasses).
    The function returns one dict per layer, representing the mapping into buckets. Returning several dicts instead of
    a single one allows classes to not be named uniquely across layers. E.g. each layer's classes could simply be
    numbered from 0 upwards.
    """
    current_nodes = [class_tree.root]
    next_nodes = []
    bucket_assignments_per_layer = []

    while True:
        # Find maximum number of children in current layer
        max_children = 0
        for current_node_id in current_nodes:
            current_node = class_tree.get_node(current_node_id)
            if not current_node.is_leaf():
                children_ids = current_node.successors(0)
                next_nodes.append(children_ids)
                max_children = max(max_children, len(children_ids))

        if len(next_nodes) == 0:
            break

        # Randomly assign each node that belongs to a given parent to a different bucket
        # Note: This might, in rare cases, lead to unbalanced bucket fillings. Ff e.g. most classes have three
        # children and one class has four children, then it could happen that the fourth bucket only contains
        # one class, while the others are filled equally.
        bucket_assignment = dict()
        bucket_list = list(range(max_children))
        for children_group in next_nodes:
            shuffle(bucket_list)
            for idx, child in enumerate(children_group):
                bucket_assignment[child] = bucket_list[idx]

        bucket_assignments_per_layer.append(bucket_assignment)
        current_nodes = [child_id for children_group in next_nodes for child_id in children_group]
        next_nodes = []

    return bucket_assignments_per_layer


def convert_classes_to_buckets(class_vector: List[int], bucket_assignments_per_layer: List[Dict]) -> List[int]:
    converted_target = [bucket_assignment[class_value] for class_value, bucket_assignment in
                        zip(class_vector, bucket_assignments_per_layer)]
    return converted_target


def get_classes_to_buckets_function(bucket_assignments_per_layer: List[Dict]) -> Callable:
    return lambda class_vector: convert_classes_to_buckets(class_vector, bucket_assignments_per_layer)


def convert_buckets_to_classes(bucket_vector: List[int], class_tree: Tree) -> List[int]:
    current_node = class_tree.root
    targets = []
    for bucket_id in bucket_vector:
        current_node = class_tree.get_node(current_node).successors(0)[bucket_id]
        targets.append(current_node)
    return targets


def get_buckets_to_classes_function(class_tree: Tree) -> Callable:
    return lambda bucket_vector: convert_buckets_to_classes(bucket_vector, class_tree)
