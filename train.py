"""
Trains a model using the TPS module, using the dataset created from create_dataset.py. This is only meant to
demonstrate how to use the TPS module.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch import softmax
from torch.nn import CrossEntropyLoss

from data_processing import compute_class_buckets, get_classes_to_buckets_function, get_buckets_to_classes_function
from tps import TPS


def load_data(data_dir):
    X = np.load(str(data_dir / "X.npy"))
    Y = np.load(str(data_dir / "Y.npy"))

    with open(str(data_dir / "class_tree.pickle"), "rb") as f:
        tree_structure = pickle.load(f)

    with open(str(data_dir / "classes_per_level.txt"), "r") as f:
        classes_per_Level = f.read()
        classes_per_Level = list(map(int, classes_per_Level.split()))

    Y = convert_classes_to_buckets(tree_structure, Y)

    X = torch.as_tensor(X, dtype=torch.float)
    Y = torch.as_tensor(Y)
    return X, Y, tree_structure, classes_per_Level


def convert_classes_to_buckets(tree_structure, Y):
    convert_class_to_bucket = compute_class_buckets(tree_structure)
    conversion_function = get_classes_to_buckets_function(convert_class_to_bucket)
    Y_bucket = np.apply_along_axis(conversion_function, axis=1, arr=Y)
    return Y_bucket


def build_model(input_dim, classes_per_level):
    model = torch.nn.Sequential(torch.nn.Linear(input_dim, 20, bias=True), torch.nn.ReLU(),
                                torch.nn.Linear(20, 20, bias=True), torch.nn.ReLU(),
                                torch.nn.Linear(20, 20, bias=True), torch.nn.ReLU(),
                                torch.nn.Linear(20, 20, bias=True), torch.nn.ReLU(),
                                TPS(20, classes_per_level))
    return model


def train_model(X, Y, model, optimizer, loss_func, n_epochs):
    for ep in range(n_epochs):
        losses = []
        Y_pred_list = model(X)
        for i, y_pred in enumerate(Y_pred_list):
            loss = loss_func(y_pred, Y[:, i])
            losses.append(loss)
        print(losses)
        loss_sum = sum(losses)

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
    return model


def compute_predictions(X, model, tree_structure):
    model.eval()
    Y_pred = model(X)
    classifications_per_level = []

    # Convert raw outputs to probabilities, then integer predictions
    for partial_output in Y_pred:
        predictions_on_level_i = softmax(partial_output, dim=1)
        predictions_on_level_i = predictions_on_level_i.detach().numpy()

        classifications_on_level_i = np.argmax(predictions_on_level_i, axis=1).tolist()
        classifications_per_level.append(classifications_on_level_i)

    # Convert from bucket representation to original classes
    Y_pred = np.array(classifications_per_level).T
    convert_buckets_to_classes = get_buckets_to_classes_function(tree_structure)
    Y_pred = np.apply_along_axis(func1d=convert_buckets_to_classes, axis=1, arr=Y_pred)
    return Y_pred


def main():
    n_epochs = 1000
    data_dir = Path("data")

    X, Y, tree_structure, classes_per_Level = load_data(data_dir)

    model = build_model(X.size()[1], classes_per_Level)
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = CrossEntropyLoss()

    model = train_model(X, Y, model, optimizer, loss_func, n_epochs)

    Y_pred = compute_predictions(X, model, tree_structure)
    print(f"\nPredictions:\n\n", Y_pred[:10, :])


if __name__ == "__main__":
    main()
