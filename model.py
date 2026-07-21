"""
Federated Averaging (FedAvg) from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_mlp_classifier
import torch
import torch.nn as nn


def build_mlp_classifier(input_size, hidden_size, num_classes):
    # TODO: return an nn.Module mapping (N, input_size) floats to (N, num_classes) logits
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes),
    )

# Step 2 - build_synthetic_dataset
def build_synthetic_dataset(num_samples, input_size, num_classes, seed):
    # TODO: build a seeded synthetic dataset of (features, labels) tensors
    generator = torch.Generator().manual_seed(seed)

    features = torch.randn(
        num_samples,
        input_size,
        generator=generator,
        dtype=torch.float32,
    )

    labels = torch.randint(
        low=0,
        high=num_classes,
        size=(num_samples,),
        generator=generator,
        dtype=torch.long,
    )

    return features, labels

# Step 3 - train_test_split_dataset
def train_test_split_dataset(features, labels, test_fraction, seed):
    # TODO: seeded shuffle of row indices, then slice into train and test sets

    num_samples = len(features)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_samples, generator=generator)

    num_test = int(num_samples * test_fraction)
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    train_features = features[train_indices]
    train_labels = labels[train_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]

    return train_features, train_labels, test_features, test_labels

# Step 4 - partition_data_iid
import torch

def partition_data_iid(train_features, train_labels, num_clients, seed):
    # TODO: Shuffle the M rows with seed and split them evenly across num_clients clients.
    if num_clients <= 0:
        num_clients = 1
        
    # shuffle
    n = train_features.shape[0]
    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(n, generator=generator)

    client_index_splits = torch.tensor_split(
        shuffled_indices,
        num_clients,
    )

    out = []
    for client_idx in client_index_splits:
        client_feature = train_features[client_idx]
        client_label = train_labels[client_idx]

        out.append((client_feature, client_label))
    
    return out

# Step 5 - partition_data_non_iid
def partition_data_non_iid(train_features, train_labels, num_clients, shards_per_client, seed):
    # TODO: Sort the data by label and assign label-contiguous shards to each client.

    if num_clients <= 0:
        num_clients = 1
    
    # sort
    sorted_indices = torch.argsort(train_labels)

    client_index_splits = torch.tensor_split(
        sorted_indices,
        num_clients,
    )

    out = []
    for client_idx in client_index_splits:
        client_feature = train_features[client_idx]
        client_label = train_labels[client_idx]

        out.append((client_feature, client_label))
    
    return out

# Step 6 - count_client_samples
def count_client_samples(client_partitions):
    # TODO: return a list of per-client sample counts in the same order

    return [len(client[1]) for client in client_partitions]

# Step 7 - iterate_client_batches
def iterate_client_batches(client_features, client_labels, batch_size, seed):
    # TODO: shuffle one client's data with the seed and slice it into batches of size B

    # shuffle
    n = client_features.shape[0]
    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(n, generator=generator)

    shuffled_features = client_features[shuffled_indices]
    shuffled_labels = client_labels[shuffled_indices]

    # batch
    feature_batches = torch.split(shuffled_features, batch_size)
    label_batches = torch.split(shuffled_labels, batch_size)


    return list(zip(feature_batches, label_batches))

# Step 8 - compute_batch_loss
import torch.nn.functional as F

def compute_batch_loss(model, batch_features, batch_labels):
    # TODO: Compute the cross-entropy loss for one batch given the model
    logits = model(batch_features)
    loss = F.cross_entropy(logits, batch_labels)

    return loss

# Step 9 - local_sgd_step
import torch
import torch.nn.functional as F

def local_sgd_step(model, optimizer, batch_features, batch_labels):
    # TODO: perform one SGD update (forward, loss, backward, step) and return the float loss
    
    model.train()

    optimizer.zero_grad()

    loss = compute_batch_loss(model, batch_features, batch_labels)

    loss.backward()

    optimizer.step()

    return loss.item()

# Step 10 - train_client_local
def train_client_local(model, client_features, client_labels, local_epochs, batch_size, learning_rate, seed):
    # TODO: train one client for local_epochs of SGD and return its state dict

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    model.train()

    for epoch in range(local_epochs):
        batches = iterate_client_batches(
            client_features,
            client_labels,
            batch_size,
            seed + epoch,
        )

        for batch_features, batch_labels in batches:
            local_sgd_step(
                model,
                optimizer,
                batch_features,
                batch_labels,
            )

    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }

# Step 11 - clone_model_state
def clone_model_state(model):
    # TODO: Copy a model's parameters into a state dict of detached, cloned tensors.

    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

# Step 12 - load_model_state
def load_model_state(model, state_dict):
    # TODO: Load a state dict of parameters back into a model.

    result = model.load_state_dict(state_dict, strict=False)

    return model

# Step 13 - initialize_global_state (not yet solved)
# TODO: implement

# Step 14 - add_state_dicts (not yet solved)
# TODO: implement

# Step 15 - scale_state_dict (not yet solved)
# TODO: implement

# Step 16 - aggregate_weighted_average (not yet solved)
# TODO: implement

# Step 17 - select_round_clients (not yet solved)
# TODO: implement

# Step 18 - run_communication_round (not yet solved)
# TODO: implement

# Step 19 - evaluate_accuracy (not yet solved)
# TODO: implement

# Step 20 - run_fedavg (not yet solved)
# TODO: implement

# Step 21 - train_centralized_baseline (not yet solved)
# TODO: implement

# Step 22 - run_fedavg_iid (not yet solved)
# TODO: implement

# Step 23 - run_fedavg_non_iid (not yet solved)
# TODO: implement

# Step 24 - compute_non_iid_gap (not yet solved)
# TODO: implement

# Step 25 - rounds_to_target_vs_local_epochs (not yet solved)
# TODO: implement

# Step 26 - accuracy_vs_client_fraction (not yet solved)
# TODO: implement

