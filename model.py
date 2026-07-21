"""
Federated Averaging (FedAvg) from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_mlp_classifier
import torch
import torch.nn as nn

class _MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


def build_mlp_classifier(input_size, hidden_size, num_classes):
    # TODO: return an nn.Module mapping (N, input_size) floats to (N, num_classes) logits
    return _MLPClassifier(
        input_size,
        hidden_size,
        num_classes,
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

# Step 13 - initialize_global_state
def initialize_global_state(input_size, hidden_size, num_classes, seed):
    # TODO: seed torch, build a fresh MLP, and return its cloned starting state dict

    torch.manual_seed(seed)

    model = build_mlp_classifier(input_size, hidden_size, num_classes)

    return clone_model_state(model)

# Step 14 - add_state_dicts
def add_state_dicts(state_a, state_b):
    # TODO: return a new state dict with elementwise sums per matching key

    return {name: state_a[name] + state_b[name] for name in state_a.keys()}

# Step 15 - scale_state_dict
def scale_state_dict(state_dict, weight):
    # TODO: return a new state dict with each tensor multiplied by weight

    return {name: state*weight for name, state in state_dict.items()}

# Step 16 - aggregate_weighted_average
from functools import reduce

def aggregate_weighted_average(client_states, client_sample_counts):
    # TODO: combine client state dicts into a sample-weighted FedAvg average

    total_sample = sum(client_sample_counts)
    scaled_states = [scale_state_dict(states, count/total_sample) for states, count in zip(client_states, client_sample_counts)]

    avg_states = reduce(add_state_dicts, scaled_states)

    return avg_states

# Step 17 - select_round_clients
import builtins
import torch

def select_round_clients(num_clients, client_fraction, seed):
    # TODO: pick max(1, round(client_fraction*num_clients)) distinct client indices
    num_clients = int(num_clients)
    client_fraction = float(client_fraction)
    seed = int(seed)

    num_selected = max(
        1,
        builtins.round(client_fraction * num_clients),
    )
    num_selected = min(num_selected, num_clients)

    generator = torch.Generator()
    generator.manual_seed(seed)

    selected_clients = torch.multinomial(
        torch.ones(num_clients),
        num_samples=num_selected,
        replacement=False,
        generator=generator,
    )

    return torch.sort(selected_clients).values.tolist()

# Step 18 - run_communication_round
def run_communication_round(global_state, client_partitions, selected_clients, model_config, local_epochs, batch_size, learning_rate, seed):
    # TODO: train each selected client from the global state, then weighted-average their states
    input_size = int(model_config["input_size"])
    hidden_size = int(model_config["hidden_size"])
    num_classes = int(model_config["num_classes"])

    client_states = []
    client_sample_counts = []

    for client in selected_clients:
        client_index = int(client)

        client_features, client_labels = client_partitions[client_index]

        model = build_mlp_classifier(
            input_size,
            hidden_size,
            num_classes,
        )
        load_model_state(model, global_state)

        local_state = train_client_local(
            model,
            client_features,
            client_labels,
            int(local_epochs),
            int(batch_size),
            float(learning_rate),
            int(seed) + client_index,
        )

        client_states.append(local_state)
        client_sample_counts.append(client_labels.shape[0])

    return aggregate_weighted_average(
        client_states,
        client_sample_counts,
    )

# Step 19 - evaluate_accuracy
def evaluate_accuracy(model, test_features, test_labels):
    # TODO: run a no-grad forward pass and return the fraction of correct argmax predictions

    model.eval()

    with torch.no_grad():
        logits = model(test_features)
        predictions = torch.argmax(logits, dim=-1)

        accuracy = (predictions == test_labels).float().mean()

    return accuracy.item()

# Step 20 - run_fedavg
def run_fedavg(client_partitions, test_features, test_labels, model_config, num_rounds, client_fraction, local_epochs, batch_size, learning_rate, seed):
    # TODO: init global state, then loop rounds: select clients, run round, evaluate.
    input_size = int(model_config["input_size"])
    hidden_size = int(model_config["hidden_size"])
    num_classes = int(model_config["num_classes"])

    num_rounds = int(num_rounds)
    client_fraction = float(client_fraction)
    local_epochs = int(local_epochs)
    batch_size = int(batch_size)
    learning_rate = float(learning_rate)
    seed = int(seed)

    num_clients = len(client_partitions)

    global_state = initialize_global_state(
        input_size,
        hidden_size,
        num_classes,
        seed,
    )

    accuracy_history = []

    for round_index in range(num_rounds):
        round_seed = seed + round_index

        selected_clients = select_round_clients(
            num_clients,
            client_fraction,
            round_seed,
        )

        global_state = run_communication_round(
            global_state,
            client_partitions,
            selected_clients,
            model_config,
            local_epochs,
            batch_size,
            learning_rate,
            round_seed,
        )

        model = build_mlp_classifier(
            input_size,
            hidden_size,
            num_classes,
        )
        load_model_state(model, global_state)

        accuracy_history.append(
            evaluate_accuracy(model, test_features, test_labels)
        )

    model = build_mlp_classifier(
        input_size,
        hidden_size,
        num_classes,
    )
    load_model_state(model, global_state)

    return model, accuracy_history

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

