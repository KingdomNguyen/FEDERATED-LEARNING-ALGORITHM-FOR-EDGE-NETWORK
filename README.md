# FEDERATED-LEARNING-ALGORITHM-FOR-EDGE-NETWORK
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)

*Federated Learning in Wireless Edge Network*

## Table of Contents
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)
- [License](#license)

## Problem Statement
The project focuses on Federated Learning (FL) in wireless edge networks, inspired by the paper "*Communication-Efficient Learning of Deep Networks from Decentralized Data*" by McMahan et al.( this is not paper source code, you may find original code in [https://github.com/AshwinRJ/Federated-Learning-PyTorch](https://github.com/AshwinRJ/Federated-Learning-PyTorch)). 

The core problem addressed in this research revolves around training machine learning models in scenarios where data is decentralized, privacy-sensitive, and distributed across a large number of devices—such as smartphones, IoT devices, or edge servers—without centralizing the raw data. This challenge emerges from real-world constraints where users generate valuable data locally (e.g., typing patterns, photos, or sensor readings), but sharing this data directly to a central server is impractical or undesirable due to privacy concerns, legal regulations (like GDPR), or communication bottlenecks.

Traditional machine learning relies on collecting all data in a central server for training, which becomes infeasible when dealing with sensitive or massive datasets. For example, millions of users typing on mobile keyboards generate text data that could improve language models, but uploading all keystrokes to a cloud server would expose private messages, passwords, or URLs. Similarly, medical data from wearable devices cannot be centralized due to privacy laws.

To overcome this, the paper proposes Federated Learning (FL), a framework where devices collaboratively train a shared model while keeping their raw data locally. Instead of sending data to the server, devices compute updates to the model (e.g., gradients or weights) based on their local data, and only these updates are shared. The server aggregates these updates to improve the global model iteratively.
## Methodology
1. Data Partitioning (Non-IID Setup)
   In FL, data is distributed across devices in a non-IID (non-identically distributed) manner. For example, one user’s phone might have mostly cat photos, while another’s has landscapes.This can be simulated by splitting MNIST so each client (device) holds data from only 2 digit classes, creating a realistic non-IID scenario:
   ```python
   def split_mnist_data_non_iid(images, labels, num_clients=100, classes_per_client=2, samples_per_class=300):
    client_data = []
    all_classes = list(range(10))
    class_indices = {k: np.where(labels == k)[0] for k in all_classes}

    for i in range(num_clients):
        chosen_classes = np.random.choice(all_classes, classes_per_client, replace=False)
        indices = []
        for cls in chosen_classes:
            cls_indices = class_indices[cls]
            chosen = np.random.choice(cls_indices, samples_per_class, replace=False)
            indices.extend(chosen)
        # ... (shuffle and create client datasets)
   ```
2. Federated Averaging (FedAvg)
   Centralized training is impossible; we need a way to aggregate local updates without sharing raw data. To address this, FedAvg averages model parameters from clients instead of raw gradients:
   - Local Training: Each client trains the global model on its local data.
   - Averaging: The server combines these updates proportionally to each client’s dataset size.
   ```python
   def federated_averaging(self, client_updates, selected_clients):
    global_dict = self.global_model.state_dict()
    total_samples = sum([len(self.client_data[i]) for i in selected_clients])
    
    for key in global_dict:
        global_dict[key] = torch.zeros_like(global_dict[key])
        for i, idx in enumerate(selected_clients):
            weight = len(self.client_data[idx]) / total_samples  # Weight by dataset size
            global_dict[key] += weight * client_updates[i][key]  # Weighted average
    self.global_model.load_state_dict(global_dict)
   ```
3. Client Selection (Resource-Aware)
    In edge networks, devices have varying resources (battery, bandwidth). Training must prioritize "capable" devices in order to avoid stragglers (slow devices) and saves energy for low-battery devices. I select clients based on a weighted score of their resources:
   ```python
   def select_clients(self, fraction):
    num_selected = max(1, int(self.num_clients * fraction))
    client_scores = []
    for i in range(self.num_clients):
        res = self.client_resources[i]
        score = res['compute_power'] * 0.4 + res['bandwidth'] * 0.3 + res['battery'] * 0.3
        client_scores.append(score)
    selected_indices = np.argsort(client_scores)[-num_selected:]  # Top performers
    return selected_indices.tolist()
   ```
4. Local Training
   Devices must train efficiently with limited compute power.We use SGD for local training, with configurable epochs (LOCAL_EPOCHS):
   ```python
   def train_client(model, train_loader, epochs, lr, device):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):  # Multiple local passes
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()
   ```
5. Evaluation
   Measure global model performance without centralized data by test the aggregated model on a held-out global test set to ensure the federated model generalizes beyond individual clients’ data.
   ```python
   def test_global_model(model, test_loader, device):
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target).sum().item()
    return 100 * correct / total_samples
   ```
## Installation
### Prerequisites
- Python 3.8+ (Tested with 3.10)
- PyTorch
- MNIST dataset
## Result

