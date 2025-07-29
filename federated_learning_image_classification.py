import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random
# ==================== MNIST Loader (IDX Format) ====================
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
        return images.astype(np.float32) / 255.0
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
def split_mnist_data_non_iid(images, labels, num_clients=100, classes_per_client=2, samples_per_class=300):
    # Reduce samples_per_class to allow more overlap
    client_data = []
    all_classes = list(range(10))
    class_indices = {k: np.where(labels == k)[0] for k in all_classes}

    # Ensure all classes are covered by some clients
    for i in range(num_clients):
        chosen_classes = np.random.choice(all_classes, classes_per_client, replace=False)
        indices = []
        for cls in chosen_classes:
            cls_indices = class_indices[cls]
            chosen = np.random.choice(cls_indices, samples_per_class, replace=False)
            indices.extend(chosen)
        # Add some overlapping samples
        shared_indices = np.random.choice(np.arange(len(images)), 100, replace=False)
        indices.extend(shared_indices)
        # Shuffle
        np.random.shuffle(indices)
        X = images[indices]
        y = labels[indices]
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        client_data.append(TensorDataset(X_tensor, y_tensor))
    return client_data
# ==================== Model Definition ====================
class SimpleNN(nn.Module):
    def __init__(self):  # Không cần tham số input_dim, output_dim
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)  # 10 classes cho MNIST
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)  # Reshape thành [batch, 1, 28, 28]
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# ==================== Training & Evaluation ====================
def train_client(model, train_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()
def test_global_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total
# ==================== Federated Learning Simulation ====================
class WirelessEdgeNetwork:
    def __init__(self, num_clients, input_dim, output_dim, client_data, test_dataset):
        self.num_clients = num_clients
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = SimpleNN().to(self.device)
        self.client_data = client_data
        self.test_data = test_dataset
        self.client_resources = {
            i: {
                'compute_power': np.random.uniform(0.5, 1.5),
                'bandwidth': np.random.uniform(5, 20),
                'battery': np.random.uniform(0.4, 1.0)
            }
            for i in range(num_clients)
        }
    def select_clients(self, fraction):
        num_selected = max(1, int(self.num_clients * fraction))
        client_scores = []
        for i in range(self.num_clients):
            res = self.client_resources[i]
            score = res['compute_power'] * 0.4 + res['bandwidth'] * 0.3 + res['battery'] * 0.3
            client_scores.append(score)
        selected_indices = np.argsort(client_scores)[-num_selected:]
        return selected_indices.tolist()
    def federated_averaging(self, client_updates, selected_clients):
        global_dict = self.global_model.state_dict()
        total_samples = sum([len(self.client_data[i]) for i in selected_clients])
        for key in global_dict:
            global_dict[key] = torch.zeros_like(global_dict[key])
        for key in global_dict:
            for i, idx in enumerate(selected_clients):
                # Simple averaging by sample count only
                weight = len(self.client_data[idx]) / total_samples
                global_dict[key] += weight * client_updates[i][key]
        self.global_model.load_state_dict(global_dict)
    def run_federated_learning(self, num_rounds, local_epochs, fraction, batch_size, lr):
        test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)
        accuracies = []
        for round in range(num_rounds):
            print(f"\nRound {round+1}/{num_rounds}")
            selected_clients = self.select_clients(fraction)
            print(f"Selected clients: {selected_clients}")
            client_updates = []
            for idx in selected_clients:
                print(f"Training client {idx}...")
                train_loader = DataLoader(self.client_data[idx], batch_size=batch_size, shuffle=True)
                local_model = SimpleNN().to(self.device)
                local_model.load_state_dict(self.global_model.state_dict())
                local_update = train_client(local_model, train_loader, local_epochs, lr, self.device)
                client_updates.append(local_update)
            self.federated_averaging(client_updates, selected_clients)
            acc = test_global_model(self.global_model, test_loader, self.device)
            accuracies.append(acc)
            print(f"Global model accuracy: {acc:.2f}%")
            for idx in selected_clients:
                self.client_resources[idx]['battery'] *= 0.95
        return accuracies
# ==================== Main Execution ====================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    # Load MNIST
    train_images = load_mnist_images(r"C:\MNIST\train-images.idx3-ubyte")
    train_labels = load_mnist_labels(r"C:\MNIST\train-labels.idx1-ubyte")
    test_images = load_mnist_images(r"C:\MNIST\t10k-images.idx3-ubyte")
    test_labels = load_mnist_labels(r"C:\MNIST\t10k-labels.idx1-ubyte")
    client_data = split_mnist_data_non_iid(train_images, train_labels)
    test_X_tensor = torch.tensor(test_images, dtype=torch.float32)
    test_y_tensor = torch.tensor(test_labels, dtype=torch.long)
    test_dataset = TensorDataset(test_X_tensor, test_y_tensor)
    # Parameters
    NUM_CLIENTS = 100
    NUM_ROUNDS = 20
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 10
    FRACTION = 0.1
    LEARNING_RATE = 0.01
    INPUT_DIM = 784
    OUTPUT_DIM = 10
    # Run
    edge_network = WirelessEdgeNetwork(NUM_CLIENTS, INPUT_DIM, OUTPUT_DIM, client_data, test_dataset)
    accs = edge_network.run_federated_learning(NUM_ROUNDS, LOCAL_EPOCHS, FRACTION, BATCH_SIZE, LEARNING_RATE)
    # Plot
    plt.plot(range(1, NUM_ROUNDS+1), accs, marker='o')
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy (%)")
    plt.title("Federated Learning on MNIST (Edge Network)")
    plt.grid(True)
    plt.show()
