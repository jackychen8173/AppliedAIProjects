import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def get_predictions(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_confs = []

    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_confs.extend(confs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_confs)


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def cost_weighted_accuracy(y_true, y_pred, cost_matrix):
    total_cost = 0
    max_cost = 0

    for t, p in zip(y_true, y_pred):
        total_cost += cost_matrix[t, p]
        max_cost += cost_matrix[t, t]

    return max_cost / total_cost


def plot_training_curves(train_losses, val_accuracies, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
