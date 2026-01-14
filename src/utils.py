import os
import random
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the device (cuda or cpu)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")
    return device


def split_dataset(image_paths: list, labels: list, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
    """
    Split dataset into train, val, test sets maintaining class balance.
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        train_ratio: Proportion for training (default 0.8)
        val_ratio: Proportion for validation (default 0.1)
        seed: Random seed
    
    Returns:
        Tuple of (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)
    """
    set_seed(seed)
    
    # Group by class
    class_indices = {}
    for idx, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    
    # Split each class separately
    for class_label, indices in class_indices.items():
        random.shuffle(indices)
        n = len(indices)
        train_idx = int(n * train_ratio)
        val_idx = train_idx + int(n * val_ratio)
        
        for i in indices[:train_idx]:
            train_paths.append(image_paths[i])
            train_labels.append(labels[i])
        
        for i in indices[train_idx:val_idx]:
            val_paths.append(image_paths[i])
            val_labels.append(labels[i])
        
        for i in indices[val_idx:]:
            test_paths.append(image_paths[i])
            test_labels.append(labels[i])
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dict with accuracy, precision, recall, f1
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return metrics


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """Get confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list):
    """Get detailed classification report."""
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)


def create_output_dir(output_dir: str, modality: str, model_name: str):
    """Create output directory structure."""
    path = Path(output_dir) / modality / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path