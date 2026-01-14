import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import COVIDDataset, load_data_from_directory
from src.models import create_model
from src.utils import (
    set_seed, get_device, split_dataset, compute_metrics,
    get_confusion_matrix, get_classification_report
)


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    return all_predictions, all_labels


def main(args):
    """Main evaluation function."""
    set_seed(42)
    device = get_device()
    
    print(f"\n{'='*60}")
    print(f"Evaluating {args.model.upper()} on {args.modality.upper()} images")
    print(f"{'='*60}")
    
    # Load checkpoint
    if not torch.cuda.is_available():
        checkpoint = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(args.ckpt_path)
    
    config = checkpoint['config']
    print(f"\nCheckpoint info:")
    print(f"  - Model: {config['model']}")
    print(f"  - Modality: {config['modality']}")
    print(f"  - Image size: {config['img_size']}")
    print(f"  - Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
    
    # Create model and load weights
    model = create_model(args.model, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Load data
    image_paths, labels = load_data_from_directory(args.data_root, args.modality)
    
    # Split data (use same split as training)
    _, _, _, _, test_paths, test_labels = split_dataset(
        image_paths, labels, train_ratio=0.8, val_ratio=0.1, seed=42
    )
    
    # Create test dataset and loader
    test_dataset = COVIDDataset(test_paths, test_labels, img_size=args.img_size, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"\nTest set size: {len(test_paths)}")
    print(f"  - Healthy: {test_labels.count(0)}")
    print(f"  - Disease: {test_labels.count(1)}")
    
    # Evaluate
    print(f"\nEvaluating on test set...")
    predictions, labels_array = evaluate(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(labels_array, predictions)
    
    print(f"\n{'='*60}")
    print(f"METRICS")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    
    # Confusion matrix
    cm = get_confusion_matrix(labels_array, predictions)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Healthy  Disease")
    print(f"Actual Healthy  {cm[0, 0]:5d}    {cm[0, 1]:5d}")
    print(f"       Disease  {cm[1, 0]:5d}    {cm[1, 1]:5d}")
    
    # Classification report
    class_names = ['Healthy', 'Disease']
    report = get_classification_report(labels_array, predictions, class_names)
    print(f"\nClassification Report:")
    print(report)
    
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate COVID classifier')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory of dataset')
    parser.add_argument('--modality', type=str, default='ct', choices=['ct', 'xray'], help='Imaging modality')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'vgg16', 'vit'], help='Model type')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    
    args = parser.parse_args()
    main(args)