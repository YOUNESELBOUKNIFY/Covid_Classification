import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import COVIDDataset, load_data_from_directory
from models import create_model
from utils import set_seed, get_device, split_dataset, create_output_dir


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def train(args):
    """Main training function."""
    # Set seed
    set_seed(args.seed)
    device = get_device()
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir, args.modality, args.model)
    
    # Load data
    print(f"\n{'='*60}")
    print(f"Training {args.model.upper()} on {args.modality.upper()} images")
    print(f"{'='*60}")
    
    image_paths, labels = load_data_from_directory(args.data_root, args.modality)
    
    # Split data
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = split_dataset(
        image_paths, labels, train_ratio=0.8, val_ratio=0.1, seed=args.seed
    )
    
    print(f"\nData split:")
    print(f"  - Train: {len(train_paths)} ({len(train_paths)/len(image_paths)*100:.1f}%)")
    print(f"    * Healthy: {train_labels.count(0)}")
    print(f"    * Disease: {train_labels.count(1)}")
    print(f"  - Validation: {len(val_paths)} ({len(val_paths)/len(image_paths)*100:.1f}%)")
    print(f"    * Healthy: {val_labels.count(0)}")
    print(f"    * Disease: {val_labels.count(1)}")
    print(f"  - Test: {len(test_paths)} ({len(test_paths)/len(image_paths)*100:.1f}%)")
    print(f"    * Healthy: {test_labels.count(0)}")
    print(f"    * Disease: {test_labels.count(1)}")
    
    # Create datasets and loaders
    train_dataset = COVIDDataset(train_paths, train_labels, img_size=args.img_size, augment=True)
    val_dataset = COVIDDataset(val_paths, val_labels, img_size=args.img_size, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = create_model(args.model, num_classes=2)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.model.upper()}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"{'='*60}")
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'config': {
                    'model': args.model,
                    'modality': args.modality,
                    'img_size': args.img_size,
                    'num_classes': 2,
                }
            }
            best_path = output_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Best model saved to {best_path}")
        
        # Save last model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'config': {
                'model': args.model,
                'modality': args.modality,
                'img_size': args.img_size,
                'num_classes': 2,
            }
        }
        last_path = output_dir / 'last.pt'
        torch.save(checkpoint, last_path)
    
    print(f"{'='*60}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoint saved to: {output_dir}")
    
    return model, best_val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train COVID classifier')
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory of dataset')
    parser.add_argument('--modality', type=str, default='ct', choices=['ct', 'xray'], help='Imaging modality')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'vgg16', 'vit'], help='Model type')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    
    args = parser.parse_args()
    train(args)