import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class COVIDDataset(Dataset):
    """Dataset for COVID classification (Healthy vs Disease)."""
    
    def __init__(self, image_paths: list, labels: list, img_size: int = 224, augment: bool = False):
        """
        Args:
            image_paths: List of image file paths
            labels: List of labels (0=Healthy, 1=Disease)
            img_size: Image size for resizing
            augment: Whether to apply data augmentation
        """
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        
        # Normalization using ImageNet statistics
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open image and convert to RGB if needed
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        return img, label


def load_data_from_directory(data_root: str, modality: str):
    """
    Load images from directory structure: root/modality/{Healthy,Disease}/
    
    Args:
        data_root: Root directory path
        modality: 'ct' or 'xray'
    
    Returns:
        Tuple of (image_paths, labels)
    """
    image_paths = []
    labels = []
    
    modality_map = {'ct': 'CT Scan', 'xray': 'X-ray'}
    modality_dir = modality_map.get(modality.lower(), modality)
    
    data_path = Path(data_root) / 'Covid' / modality_dir
    
    # Check if directory exists
    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_path}")
    
    # Load Healthy images (label 0)
    healthy_dir = data_path / 'Healthy'
    if healthy_dir.exists():
        for img_file in healthy_dir.glob('*'):
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_paths.append(str(img_file))
                labels.append(0)
    else:
        raise FileNotFoundError(f"Healthy directory not found: {healthy_dir}")
    
    # Load Disease images (label 1)
    disease_dir = data_path / 'Disease'
    if disease_dir.exists():
        for img_file in disease_dir.glob('*'):
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_paths.append(str(img_file))
                labels.append(1)
    else:
        raise FileNotFoundError(f"Disease directory not found: {disease_dir}")
    
    print(f"\nLoaded data from: {data_path}")
    print(f"Total images: {len(image_paths)}")
    print(f"  - Healthy: {labels.count(0)}")
    print(f"  - Disease: {labels.count(1)}")
    
    return image_paths, labels