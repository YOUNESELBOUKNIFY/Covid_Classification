import torch
import torch.nn as nn
import torchvision.models as models


class ClassificationHead(nn.Module):
    """Unified classification head for all models."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.5, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNNClassifier(nn.Module):
    """Custom CNN classifier."""
    
    def __init__(self, num_classes: int = 2, hidden_dim: int = 256, dropout: float = 0.5):
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = ClassificationHead(512, hidden_dim, dropout, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


class VGG16Classifier(nn.Module):
    """VGG16 classifier with transfer learning."""
    
    def __init__(self, num_classes: int = 2, hidden_dim: int = 256, dropout: float = 0.5, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained VGG16
        vgg16 = models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Remove the classification head, keep only features
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        
        # Replace classifier with custom head
        self.head = ClassificationHead(512, hidden_dim, dropout, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


class ViTClassifier(nn.Module):
    """Vision Transformer classifier with transfer learning."""
    
    def __init__(self, num_classes: int = 2, hidden_dim: int = 256, dropout: float = 0.5, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained ViT
        vit = models.vit_b_16(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Extract feature extractor (everything except the head)
        self.vit = vit
        self.vit.heads = nn.Identity()  # Remove original head
        
        # Replace with custom head (ViT outputs 768 dimensions)
        self.head = ClassificationHead(768, hidden_dim, dropout, num_classes)
    
    def forward(self, x):
        # ViT forward pass
        x = self.vit._process_input(x)
        n, _, c = x.shape
        
        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat((batch_class_token, x), dim=1)
        
        x = self.vit.encoder(x)
        x = x[:, 0]  # Get class token output
        
        x = self.head(x)
        return x


def create_model(model_name: str, num_classes: int = 2, hidden_dim: int = 256, dropout: float = 0.5):
    """
    Create model by name.
    
    Args:
        model_name: 'cnn', 'vgg16', or 'vit'
        num_classes: Number of output classes
        hidden_dim: Hidden dimension for classification head
        dropout: Dropout rate
    
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    if model_name == 'cnn':
        model = CNNClassifier(num_classes, hidden_dim, dropout)
    elif model_name == 'vgg16':
        model = VGG16Classifier(num_classes, hidden_dim, dropout, pretrained=True)
    elif model_name == 'vit':
        model = ViTClassifier(num_classes, hidden_dim, dropout, pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model