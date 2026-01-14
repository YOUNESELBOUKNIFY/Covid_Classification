import torch
import torch.nn as nn
from torchvision import models


class ClassificationHead(nn.Module):
    """Unified classification head for all models: Linear -> ReLU -> Dropout -> Linear"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.5, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNNClassifier(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.5, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 512, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (B,512,1,1)
        )
        self.head = ClassificationHead(input_dim=512, hidden_dim=hidden_dim, dropout=dropout, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)        # (B,512,1,1)
        x = torch.flatten(x, 1)     # (B,512)
        logits = self.head(x)       # (B,2)
        return logits


class VGG16Classifier(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.5, num_classes: int = 2, freeze_backbone: bool = True):
        super().__init__()

        # VGG16 pretrained
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # dimension avant classifier (c'est 25088 pour input 224x224)
        in_dim = vgg.classifier[0].in_features  # 25088

        # enlever le classifier d'origine
        vgg.classifier = nn.Identity()
        self.backbone = vgg

        # head unifiée
        self.head = ClassificationHead(input_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout, num_classes=num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)    # (B,25088)
        logits = self.head(feats)   # (B,2)
        return logits


class ViTClassifier(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.5, num_classes: int = 2, freeze_backbone: bool = True):
        super().__init__()

        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

        # dimension embedding ViT (768)
        in_dim = vit.heads.head.in_features

        # enlever la tête d'origine
        vit.heads = nn.Identity()
        self.backbone = vit

        self.head = ClassificationHead(input_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout, num_classes=num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)    # (B,768)
        logits = self.head(feats)   # (B,2)
        return logits


def create_model(
    model_name: str,
    num_classes: int = 2,
    hidden_dim: int = 256,
    dropout: float = 0.5,
    freeze_backbone: bool = True,
):
    model_name = model_name.lower().strip()

    if model_name == "cnn":
        return CNNClassifier(hidden_dim=hidden_dim, dropout=dropout, num_classes=num_classes)

    if model_name == "vgg16":
        return VGG16Classifier(hidden_dim=hidden_dim, dropout=dropout, num_classes=num_classes, freeze_backbone=freeze_backbone)

    if model_name == "vit":
        return ViTClassifier(hidden_dim=hidden_dim, dropout=dropout, num_classes=num_classes, freeze_backbone=freeze_backbone)

    raise ValueError(f"Unknown model: {model_name}. Choose from: cnn | vgg16 | vit")
