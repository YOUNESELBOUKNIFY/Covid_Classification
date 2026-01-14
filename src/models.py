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
    
# ----------------------------
#  2) VGG16 from scratch (NO TL)
# ----------------------------
class VGG16ScratchClassifier(nn.Module):
    """
    VGG16 random init (from scratch).
    Output features dim = 25088 for 224x224 (512*7*7) after flatten.
    """
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.5, num_classes: int = 2):
        super().__init__()

        # torchvision compat: weights=None (new) / pretrained=False (old)
        try:
            vgg = models.vgg16(weights=None)
        except TypeError:
            vgg = models.vgg16(pretrained=False)

        in_dim = vgg.classifier[0].in_features  # 25088 for 224x224
        vgg.classifier = nn.Identity()          # keep features+avgpool+flatten -> (B,25088)
        self.backbone = vgg

        self.head = ClassificationHead(input_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)    # (B,25088)
        return self.head(feats)     # (B,2)

# ----------------------------
#  3) ViT from scratch (depth=1 or 2 blocks)
# ----------------------------
class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.
    Uses Conv2d with kernel=stride=patch_size to get (B, embed_dim, H/ps, W/ps)
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 256):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,224,224)
        x = self.proj(x)  # (B,embed_dim,14,14) if patch=16
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)

        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        return x


class MiniViTBackbone(nn.Module):
    """
    Minimal ViT backbone from scratch.
    Returns CLS embedding (B, embed_dim).
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 1,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # init (simple)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)  # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        x = torch.cat([cls, x], dim=1)          # (B,1+N,D)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # (B,D)
        return cls_out


class ViTScratchClassifier(nn.Module):
    """
    ViT from scratch with configurable depth:
      - vit1: depth=1
      - vit2: depth=2
    """
    def __init__(
        self,
        depth: int,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        vit_dropout: float = 0.1,
        head_hidden: int = 256,
        head_dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super().__init__()
        self.backbone = MiniViTBackbone(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=vit_dropout,
        )
        self.head = ClassificationHead(input_dim=embed_dim, hidden_dim=head_hidden, dropout=head_dropout, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)  # (B,embed_dim)
        return self.head(feats)   # (B,2)



def create_model(
    model_name: str,
    num_classes: int = 2,
    hidden_dim: int = 256,
    dropout: float = 0.5,
    freeze_backbone: bool = True,
    img_size: int = 224,         # utile pour ViT scratch
):
    model_name = model_name.lower().strip()

    if model_name == "cnn":
        return CNNClassifier(hidden_dim=hidden_dim, dropout=dropout, num_classes=num_classes)

    # -------- VGG16 Transfer Learning (si tu le gardes) --------
    if model_name == "vgg16":
        return VGG16Classifier(
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone
        )

    # -------- VGG16 from scratch (sans TL) --------
    if model_name in {"vgg16_scratch", "vgg16_scrach", "vgg16_from_scratch"}:
        return VGG16ScratchClassifier(
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_classes=num_classes
        )

    # -------- ViT Transfer Learning (si tu le gardes) --------
    if model_name == "vit":
        return ViTClassifier(
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone
        )

    # -------- ViT from scratch depth=1 / depth=2 --------
    if model_name in {"vit1", "vit-1"}:
        return ViTScratchClassifier(
            depth=1,
            img_size=img_size,
            head_hidden=hidden_dim,
            head_dropout=dropout,
            num_classes=num_classes
        )

    if model_name in {"vit2", "vit-2"}:
        return ViTScratchClassifier(
            depth=2,
            img_size=img_size,
            head_hidden=hidden_dim,
            head_dropout=dropout,
            num_classes=num_classes
        )

    raise ValueError(
        f"Unknown model: {model_name}. Choose from: cnn | vgg16 | vgg16_scratch | vit | vit1 | vit2"
    )

