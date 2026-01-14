import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

from src.models import create_model
from src.dataset import COVIDDataset
from src.utils import get_device


def predict_image(model, image_path, img_size=224, device='cpu'):
    """
    Predict class for a single image.
    
    Args:
        model: Model instance
        image_path: Path to image file
        img_size: Image size
        device: Device to use
    
    Returns:
        Tuple of (predicted_class, probabilities)
    """
    model.eval()
    
    # Load and transform image
    img = Image.open(image_path).convert('RGB')
    
    # Create temporary dataset to use its transform
    dataset = COVIDDataset([image_path], [0], img_size=img_size, augment=False)
    img_tensor = dataset[0][0].unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(1).item()
        confidence = probabilities[0, predicted_class].item()
    
    return predicted_class, probabilities[0].cpu().numpy(), confidence


def main(args):
    """Main prediction function."""
    device = get_device()
    
    # Check image exists
    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"Error: Image not found at {args.image_path}")
        return
    
    # Load checkpoint
    if not torch.cuda.is_available():
        checkpoint = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(args.ckpt_path)
    
    config = checkpoint['config']
    
    # Create model and load weights
    model = create_model(args.model, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Predict
    print(f"\n{'='*60}")
    print(f"COVID Classification Prediction")
    print(f"{'='*60}")
    print(f"Model: {args.model.upper()}")
    print(f"Image: {args.image_path}")
    
    predicted_class, probabilities, confidence = predict_image(
        model, args.image_path, img_size=args.img_size, device=device
    )
    
    class_names = ['Healthy', 'Disease']
    predicted_name = class_names[predicted_class]
    
    print(f"\nPrediction: {predicted_name}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"\nProbabilities:")
    print(f"  - Healthy: {probabilities[0]*100:.2f}%")
    print(f"  - Disease: {probabilities[1]*100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict COVID classification for an image')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'vgg16', 'vit'], help='Model type')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    
    args = parser.parse_args()
    main(args)