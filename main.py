"""
Main execution script for COVID Classification pipeline.
Orchestrates training, evaluation, and prediction for all models and modalities.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.train import train
from src.eval import main as evaluate
from src.predict import main as predict


def create_parser():
    """Create main argument parser."""
    parser = argparse.ArgumentParser(
        description='COVID Classification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models on CT images
  python main.py --action train_all --modality ct --epochs 10

  # Train single model
  python main.py --action train --modality ct --model vgg16 --epochs 15

  # Evaluate all models
  python main.py --action eval_all --modality ct

  # Evaluate single model
  python main.py --action eval --modality ct --model vgg16

  # Make prediction
  python main.py --action predict --model vgg16 --image_path test.jpg

  # Full pipeline (train all, eval all, predict)
  python main.py --action full_pipeline --modality ct --epochs 10 --image_path test.jpg
        """
    )
    
    # Action argument
    parser.add_argument(
        '--action',
        type=str,
        required=True,
        choices=['train', 'eval', 'predict', 'train_all', 'eval_all', 'full_pipeline'],
        help='Action to perform'
    )
    
    # Training arguments
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory of dataset')
    parser.add_argument('--modality', type=str, choices=['ct', 'xray'],
                        help='Imaging modality (required for train/eval)')
    parser.add_argument('--model', type=str, choices=['cnn', 'vgg16', 'vit'],
                        default='vgg16', help='Model type')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training/evaluation')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for checkpoints')
    
    # Evaluation arguments
    parser.add_argument('--ckpt_path', type=str,
                        help='Path to checkpoint for evaluation/prediction')
    
    # Prediction arguments
    parser.add_argument('--image_path', type=str,
                        help='Path to image for prediction')
    
    return parser


def train_all(args):
    """Train all three models on specified modality."""
    if not args.modality:
        print("Error: --modality required for train_all action")
        return
    
    models = ['cnn', 'vgg16', 'vit']
    results = {}
    
    print(f"\n{'='*70}")
    print(f"TRAINING ALL MODELS ON {args.modality.upper()} IMAGES")
    print(f"{'='*70}")
    
    for model_name in models:
        print(f"\n{'─'*70}")
        print(f"Training {model_name.upper()}...")
        print(f"{'─'*70}")
        
        train_args = argparse.Namespace(
            data_root=args.data_root,
            modality=args.modality,
            model=model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            img_size=args.img_size,
            seed=args.seed,
            output_dir=args.output_dir
        )
        
        try:
            model, best_acc = train(train_args)
            results[model_name] = best_acc
            print(f"✓ {model_name.upper()} training completed. Best accuracy: {best_acc:.2f}%")
        except Exception as e:
            print(f"✗ {model_name.upper()} training failed: {str(e)}")
            results[model_name] = None
    
    # Summary
    print(f"\n{'='*70}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*70}")
    for model_name, accuracy in results.items():
        status = f"{accuracy:.2f}%" if accuracy else "FAILED"
        print(f"{model_name.upper():10} : {status}")
    print(f"{'='*70}\n")
    
    return results


def eval_all(args):
    """Evaluate all three models on specified modality."""
    if not args.modality:
        print("Error: --modality required for eval_all action")
        return
    
    models = ['cnn', 'vgg16', 'vit']
    results = {}
    
    print(f"\n{'='*70}")
    print(f"EVALUATING ALL MODELS ON {args.modality.upper()} IMAGES")
    print(f"{'='*70}")
    
    for model_name in models:
        print(f"\n{'─'*70}")
        print(f"Evaluating {model_name.upper()}...")
        print(f"{'─'*70}")
        
        ckpt_path = Path(args.output_dir) / args.modality / model_name / 'best.pt'
        
        if not ckpt_path.exists():
            print(f"✗ Checkpoint not found: {ckpt_path}")
            print(f"  Please train the model first!")
            results[model_name] = None
            continue
        
        eval_args = argparse.Namespace(
            ckpt_path=str(ckpt_path),
            data_root=args.data_root,
            modality=args.modality,
            model=model_name,
            batch_size=args.batch_size,
            img_size=args.img_size
        )
        
        try:
            evaluate(eval_args)
            results[model_name] = "OK"
            print(f"✓ {model_name.upper()} evaluation completed")
        except Exception as e:
            print(f"✗ {model_name.upper()} evaluation failed: {str(e)}")
            results[model_name] = None
    
    # Summary
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}")
    for model_name, status in results.items():
        status_str = "COMPLETED" if status else "FAILED"
        print(f"{model_name.upper():10} : {status_str}")
    print(f"{'='*70}\n")
    
    return results


def run_predict(args):
    """Run prediction on a single image."""
    if not args.image_path:
        print("Error: --image_path required for predict action")
        return
    
    if not args.ckpt_path:
        # Try to find best checkpoint
        ckpt_path = Path(args.output_dir) / 'ct' / args.model / 'best.pt'
        if not ckpt_path.exists():
            print(f"Error: --ckpt_path required or checkpoint not found at {ckpt_path}")
            return
        args.ckpt_path = str(ckpt_path)
    
    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"Error: Image not found at {args.image_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"PREDICTION")
    print(f"{'='*70}")
    
    predict_args = argparse.Namespace(
        ckpt_path=args.ckpt_path,
        image_path=args.image_path,
        model=args.model,
        img_size=args.img_size
    )
    
    try:
        predict(predict_args)
        print(f"✓ Prediction completed")
    except Exception as e:
        print(f"✗ Prediction failed: {str(e)}")
    
    print(f"{'='*70}\n")


def full_pipeline(args):
    """Execute full pipeline: train all, eval all, predict."""
    if not args.modality:
        print("Error: --modality required for full_pipeline action")
        return
    
    print(f"\n{'='*70}")
    print(f"FULL PIPELINE EXECUTION")
    print(f"{'='*70}")
    
    # Step 1: Train all models
    print(f"\n[STEP 1/3] Training all models...")
    train_results = train_all(args)
    
    # Step 2: Evaluate all models
    print(f"\n[STEP 2/3] Evaluating all models...")
    eval_results = eval_all(args)
    
    # Step 3: Make prediction (if image provided)
    if args.image_path:
        print(f"\n[STEP 3/3] Making predictions...")
        # Use best model for prediction
        ckpt_path = Path(args.output_dir) / args.modality / args.model / 'best.pt'
        if ckpt_path.exists():
            args.ckpt_path = str(ckpt_path)
            run_predict(args)
        else:
            print(f"Skipping prediction: checkpoint not found")
    else:
        print(f"\n[STEP 3/3] Skipped (--image_path not provided)")
    
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETED")
    print(f"{'='*70}\n")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    print(f"\n{'#'*70}")
    print(f"# COVID CLASSIFICATION PIPELINE")
    print(f"# Action: {args.action.upper()}")
    print(f"{'#'*70}")
    
    try:
        if args.action == 'train':
            if not args.modality:
                print("Error: --modality required for train action")
                sys.exit(1)
            train(args)
        
        elif args.action == 'eval':
            if not args.modality:
                print("Error: --modality required for eval action")
                sys.exit(1)
            if not args.ckpt_path:
                # Try to find best checkpoint
                ckpt_path = Path(args.output_dir) / args.modality / args.model / 'best.pt'
                if not ckpt_path.exists():
                    print(f"Error: --ckpt_path required or checkpoint not found")
                    sys.exit(1)
                args.ckpt_path = str(ckpt_path)
            evaluate(args)
        
        elif args.action == 'predict':
            run_predict(args)
        
        elif args.action == 'train_all':
            train_all(args)
        
        elif args.action == 'eval_all':
            eval_all(args)
        
        elif args.action == 'full_pipeline':
            full_pipeline(args)
        
        print("✓ Execution completed successfully!")
    
    except KeyboardInterrupt:
        print("\n\n⚠ Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()