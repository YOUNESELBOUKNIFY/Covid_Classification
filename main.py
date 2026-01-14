"""
Main execution script for COVID Classification pipeline.
Orchestrates training, evaluation, and prediction for all models and modalities.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so "src" is importable
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import train
from src.eval import main as evaluate
from src.predict import main as predict


SUPPORTED_MODELS = ["cnn", "vgg16_scratch", "vit1", "vit2"]


def create_parser():
    """Create main argument parser."""
    parser = argparse.ArgumentParser(
        description="COVID Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models on CT images
  python main.py --action train_all --modality ct --epochs 10

  # Train single model (VGG16 scratch)
  python main.py --action train --modality ct --model vgg16_scratch --epochs 10

  # Train single model (ViT scratch depth=1)
  python main.py --action train --modality ct --model vit1 --epochs 10

  # Evaluate all models
  python main.py --action eval_all --modality ct

  # Evaluate single model (auto finds best.pt if --ckpt_path omitted)
  python main.py --action eval --modality ct --model vit2

  # Make prediction (auto finds best.pt if --ckpt_path omitted)
  python main.py --action predict --model vgg16_scratch --image_path test.jpg --modality ct
        """
    )

    # Action argument
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=["train", "eval", "predict", "train_all", "eval_all", "full_pipeline"],
        help="Action to perform",
    )

    # Common arguments
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory of dataset (parent of Covid/ or Covid/ itself)")
    parser.add_argument("--modality", type=str, choices=["ct", "xray"], help="Imaging modality (required for train/eval; recommended for predict)")
    parser.add_argument("--model", type=str, choices=SUPPORTED_MODELS, default="vgg16_scratch", help="Model type")

    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training/evaluation")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (scratch often better than 1e-3)")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for checkpoints")

    # Optional head params (si tu les ajoutes dans train.py / models.py)
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dim in ClassificationHead")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout in ClassificationHead")

    # Evaluation / Prediction
    parser.add_argument("--ckpt_path", type=str, help="Path to checkpoint for evaluation/prediction")
    parser.add_argument("--image_path", type=str, help="Path to image for prediction")

    return parser


def _default_ckpt_path(output_dir: str, modality: str | None, model: str) -> Path | None:
    """
    Try to find best checkpoint automatically.
    - If modality is provided => outputs/{modality}/{model}/best.pt
    - Else => try ct then xray.
    """
    out = Path(output_dir)

    if modality:
        p = out / modality / model / "best.pt"
        return p if p.exists() else None

    # fallback if modality omitted in predict
    for m in ["ct", "xray"]:
        p = out / m / model / "best.pt"
        if p.exists():
            return p
    return None


def train_all(args):
    """Train all models on specified modality."""
    if not args.modality:
        print("Error: --modality required for train_all action")
        return

    models_list = SUPPORTED_MODELS
    results = {}

    print(f"\n{'='*70}")
    print(f"TRAINING ALL MODELS ON {args.modality.upper()} IMAGES")
    print(f"{'='*70}")

    for model_name in models_list:
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
            output_dir=args.output_dir,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )

        try:
            _, best_acc = train(train_args)
            results[model_name] = best_acc
            print(f"✓ {model_name.upper()} training completed. Best accuracy: {best_acc:.2f}%")
        except Exception as e:
            print(f"✗ {model_name.upper()} training failed: {str(e)}")
            results[model_name] = None

    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    for model_name, acc in results.items():
        status = f"{acc:.2f}%" if acc is not None else "FAILED"
        print(f"{model_name.upper():14} : {status}")
    print(f"{'='*70}\n")

    return results


def eval_all(args):
    """Evaluate all models on specified modality."""
    if not args.modality:
        print("Error: --modality required for eval_all action")
        return

    models_list = SUPPORTED_MODELS
    results = {}

    print(f"\n{'='*70}")
    print(f"EVALUATING ALL MODELS ON {args.modality.upper()} IMAGES")
    print(f"{'='*70}")

    for model_name in models_list:
        print(f"\n{'─'*70}")
        print(f"Evaluating {model_name.upper()}...")
        print(f"{'─'*70}")

        ckpt_path = Path(args.output_dir) / args.modality / model_name / "best.pt"
        if not ckpt_path.exists():
            print(f"✗ Checkpoint not found: {ckpt_path}")
            results[model_name] = None
            continue

        eval_args = argparse.Namespace(
            ckpt_path=str(ckpt_path),
            data_root=args.data_root,
            modality=args.modality,
            model=model_name,
            batch_size=args.batch_size,
            img_size=args.img_size,
        )

        try:
            evaluate(eval_args)
            results[model_name] = "OK"
            print(f"✓ {model_name.upper()} evaluation completed")
        except Exception as e:
            print(f"✗ {model_name.upper()} evaluation failed: {str(e)}")
            results[model_name] = None

    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    for model_name, status in results.items():
        print(f"{model_name.upper():14} : {'COMPLETED' if status else 'FAILED'}")
    print(f"{'='*70}\n")

    return results


def run_predict(args):
    """Run prediction on a single image."""
    if not args.image_path:
        print("Error: --image_path required for predict action")
        return

    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"Error: Image not found at {args.image_path}")
        return

    if not args.ckpt_path:
        ckpt = _default_ckpt_path(args.output_dir, args.modality, args.model)
        if ckpt is None:
            print("Error: --ckpt_path required (or train first so best.pt exists).")
            return
        args.ckpt_path = str(ckpt)

    print(f"\n{'='*70}")
    print("PREDICTION")
    print(f"{'='*70}")

    predict_args = argparse.Namespace(
        ckpt_path=args.ckpt_path,
        image_path=args.image_path,
        model=args.model,
        img_size=args.img_size,
    )

    try:
        predict(predict_args)
        print("✓ Prediction completed")
    except Exception as e:
        print(f"✗ Prediction failed: {str(e)}")

    print(f"{'='*70}\n")


def full_pipeline(args):
    """Execute full pipeline: train all -> eval all -> predict with best model."""
    if not args.modality:
        print("Error: --modality required for full_pipeline action")
        return

    print(f"\n{'='*70}")
    print("FULL PIPELINE EXECUTION")
    print(f"{'='*70}")

    print("\n[STEP 1/3] Training all models...")
    train_results = train_all(args) or {}

    print("\n[STEP 2/3] Evaluating all models...")
    _ = eval_all(args)

    # pick best model for prediction (highest best_acc)
    best_model = None
    best_acc = -1.0
    for m, acc in train_results.items():
        if acc is not None and acc > best_acc:
            best_acc = acc
            best_model = m

    if args.image_path and best_model is not None:
        print(f"\n[STEP 3/3] Predicting with best model: {best_model} (val acc={best_acc:.2f}%)")
        args.model = best_model
        ckpt = _default_ckpt_path(args.output_dir, args.modality, args.model)
        if ckpt is not None:
            args.ckpt_path = str(ckpt)
            run_predict(args)
        else:
            print("Skipping prediction: checkpoint not found.")
    else:
        print("\n[STEP 3/3] Skipped (no --image_path provided OR training failed)")

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETED")
    print(f"{'='*70}\n")


def main():
    parser = create_parser()
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print("# COVID CLASSIFICATION PIPELINE")
    print(f"# Action: {args.action.upper()}")
    print(f"{'#'*70}")

    try:
        if args.action == "train":
            if not args.modality:
                print("Error: --modality required for train action")
                sys.exit(1)
            train(args)

        elif args.action == "eval":
            if not args.modality:
                print("Error: --modality required for eval action")
                sys.exit(1)

            if not args.ckpt_path:
                ckpt = _default_ckpt_path(args.output_dir, args.modality, args.model)
                if ckpt is None:
                    print("Error: --ckpt_path required (or train first so best.pt exists).")
                    sys.exit(1)
                args.ckpt_path = str(ckpt)

            evaluate(args)

        elif args.action == "predict":
            run_predict(args)

        elif args.action == "train_all":
            train_all(args)

        elif args.action == "eval_all":
            eval_all(args)

        elif args.action == "full_pipeline":
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


if __name__ == "__main__":
    main()
