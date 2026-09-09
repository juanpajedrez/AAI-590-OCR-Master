'''
Author: Juan Pablo Triana Martinez
Date: 2026-03-25
Script that trains a segmentation model for semantic PDF layout segmentation on DocLayNet data.
The architecture is selectable via --arch (defaults to linknet-resnet).

Usage example:
    python scripts/train_semantic_layout.py
    python scripts/train_semantic_layout.py --arch unet-resnet18
    python scripts/train_semantic_layout.py --arch fpn-resnet18 --epochs 10 --lr 5e-4
    python scripts/train_semantic_layout.py --arch bisenet-resnet18
    python scripts/train_semantic_layout.py --arch swiftnet-resnet18
    python scripts/train_semantic_layout.py --arch deeplabv3-mobilenetv2
    python scripts/train_semantic_layout.py --arch unet-mobilenetv2
    python scripts/train_semantic_layout.py --weight_ce 1.0 --weight_dice 0.5
    python scripts/train_semantic_layout.py --loss_fn dice --smooth 1e-7
    python scripts/train_semantic_layout.py --loss_fn cross-entropy
    python scripts/train_semantic_layout.py --dataset_name my_subsample --no_ignore_background
'''

import sys
import time
import random
from pathlib import Path
import argparse
import torch

# Allow imports from project root (src.*)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_dataloaders_text_detection
from src.models import build_model, ARCH_CHOICES
from src.utils import MetadataRetriever, benchmark_model, print_benchmark, save_benchmark
from src.training import LOSS_FN_CHOICES, get_loss_fn, train, create_writer, add_hparams_to_writer, save_model


def set_seeds(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across Python, PyTorch CPU and GPU."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":

    # ------------------------------------------------------------------ #
    #  Argument parser                                                     #
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(
        description="Train a segmentation model for semantic PDF layout segmentation on DocLayNet."
    )

    # --- Architecture arguments ---
    parser.add_argument("--arch",
                        type=str,
                        default="linknet-resnet",
                        choices=list(ARCH_CHOICES),
                        help="segmentation architecture to train (default: linknet-resnet)")

    # --- Data arguments ---
    parser.add_argument("-dp", "--data_path",
                        type=str,
                        default=str(Path().cwd() / "data"),
                        help="path to the data folder (default: ./data)")
    parser.add_argument("--dataset_name",
                        type=str,
                        default="google_collab_seed_86",
                        help="name of the dataset sub-folder inside data_path")

    # --- Dataloader arguments ---
    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        help="number of samples per batch (default: 8)")
    parser.add_argument("--new_height",
                        type=int,
                        default=512,
                        help="image height after resizing (default: 512)")
    parser.add_argument("--new_width",
                        type=int,
                        default=512,
                        help="image width after resizing (default: 512)")
    parser.add_argument("--num_workers",
                        type=int,
                        default=0,
                        help="dataloader worker processes (default: 0)")
    parser.add_argument("--pin_memory",
                        action="store_true",
                        default=False,
                        help="enable pinned memory in dataloaders (default: False)")

    # --- Augmentation arguments ---
    parser.add_argument("--jitter_brightness",
                        type=float,
                        default=0.2,
                        help="ColorJitter brightness factor for training transform (default: 0.2)")
    parser.add_argument("--jitter_contrast",
                        type=float,
                        default=0.2,
                        help="ColorJitter contrast factor for training transform (default: 0.2)")
    parser.add_argument("--jitter_saturation",
                        type=float,
                        default=0.2,
                        help="ColorJitter saturation factor for training transform (default: 0.2)")
    parser.add_argument("--jitter_hue",
                        type=float,
                        default=0.1,
                        help="ColorJitter hue factor for training transform (default: 0.1)")

    # --- Training arguments ---
    parser.add_argument("--epochs",
                        type=int,
                        default=5,
                        help="number of training epochs (default: 5)")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-3,
                        help="Adam optimizer learning rate (default: 1e-3)")
    parser.add_argument("--loss_fn",
                        type=str,
                        default="combined",
                        choices=list(LOSS_FN_CHOICES),
                        help="loss function to train against (default: combined). "
                             "'dice' uses DiceLoss (--smooth applies), "
                             "'cross-entropy' uses CrossEntropyLoss, "
                             "'combined' uses CE + Dice (--weight_ce / --weight_dice apply)")
    parser.add_argument("--smooth",
                        type=float,
                        default=1e-7,
                        help="smoothing factor for DiceLoss, used by --loss_fn dice and combined (default: 1e-7)")
    parser.add_argument("--weight_ce",
                        type=float,
                        default=1.0,
                        help="weight for CE loss in CombinedLoss, only used by --loss_fn combined (default: 1.0)")
    parser.add_argument("--weight_dice",
                        type=float,
                        default=0.5,
                        help="weight for Dice loss in CombinedLoss, only used by --loss_fn combined (default: 0.5)")
    parser.add_argument("--reduction",
                        type=str,
                        default="macro",
                        choices=["macro", "micro"],
                        help="metric averaging strategy (default: macro)")
    parser.add_argument("--no_ignore_background",
                        action="store_true",
                        default=False,
                        help="include background class (0) in loss and metrics (default: background is ignored)")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed for reproducibility (default: 42)")

    # --- TensorBoard arguments ---
    parser.add_argument("--experiment_name",
                        type=str,
                        default="DocLayNet_text_detection",
                        help="TensorBoard experiment label (default: DocLayNet_text_detection)")

    # --- Model saving arguments ---
    parser.add_argument("--target_dir",
                        type=str,
                        default="models",
                        help="directory to save the trained model (default: models)")
    parser.add_argument("--model_name",
                        type=str,
                        default=None,
                        help="filename for the saved model, must end in .pth or .pt "
                             "(default: <arch>_semantic_layout.pth)")

    # --- Benchmark arguments ---
    parser.add_argument("--benchmark_dir",
                        type=str,
                        default=str(Path("experiments") / "benchmarks"),
                        help="directory to save the benchmark JSON report "
                             "(default: experiments/benchmarks)")

    args = parser.parse_args()
    ignore_background = not args.no_ignore_background

    # Derive the model file name from the architecture when not provided
    if args.model_name is None:
        args.model_name = f"{args.arch.replace('-', '_')}_semantic_layout.pth"

    # ------------------------------------------------------------------ #
    #  Setup                                                               #
    # ------------------------------------------------------------------ #
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    data_path = Path(args.data_path)

    # ------------------------------------------------------------------ #
    #  Resolve number of classes from dataset metadata                    #
    # ------------------------------------------------------------------ #
    print("[INFO] Resolving number of semantic classes from dataset metadata...")
    metadata_retriever = MetadataRetriever(
        data_path=data_path,
        dataset_name=args.dataset_name,
        split_analyze="test"
    )
    supercategories = metadata_retriever.get_metadata_supercategories()
    num_classes = len(supercategories) + 1  # +1 for background class 0
    print(f"[INFO] Found {num_classes} classes (including background).")

    # ------------------------------------------------------------------ #
    #  Dataloaders                                                         #
    # ------------------------------------------------------------------ #
    print("[INFO] Loading semantic-layout dataloaders (auto-computing training mean/std)...")
    train_dl, val_dl, test_dl = get_dataloaders_text_detection(
        data_path=data_path,
        dataset_name=args.dataset_name,
        mask_type="semantic-layout",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        new_height=args.new_height,
        new_width=args.new_width,
        jitter_brightness=args.jitter_brightness,
        jitter_contrast=args.jitter_contrast,
        jitter_saturation=args.jitter_saturation,
        jitter_hue=args.jitter_hue,
    )

    # ------------------------------------------------------------------ #
    #  Model, loss, optimizer                                              #
    # ------------------------------------------------------------------ #
    print(f"[INFO] Building '{args.arch}' model (N={num_classes} for semantic segmentation)...")
    model = build_model(arch=args.arch, Cin=3, N=num_classes).to(device)

    print(f"[INFO] Using '{args.loss_fn}' loss function...")
    loss_fn = get_loss_fn(
        loss_name=args.loss_fn,
        binary=False,
        smooth=args.smooth,
        weight_ce=args.weight_ce,
        weight_dice=args.weight_dice,
        ignore_background=ignore_background,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------ #
    #  TensorBoard writer                                                  #
    # ------------------------------------------------------------------ #
    writer = create_writer(
        experiment_name=args.experiment_name,
        model_name=args.arch,
        extra=f"semantic_{args.loss_fn}"
    )

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #
    print(f"[INFO] Starting semantic training for {args.epochs} epoch(s)...")
    train_start = time.perf_counter()
    results = train(
        model=model,
        train_dataloader=train_dl,
        test_dataloader=val_dl,
        optimizer=optimizer,
        loss_fn=loss_fn,
        simulate_batch_size=args.batch_size,
        simulate_new_channels=3,
        simulate_new_height=args.new_height,
        simulate_new_width=args.new_width,
        epochs=args.epochs,
        device=device,
        passed_writer=writer,
        binary=False,
        ignore_background=ignore_background,
        reduction=args.reduction,
    )
    training_time_s = time.perf_counter() - train_start
    print(f"[INFO] Total training time: {training_time_s:.1f} s "
          f"({training_time_s / args.epochs:.1f} s/epoch)")

    # ------------------------------------------------------------------ #
    #  Efficiency benchmark (params, FLOPs, latency, memory)               #
    # ------------------------------------------------------------------ #
    print("[INFO] Running efficiency benchmark...")
    report = benchmark_model(
        model=model,
        input_size=(1, 3, args.new_height, args.new_width),
        device=device,
        arch_name=args.arch,
        training_time_s=training_time_s,
        epochs=args.epochs,
    )
    print_benchmark(report)
    save_benchmark(
        report=report,
        target_dir=args.benchmark_dir,
        file_name=f"{args.arch.replace('-', '_')}_semantic_{args.loss_fn}.json",
    )

    # ------------------------------------------------------------------ #
    #  Log hyperparameters                                                 #
    # ------------------------------------------------------------------ #
    add_hparams_to_writer(
        writer=writer,
        batch_size=args.batch_size,
        new_height=args.new_height,
        new_width=args.new_width,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        results=results,
    )

    # ------------------------------------------------------------------ #
    #  Save model                                                          #
    # ------------------------------------------------------------------ #
    save_model(
        model=model,
        target_dir=args.target_dir,
        model_name=args.model_name,
    )

    print("[INFO] Semantic PDF layout segmentation training complete.")
