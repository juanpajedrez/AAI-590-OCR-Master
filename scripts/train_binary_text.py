'''
Author: Juan Pablo Triana Martinez
Date: 2026-03-25
Script that trains a LinkNet model for binary text segmentation on DocLayNet data.

Usage example:
    python scripts/train_binary_text.py
    python scripts/train_binary_text.py --epochs 10 --lr 5e-4 --batch_size 16
    python scripts/train_binary_text.py --dataset_name my_subsample --epochs 20 --reduction micro
'''

import sys
import random
from pathlib import Path
import argparse
import torch

# Allow imports from project root (src.*)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_dataloaders_text_detection
from src.models import LinknetModel
from src.training import DiceLoss, train, create_writer, add_hparams_to_writer, save_model


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
        description="Train a LinkNet model for binary text segmentation on DocLayNet."
    )

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

    # --- Training arguments ---
    parser.add_argument("--epochs",
                        type=int,
                        default=5,
                        help="number of training epochs (default: 5)")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-3,
                        help="Adam optimizer learning rate (default: 1e-3)")
    parser.add_argument("--smooth",
                        type=float,
                        default=1e-7,
                        help="smoothing factor for DiceLoss (default: 1e-7)")
    parser.add_argument("--reduction",
                        type=str,
                        default="macro",
                        choices=["macro", "micro"],
                        help="metric averaging strategy (default: macro)")
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
                        default="linknet_binary_text.pth",
                        help="filename for the saved model, must end in .pth or .pt (default: linknet_binary_text.pth)")

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    #  Setup                                                               #
    # ------------------------------------------------------------------ #
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    data_path = Path(args.data_path)

    # ------------------------------------------------------------------ #
    #  Dataloaders                                                         #
    # ------------------------------------------------------------------ #
    print("[INFO] Loading binary-text dataloaders...")
    train_dl, val_dl, test_dl = get_dataloaders_text_detection(
        data_path=data_path,
        dataset_name=args.dataset_name,
        mask_type="binary-text",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        new_height=args.new_height,
        new_width=args.new_width,
        transform=None,
    )

    # ------------------------------------------------------------------ #
    #  Model, loss, optimizer                                              #
    # ------------------------------------------------------------------ #
    print("[INFO] Building LinkNet model (N=1 for binary segmentation)...")
    model = LinknetModel(Cin=3, N=1).to(device)

    loss_fn = DiceLoss(smooth=args.smooth, ignore_background=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------ #
    #  TensorBoard writer                                                  #
    # ------------------------------------------------------------------ #
    writer = create_writer(
        experiment_name=args.experiment_name,
        model_name="LinkNet",
        extra="binary"
    )

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #
    print(f"[INFO] Starting binary training for {args.epochs} epoch(s)...")
    results = train(
        model=model,
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        optimizer=optimizer,
        loss_fn=loss_fn,
        simulate_batch_size=args.batch_size,
        simulate_new_channels=1,
        simulate_new_height=args.new_height,
        simulate_new_width=args.new_width,
        epochs=args.epochs,
        device=device,
        passed_writer=writer,
        binary=True,
        ignore_background=False,
        reduction=args.reduction,
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

    print("[INFO] Binary text segmentation training complete.")
