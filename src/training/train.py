'''
Author: Juan Pablo Triana Martinez
Date: 2026-03-24
The following contains the PyTorch train functions related with the following:
    - linknet-resnet -> text_detection (binary and semantic).
'''

import torch
from typing import Tuple, Dict, List
from collections import defaultdict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from src.utils import get_binary_metrics, get_semantic_metrics, get_soft_metrics


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               binary: bool = True,
               ignore_background: bool = False,
               reduction: str = "macro") -> Tuple[float, Dict[str, float]]:
    """Trains a PyTorch model for a single epoch.

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        binary: If True, runs binary segmentation metrics; else multiclass.
        ignore_background: If True, skips class 0 in semantic metric computation.
        reduction: "macro" or "micro" averaging for semantic metrics.

    Returns:
        Tuple of (avg_loss, avg_metrics_dict).
    """
    assert reduction in ["macro", "micro"], "Reduction must be either 'macro' or 'micro'."

    model.train()

    train_loss = 0.0
    total_metrics: Dict[str, float] = defaultdict(float)
    num_batches = 0

    for batch, (X_imgs, X_masks, X_metadata) in enumerate(dataloader):
        X_imgs = X_imgs.to(device)
        X_masks = X_masks.to(device)

        # Forward pass
        X_logits = model(X_imgs)

        # Compute loss
        loss = loss_fn(X_logits, X_masks)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metric computation (no gradient tracking)
        with torch.no_grad():
            if binary:
                pixel_metrics = get_binary_metrics(X_logits, X_masks)
                region_metrics = get_soft_metrics(X_logits, X_masks, is_binary=True)
            else:
                pixel_metrics = get_semantic_metrics(
                    logits=X_logits,
                    targets=X_masks,
                    num_classes=X_logits.shape[1],
                    ignore_background=ignore_background,
                    reduction=reduction
                )
                region_metrics = get_soft_metrics(X_logits, X_masks, is_binary=False)

            for k, v in pixel_metrics.items():
                total_metrics[k] += float(v)
            for k, v in region_metrics.items():
                total_metrics[k] += float(v.mean())

        num_batches += 1

    avg_loss = train_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    return avg_loss, avg_metrics


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device,
              binary: bool = True,
              ignore_background: bool = False,
              reduction: str = "macro") -> Tuple[float, Dict[str, float]]:
    """Tests a PyTorch model for a single epoch.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        binary: If True, runs binary segmentation metrics; else multiclass.
        ignore_background: If True, skips class 0 in semantic metric computation.
        reduction: "macro" or "micro" averaging for semantic metrics.

    Returns:
        Tuple of (avg_loss, avg_metrics_dict).
    """
    assert reduction in ["macro", "micro"], "Reduction must be either 'macro' or 'micro'."

    model.eval()

    test_loss = 0.0
    total_metrics: Dict[str, float] = defaultdict(float)
    num_batches = 0

    with torch.inference_mode():
        for batch, (X_imgs, X_masks, X_metadata) in enumerate(dataloader):
            X_imgs = X_imgs.to(device)
            X_masks = X_masks.to(device)

            X_logits = model(X_imgs)

            loss = loss_fn(X_logits, X_masks)
            test_loss += loss.item()

            if binary:
                pixel_metrics = get_binary_metrics(X_logits, X_masks)
                region_metrics = get_soft_metrics(X_logits, X_masks, is_binary=True)
            else:
                pixel_metrics = get_semantic_metrics(
                    logits=X_logits,
                    targets=X_masks,
                    num_classes=X_logits.shape[1],
                    ignore_background=ignore_background,
                    reduction=reduction
                )
                region_metrics = get_soft_metrics(X_logits, X_masks, is_binary=False)

            for k, v in pixel_metrics.items():
                total_metrics[k] += float(v)
            for k, v in region_metrics.items():
                total_metrics[k] += float(v.mean())

            num_batches += 1

    avg_loss = test_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    return avg_loss, avg_metrics


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          simulate_batch_size: int,
          simulate_new_channels: int,
          simulate_new_height: int,
          simulate_new_width: int,
          epochs: int,
          device: torch.device,
          passed_writer: SummaryWriter,
          binary: bool = True,
          ignore_background: bool = False,
          reduction: str = "macro") -> Dict[str, List]:
    """Trains and tests a PyTorch model for a number of epochs.

    Passes the model through train_step() and test_step() each epoch,
    logging loss and all pixel/region metrics to TensorBoard.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: DataLoader for training data.
        test_dataloader: DataLoader for test/validation data.
        optimizer: A PyTorch optimizer.
        loss_fn: A PyTorch loss function.
        simulate_batch_size: Batch size used for add_graph in TensorBoard.
        simulate_new_channels: Channels used for add_graph (1=binary, C=semantic).
        simulate_new_height: Height used for add_graph.
        simulate_new_width: Width used for add_graph.
        epochs: Number of training epochs.
        device: Target compute device.
        passed_writer: A TensorBoard SummaryWriter instance (or None).
        binary: If True, binary segmentation mode; else multiclass.
        ignore_background: If True, skips class 0 in semantic metrics.
        reduction: "macro" or "micro" averaging for semantic metrics.

    Returns:
        Dictionary mapping metric names to lists of per-epoch values.
    """
    results: Dict[str, List] = defaultdict(list)
    model.to(device)

    # Log model graph once before training
    if passed_writer:
        passed_writer.add_graph(
            model=model,
            input_to_model=torch.randn(
                simulate_batch_size, simulate_new_channels,
                simulate_new_height, simulate_new_width
            ).to(device))

    for epoch in tqdm(range(epochs)):
        train_loss, train_metrics = train_step(
            model=model, dataloader=train_dataloader, loss_fn=loss_fn,
            optimizer=optimizer, device=device, binary=binary,
            ignore_background=ignore_background, reduction=reduction)

        results["train_loss"].append(train_loss)
        for k, v in train_metrics.items():
            results[k + "_train"].append(v)

        test_loss, test_metrics = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn,
            device=device, binary=binary, ignore_background=ignore_background,
            reduction=reduction)

        results["test_loss"].append(test_loss)
        for k, v in test_metrics.items():
            results[k + "_test"].append(v)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"iou_pixel_train: {results['iou_pixel_train'][-1]:.4f} | "
            f"dice_pixel_train: {results['dice_pixel_train'][-1]:.4f} | "
            f"IoU_region_train: {results['IoU_region_train'][-1]:.4f} | "
            f"DsC_region_train: {results['DsC_region_train'][-1]:.4f}"
        )
        print(
            f"Epoch: {epoch+1} | "
            f"test_loss: {test_loss:.4f} | "
            f"iou_pixel_test: {results['iou_pixel_test'][-1]:.4f} | "
            f"dice_pixel_test: {results['dice_pixel_test'][-1]:.4f} | "
            f"IoU_region_test: {results['IoU_region_test'][-1]:.4f} | "
            f"DsC_region_test: {results['DsC_region_test'][-1]:.4f}"
        )

        if passed_writer:
            passed_writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
                global_step=epoch)
            for k in train_metrics.keys():
                passed_writer.add_scalars(
                    main_tag=k,
                    tag_scalar_dict={"train": train_metrics[k], "test": test_metrics[k]},
                    global_step=epoch)

    if passed_writer:
        passed_writer.close()

    return results


def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str = None) -> SummaryWriter:
    """Creates a SummaryWriter saving to a timestamped log directory.

    Log directory format: runs/<YYYY-MM-DD>/<experiment_name>/<model_name>[/<extra>]

    Args:
        experiment_name: Top-level experiment label (e.g. "DocLayNet_text_detection").
        model_name: Model label (e.g. "LinkNet").
        extra: Optional sub-label (e.g. "binary" or "semantic").

    Returns:
        A configured SummaryWriter instance.
    """
    from datetime import datetime
    import os

    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter saving to {log_dir}")
    return SummaryWriter(log_dir=log_dir)


def add_hparams_to_writer(writer: SummaryWriter,
                          batch_size: int,
                          new_height: int,
                          new_width: int,
                          num_workers: int,
                          pin_memory: bool,
                          results: Dict[str, List]) -> None:
    """Logs hyperparameters and final-epoch metrics to TensorBoard's HParams tab.

    Opens a new SummaryWriter at the same log_dir as the passed writer
    (which was already closed by train()), records all hparams alongside
    the last value of every metric in results, then closes.

    Args:
        writer:      The SummaryWriter used during training (log_dir is read from it).
        batch_size:  Batch size used by the dataloaders.
        new_height:  Image height passed to the dataloaders.
        new_width:   Image width passed to the dataloaders.
        num_workers: Number of dataloader workers.
        pin_memory:  Whether pinned memory was enabled.
        results:     The dict returned by train() — keys map to lists of per-epoch values.
    """
    hparam_dict = {
        "batch_size": batch_size,
        "new_height": new_height,
        "new_width": new_width,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    metric_dict = {f"final_{k}": float(v[-1]) for k, v in results.items()}

    hparam_writer = SummaryWriter(log_dir=writer.log_dir)
    hparam_writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
    hparam_writer.close()
    print(f"[INFO] Hyperparameters logged to {writer.log_dir}")


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)