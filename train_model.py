import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from tqdm import tqdm
import json
import torch.utils.data as data

from mot_data_loader import CreateMOTDataset, BoxClip
from build_det_graph import build_adj_graph
from head_gnn import BoxEmb
from config import *
from head_utils import get_tracklet_info
from TrackletData import TrackletData, TrackletDataset


# Function to train the tracklet graph model
def train_tracklet_graph_model(
    tracklet_graph_model, tracklet_data, optimizer, loss_weight, device
):
    tracklet_graph_model.train()
    total_loss = 0.0
    batch_loss_rec = 0.0
    batch_loss_trip = 0.0
    batch_loss_BCE = 0.0

    weight_rec, weight_trip, weight_BCE = loss_weight

    # 使用tqdm的正確方式來顯示進度
    progress_bar = tqdm(
        enumerate(tracklet_data),
        total=len(tracklet_data),
        desc="Training Tracklet Graph  ",
        leave=True,
        mininterval=0.1,  # 更新頻率設為0.1秒
    )

    for batch_idx, batch in progress_bar:
        optimizer.zero_grad()
        batch_start = torch.cuda.Event(enable_timing=True)
        batch_end = torch.cuda.Event(enable_timing=True)

        batch_start.record()

        # Move batch to device all at once
        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Forward pass using all GPUs
        loss_rec, loss_trip, loss_BCE = tracklet_graph_model(
            batch, device=device, stage="train"
        )

        # Calculate total loss
        loss = (
            weight_rec * sum([l.mean() for l in loss_rec])
            + weight_trip * sum([l.mean() for l in loss_trip])
            + weight_BCE * sum([l.mean() for l in loss_BCE])
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        batch_end.record()
        torch.cuda.synchronize()
        batch_time = batch_start.elapsed_time(batch_end)

        # Update statistics
        total_loss += loss.item()
        batch_loss_rec += sum([l.mean().item() for l in loss_rec])
        batch_loss_trip += sum([l.mean().item() for l in loss_trip])
        batch_loss_BCE += sum([l.mean().item() for l in loss_BCE])

        # Update progress bar with actual timing
        progress_bar.set_postfix(
            {
                "batch_time": f"{batch_time:.1f}ms",
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss/ (batch_idx + 1):.4f}",
            }
        )

    # Return total loss and individual losses
    return (
        total_loss / len(tracklet_data),
        batch_loss_rec / len(tracklet_data),
        batch_loss_trip / len(tracklet_data),
        batch_loss_BCE / len(tracklet_data),
    )


# Function to validate the tracklet graph model
def validate_tracklet_graph_model(tracklet_graph_model, val_data, loss_weight, device):
    tracklet_graph_model.eval()
    total_val_loss = 0.0
    val_loss_rec = 0.0
    val_loss_trip = 0.0
    val_loss_BCE = 0.0

    weight_rec, weight_trip, weight_BCE = loss_weight

    progress_bar = tqdm(
        enumerate(val_data),
        total=len(val_data),
        desc="Validating Tracklet Graph",
        leave=True,
        mininterval=0.1,
    )

    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            loss_rec, loss_trip, loss_BCE = tracklet_graph_model(
                batch, device=device, stage="train"
            )

            # Calculate batch losses
            batch_loss = (
                weight_rec * sum([l.mean() for l in loss_rec])
                + weight_trip * sum([l.mean() for l in loss_trip])
                + weight_BCE * sum([l.mean() for l in loss_BCE])
            )

            # Update statistics
            total_val_loss += batch_loss.item()
            val_loss_rec += sum([l.mean().item() for l in loss_rec])
            val_loss_trip += sum([l.mean().item() for l in loss_trip])
            val_loss_BCE += sum([l.mean().item() for l in loss_BCE])

            progress_bar.set_postfix(
                {
                    "val_loss": f"{batch_loss.item():.4f}",
                    "avg_val_loss": f"{total_val_loss / (batch_idx + 1):.4f}",
                }
            )

    return (
        total_val_loss / len(val_data),
        val_loss_rec / len(val_data),
        val_loss_trip / len(val_data),
        val_loss_BCE / len(val_data),
    )


# Main training function
def main():
    # Set up device and check GPU availability
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPUs!")
        device = torch.device("cuda")
        # Clear GPU cache
        torch.cuda.empty_cache()
    else:
        print("No GPU available, using CPU")
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Initialize tracklet graph model
    tracklet_graph_model = BoxEmb(tracklet_temporal_len + 1, device)

    # Wrap model with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        tracklet_graph_model = nn.DataParallel(tracklet_graph_model)
    tracklet_graph_model.to(device)
    print("Tracklet graph model initialized.")

    # Optimizer
    tracklet_optimizer = optim.AdamW(tracklet_graph_model.parameters(), lr=1e-4)

    # Load precomputed tracklet data
    tracklet_data_path = "precomputed_data/dance_motip_tracklet_data_T128_S5.pt"
    val_data_path = "precomputed_data/dance_motip_val_data_T128_S5.pt"
    print(f"Loading tracklet data from {tracklet_data_path}...")
    print(f"Loading tracklet data from {val_data_path}...")
    data = torch.load(tracklet_data_path, map_location="cpu")
    val_data = torch.load(val_data_path, map_location="cpu")
    dataset = TrackletDataset(data)
    val_dataset = TrackletDataset(val_data)
    print(f"Dataset loaded with attributes:", dataset.__dict__.keys())

    if not dataset:
        raise ValueError("No valid data found in the dataset")

    # Create DataLoader with batch size 4
    tracklet_data = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(8, 4 * torch.cuda.device_count()),  # 限制worker數量
        pin_memory=True,
        persistent_workers=True,  # 保持worker進程存活
        prefetch_factor=2,  # 預加載2個batch
        drop_last=True,
    )
    val_data = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(8, 4 * torch.cuda.device_count()),  # 限制worker數量
        pin_memory=True,
        persistent_workers=True,  # 保持worker進程存活
        prefetch_factor=2,  # 預加載2個batch
        drop_last=True,
    )

    print("Starting training...")
    num_epochs = 10
    best_val_loss = float("inf")
    loss_weight = [1.0, 5.0, 1.0]  # rec, triplet, BCE

    for epoch in range(num_epochs):
        try:
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Train phase
            train_loss, train_loss_rec, train_loss_trip, train_loss_BCE = (
                train_tracklet_graph_model(
                    tracklet_graph_model,
                    tracklet_data,
                    tracklet_optimizer,
                    loss_weight,
                    device,
                )
            )
            print(
                f"Training   - Loss Breakdown - Reconstruction Loss: {train_loss_rec:.6f}, "
                f"Triplet Loss: {train_loss_trip:.6f}, BCE Loss: {train_loss_BCE:.6f}"
            )
            print(f"Training   - Total Loss: {train_loss:.6f}")

            # Validation phase
            val_loss, val_loss_rec, val_loss_trip, val_loss_BCE = (
                validate_tracklet_graph_model(
                    tracklet_graph_model, val_data, loss_weight, device
                )
            )
            print(
                f"Validation - Loss Breakdown - Reconstruction Loss: {val_loss_rec:.6f}, "
                f"Triplet Loss: {val_loss_trip:.6f}, BCE Loss: {val_loss_BCE:.6f}"
            )
            print(f"Validation - Total Loss: {val_loss:.6f}")

            # Save model if it has the best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = "models/tracklet_graph_model_best.tar"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": tracklet_graph_model.state_dict(),
                        "optimizer_state_dict": tracklet_optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    save_path,
                )
                print("*" * 50)
                print(f"Saved best model with validation loss: {val_loss:.6f}")
                print("*" * 50)

            # Save regular checkpoint
            save_path = f"models/tracklet_graph_model_epoch_{epoch + 1}.tar"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": tracklet_graph_model.state_dict(),
                    "optimizer_state_dict": tracklet_optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                save_path,
            )

        except RuntimeError as e:
            if "out of memory" in str(e):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print("Out of memory error, cleaning up and continuing...")
                continue
            else:
                raise e


if __name__ == "__main__":
    main()
