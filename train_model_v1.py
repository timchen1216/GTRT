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
from head_gnn_v1 import BoxEmb
from config import *
from head_utils import get_tracklet_info
from TrackletData import TrackletData, TrackletDataset


# Function to train the tracklet graph model
def train_tracklet_graph_model(tracklet_graph_model, tracklet_data, optimizer, device):
    tracklet_graph_model.train()
    total_loss = 0.0
    batch_loss_rec = 0.0
    batch_loss_trip = 0.0
    batch_loss_BCE = 0.0

    for batch in tqdm(
        tracklet_data,
        desc="Training Tracklet Graph",
        total=len(tracklet_data),
    ):
        optimizer.zero_grad()
        batch_size = batch["tracklet_embs"].size(0)
        total_batch_loss = 0

        # Process one sample at a time
        for i in range(batch_size):
            # Create a single sample dictionary
            single_sample = {
                k: (
                    v[i : i + 1].squeeze(0).to(device)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in batch.items()
            }
            # for k, v in single_sample.items():
            #     print(
            #         f"Key: {k}, Shape: {v.shape if isinstance(v, torch.Tensor) else v}"
            #     )

            # Forward pass with single sample
            loss_rec, loss_trip, loss_BCE = tracklet_graph_model(
                single_sample, device=device, stage="train"
            )
            loss = sum(loss_rec) + sum(loss_trip) + sum(loss_BCE)
            total_batch_loss += loss

            # Accumulate individual losses
            batch_loss_rec += sum(loss_rec).item()
            batch_loss_trip += sum(loss_trip).item()
            batch_loss_BCE += sum(loss_BCE).item()

        # Backward pass for accumulated loss
        total_batch_loss.backward()
        optimizer.step()
        total_loss += total_batch_loss.item()

    # Return total loss and individual losses
    return (
        total_loss / len(tracklet_data),
        batch_loss_rec / len(tracklet_data),
        batch_loss_trip / len(tracklet_data),
        batch_loss_BCE / len(tracklet_data),
    )


# Main training function
def main():
    # Set up device with error handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Clear GPU cache
        torch.cuda.empty_cache()
        # Check CUDA initialization
        try:
            torch.cuda.init()
        except RuntimeError as e:
            print(f"CUDA initialization error: {e}")
            print("Falling back to CPU...")
            device = torch.device("cpu")

    print(f"Using device: {device}")

    # Initialize tracklet graph model
    tracklet_graph_model = BoxEmb(tracklet_temporal_len + 1, device)
    # tracklet_graph_model = nn.DataParallel(tracklet_graph_model)
    tracklet_graph_model.to(device)
    print("Tracklet graph model initialized.")

    # Optimizer
    tracklet_optimizer = optim.AdamW(tracklet_graph_model.parameters(), lr=1e-4)

    # Load precomputed tracklet data
    # tracklet_data_path = "precomputed_data/tracklet_data.pt"
    tracklet_data_path = "precomputed_data/dance_motip_tracklet_data.pt"
    print(f"Loading tracklet data from {tracklet_data_path}...")
    data = torch.load(tracklet_data_path, map_location="cpu")
    dataset = TrackletDataset(data)

    print(f"Dataset loaded with attributes:", dataset.__dict__.keys())

    if not dataset:
        raise ValueError("No valid data found in the dataset")

    # Create DataLoader
    tracklet_data = DataLoader(
        dataset,
        batch_size=batch_size,  # 可以調整為更大的batch size
        shuffle=False,
        num_workers=4,
        collate_fn=None,  # 確保默認的collate_fn能正確處理數據
    )

    print("Starting training...")
    num_epochs = 100
    for epoch in range(num_epochs):
        try:
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Train tracklet graph model
            tracklet_loss, loss_rec, loss_trip, loss_BCE = train_tracklet_graph_model(
                tracklet_graph_model, tracklet_data, tracklet_optimizer, device
            )
            print(
                f"Loss Breakdown - Reconstruction Loss: {loss_rec:.6f}, "
                f"Triplet Loss: {loss_trip:.6f}, BCE Loss: {loss_BCE:.6f}"
            )
            print(f"Total Loss: {tracklet_loss:.6f}")

            # Save model
            save_path = f"models/tracklet_graph_model_epoch_{epoch + 1}.tar"
            torch.save(
                {"model_state_dict": tracklet_graph_model.state_dict()},
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
