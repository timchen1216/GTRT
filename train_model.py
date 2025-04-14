import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from tqdm import tqdm

from mot_data_loader import CreateMOTDataset, BoxClip
from build_det_graph import build_adj_graph
from head_gnn import BoxEmb
from config import *
from head_utils import get_tracklet_info


# Function to train the tracklet graph model
def train_tracklet_graph_model(
    tracklet_graph_model, dataloader, gt_dataloader, optimizer, device
):
    tracklet_graph_model.train()
    total_loss = 0.0
    epoch_loss_rec = 0.0
    epoch_loss_trip = 0.0
    epoch_loss_BCE = 0.0

    for batch, gt_batch in tqdm(
        zip(dataloader, gt_dataloader), desc="Training Tracklet Graph"
    ):
        optimizer.zero_grad()

        # Prepare data
        batch_fr_ids = []
        batch_bbox = []
        batch_obj_ids = []
        batch_scores = []
        gt_batch_bbox = []  # To store ground truth bbox
        for i in range(len(batch["boxes"])):
            fr_ids = batch["fr_ids"][i].to(device).float()
            bbox = batch["boxes"][i].to(device).float()
            bbox[:, 0::2] = bbox[:, 0::2] / float(batch["width"][i].item())
            bbox[:, 1::2] = bbox[:, 1::2] / float(batch["height"][i].item())
            obj_ids = batch["obj_ids"][i].to(device)
            scores = torch.ones(len(bbox), device=device)

            # Normalize ground truth bbox
            gt_bbox = gt_batch["boxes"][i].to(device).float()
            gt_bbox[:, 0::2] = gt_bbox[:, 0::2] / float(gt_batch["width"][i].item())
            gt_bbox[:, 1::2] = gt_bbox[:, 1::2] / float(gt_batch["height"][i].item())

            batch_fr_ids.append(fr_ids)
            batch_bbox.append(bbox)
            batch_obj_ids.append(obj_ids)
            batch_scores.append(scores)
            gt_batch_bbox.append(gt_bbox)

        # Concatenate batch data
        fr_ids = torch.cat(batch_fr_ids, dim=0)
        bbox = torch.cat(batch_bbox, dim=0)
        obj_ids = torch.cat(batch_obj_ids, dim=0)
        scores = torch.cat(batch_scores, dim=0)
        gt_bbox = torch.cat(gt_batch_bbox, dim=0)  # Concatenate ground truth bbox

        # Build adjacency graph
        edge_idx, A = build_adj_graph(fr_ids, bbox)

        # Generate tracklet information
        tracklet_info = get_tracklet_info(
            tracklet_label=obj_ids,  # Use obj_ids as tracklet labels
            obj_ids=obj_ids,
            fr_ids=fr_ids,
            det_embs=bbox,
            gt_embs=gt_bbox,  # Use ground truth bbox
            scores=scores,
            temporal_len=tracklet_temporal_len,
            device=device,
            stage="train",
        )

        # Assign tracklet labels and additional data
        tracklet_data = {
            "tracklet_embs": tracklet_info["tracklet_embs"],
            "tracklet_scores": tracklet_info["tracklet_scores"],
            "A": tracklet_info["A"],
            "edge_idx": tracklet_info["edge_idx"],
            "tracklet_labels": tracklet_info["tracklet_labels"],  # Ground truth labels
            "gt_data": tracklet_info["tracklet_gt_embs"],  # Ground truth data
            "gt_vis_mask": tracklet_info["tracklet_scores"],  # Visibility mask
            "binary_label": tracklet_info["binary_label"],  # Binary labels
        }

        # Forward pass
        loss_rec, loss_trip, loss_BCE = tracklet_graph_model(
            tracklet_data, stage="train"
        )
        loss = sum(loss_rec) + sum(loss_trip) + sum(loss_BCE)

        # Accumulate individual losses
        epoch_loss_rec += sum(loss_rec).item()
        epoch_loss_trip += sum(loss_trip).item()
        epoch_loss_BCE += sum(loss_BCE).item()

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Return total loss and individual losses
    return (
        total_loss / len(dataloader),
        epoch_loss_rec / len(dataloader),
        epoch_loss_trip / len(dataloader),
        epoch_loss_BCE / len(dataloader),
    )


# Custom collate function to handle variable-sized data
def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0]:
        if key in ["boxes", "fr_ids", "obj_ids", "classes"]:  # Variable-sized fields
            collated_batch[key] = [torch.tensor(item[key]) for item in batch]
        elif key in ["img_paths"]:  # String fields
            collated_batch[key] = [item[key] for item in batch]
        else:  # Fixed-size fields
            collated_batch[key] = torch.tensor([item[key] for item in batch])
    return collated_batch


# Main training function
def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize tracklet graph model
    tracklet_graph_model = BoxEmb(tracklet_temporal_len + 1, device).to(device)

    # Optimizer
    tracklet_optimizer = optim.Adam(tracklet_graph_model.parameters(), lr=1e-4)

    # Data loader
    comp_transforms = transforms.Compose([BoxClip()])
    mot_data = CreateMOTDataset(
        data_path=train_data_path,
        temporal_len=tracklet_temporal_len,
        transform=comp_transforms,
    )
    gt_data = CreateMOTDataset(
        data_path=gt_path, temporal_len=tracklet_temporal_len, transform=comp_transforms
    )

    # Use a shared random sampler for both dataloaders
    shared_sampler = RandomSampler(mot_data)

    dataloader = DataLoader(
        mot_data,
        batch_size=batch_size,
        # sampler=shared_sampler,  # Use shared sampler
        num_workers=4,
        collate_fn=custom_collate_fn,
    )
    gt_dataloader = DataLoader(
        gt_data,
        batch_size=batch_size,
        # sampler=shared_sampler,  # Use shared sampler
        num_workers=4,
        collate_fn=custom_collate_fn,
    )

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train tracklet graph model
        tracklet_loss, loss_rec, loss_trip, loss_BCE = train_tracklet_graph_model(
            tracklet_graph_model, dataloader, gt_dataloader, tracklet_optimizer, device
        )
        print(
            f"Loss Breakdown - Reconstruction Loss: {loss_rec:.6f}, Triplet Loss: {loss_trip:.6f}, BCE Loss: {loss_BCE:.6f}"
        )
        print(f"Total Loss: {tracklet_loss:.6f}")

        # Save model
        torch.save(
            {"model_state_dict": tracklet_graph_model.state_dict()},
            f"models/tracklet_graph_model_epoch_{epoch + 1}.tar",
        )


if __name__ == "__main__":
    main()
