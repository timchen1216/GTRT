import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

import cv2
import os
import time

from mot_data_loader import CreateMOTDataset, HFlip, BoxShift, BoxClip, BoxJitter, AddFP
import build_det_graph
import head_utils, head_gnn

from config import *
from tqdm import tqdm


def det_graph_predictor(node_input, model):

    # build graph
    edge_idx, A = build_det_graph.build_adj_graph(
        node_input["fr_ids"], node_input["det_embs"]
    )

    # inference
    with torch.no_grad():
        graph_emb, _ = model(node_input["det_embs"], A, edge_idx, None)

        # get dist of embs
        tmp_idx = torch.nonzero(edge_idx[0, :] < edge_idx[1, :])
        transf_edge_idx = edge_idx[:, tmp_idx[:, 0]]
        dist = torch.norm(
            graph_emb[transf_edge_idx[0, :], :] - graph_emb[transf_edge_idx[1, :], :],
            dim=1,
        )

    return dist


def batch_test_tracklet_graph(batch_data, tracklet_graph_model):
    batch_updated_tracklet_data = []

    for b in range(len(batch_data)):
        # 準備輸入資料
        tracklet_data = {
            "tracklet_embs": batch_data[b]["tracklet_embs"].to(device),
            "tracklet_scores": batch_data[b]["tracklet_scores"].to(device),
            "A": batch_data[b]["A"].to(device),
            "edge_idx": batch_data[b]["edge_idx"].to(device),
        }

        # 如果有 ground truth 資料，則加入
        if "tracklet_labels" in batch_data[b]:
            tracklet_data["tracklet_labels"] = batch_data[b]["tracklet_labels"].to(
                device
            )
            tracklet_data["gt_data"] = batch_data[b]["gt_data"].to(device)
            tracklet_data["gt_vis_mask"] = batch_data[b]["gt_vis_mask"].to(device)
            tracklet_data["binary_label"] = batch_data[b]["binary_label"].to(device)

        # 使用模型進行推論
        with torch.no_grad():
            updated_tracklet_embs, updated_tracklet_scores = tracklet_graph_model(
                tracklet_data, stage="test"
            )

        # 更新軌跡資料
        updated_tracklet_data = {
            "updated_tracklet_embs": updated_tracklet_embs,
            "updated_tracklet_scores": updated_tracklet_scores,
        }
        batch_updated_tracklet_data.append(updated_tracklet_data)

    return batch_updated_tracklet_data


def batch_test_det_graph(batch_data, det_graph_model):

    batch_pred_tracklet_label = []

    for b in range(len(batch_data)):

        # convert to cuda
        fr_ids = batch_data[b]["fr_ids"][0].to(device).float()
        bbox = batch_data[b]["boxes"][0].to(device).float()
        bbox = bbox.float()
        bbox[:, 0::2] = bbox[:, 0::2] / float(batch_data[b]["width"].item())
        bbox[:, 1::2] = bbox[:, 1::2] / float(batch_data[b]["height"].item())
        scores = torch.ones(len(bbox), device=device)

        # graph stat
        N_node = len(bbox)
        fr_ids1 = torch.unsqueeze(fr_ids, 1)
        fr_ids2 = torch.unsqueeze(fr_ids, 0)
        delta_fr_ids = fr_ids1.cpu() - fr_ids2.cpu()
        det_edge_idx = torch.nonzero(delta_fr_ids == -1).to(device)
        agg_dist = torch.zeros(len(det_edge_idx), device=device)
        agg_cnt = torch.zeros(len(det_edge_idx), device=device)

        max_fr = torch.max(fr_ids).item()
        t_fr = 0
        end_flag = 0

        while True:
            if t_fr + T_det_window <= max_fr + 1:
                st_fr = t_fr
                end_fr = t_fr + T_det_window
            else:
                st_fr = max_fr + 1 - T_det_window
                end_fr = max_fr + 1
                end_flag = 1

            cand_ids = torch.nonzero((fr_ids >= st_fr) * (fr_ids < end_fr))
            tmp_det_embs = bbox[cand_ids[:, 0]]
            tmp_fr_ids = fr_ids[cand_ids[:, 0]] - st_fr

            node_input = {"det_embs": tmp_det_embs, "fr_ids": tmp_fr_ids}

            tmp_dist = det_graph_predictor(node_input, det_graph_model)

            tmp_edge_st_idx = torch.nonzero(
                (det_edge_idx[:, 0] >= cand_ids[0, 0])
                * (det_edge_idx[:, 0] <= cand_ids[-1, 0])
            )[0, 0]
            agg_dist[tmp_edge_st_idx : tmp_edge_st_idx + len(tmp_dist)] += tmp_dist
            agg_cnt[tmp_edge_st_idx : tmp_edge_st_idx + len(tmp_dist)] += tmp_dist

            t_fr += T_det_stride
            if end_flag == 1:
                break

        avg_dist = agg_dist / agg_cnt
        pred_edge_label = head_utils.get_pred_edge_label(
            len(fr_ids), avg_dist, det_edge_idx, emb_dist_thresh, device
        )
        tracklet_label = head_utils.get_tracklet_label(
            len(fr_ids), pred_edge_label, det_edge_idx
        )
        batch_pred_tracklet_label.append(tracklet_label)

    return batch_pred_tracklet_label


def test_tracklet_graph():
    # color table
    color_table = np.random.rand(5000, 3)

    # tracklet graph model
    tracklet_graph_model = head_gnn.BoxEmb(tracklet_temporal_len + 1, device)
    tracklet_graph_model = tracklet_graph_model.to(device)
    tracklet_graph_checkpoint = torch.load(
        tracklet_graph_model_load_path, map_location=device
    )
    tracklet_graph_model.load_state_dict(tracklet_graph_checkpoint["model_state_dict"])
    tracklet_graph_model.eval()

    # data loader
    comp_transforms = transforms.Compose([BoxClip()])
    mot_data = CreateMOTDataset(train_data_path, -1, transform=comp_transforms)
    dataloader = DataLoader(mot_data, batch_size=1, shuffle=False, num_workers=4)

    # Create output directories if they don't exist
    if save_img and not os.path.exists(save_tracklet_graph_img_dir):
        os.makedirs(save_tracklet_graph_img_dir, exist_ok=True)

    if save_txt and not os.path.exists(save_txt_dir):
        os.makedirs(save_txt_dir, exist_ok=True)

    # prediction
    batch_data = []
    for i_batch, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Prepare data for tracklet graph
        fr_ids = sample["fr_ids"][0].to(device).float()
        bbox = sample["boxes"][0].to(device).float()
        bbox[:, 0::2] = bbox[:, 0::2] / float(sample["width"].item())
        bbox[:, 1::2] = bbox[:, 1::2] / float(sample["height"].item())
        scores = torch.ones(len(bbox), device=device)

        # Build adjacency graph
        edge_idx, A = build_det_graph.build_adj_graph(fr_ids, bbox)

        # Generate tracklet information
        tracklet_info = head_utils.get_tracklet_info(
            det_ids=torch.arange(
                len(fr_ids), device=device
            ),  # Dummy labels for testing
            gt_ids=torch.zeros(len(fr_ids), device=device),  # Dummy object IDs
            fr_ids=fr_ids,
            det_bboxes=bbox,
            gt_bboxes=bbox,  # Assuming bbox is used as ground truth embeddings
            scores=scores,
            temporal_len=tracklet_temporal_len,
            device=device,
            stage="test",
        )

        # Add tracklet information to batch_data
        batch_data.append(
            {
                "tracklet_embs": tracklet_info["tracklet_embs"],
                "tracklet_scores": tracklet_info["tracklet_scores"],
                "A": tracklet_info["A"],
                "edge_idx": tracklet_info["edge_idx"],
            }
        )

        # Process batch if it reaches the batch size
        if len(batch_data) == batch_size:
            batch_updated_tracklet_data = batch_test_tracklet_graph(
                batch_data, tracklet_graph_model
            )
            batch_data = []  # Reset batch data

            # 處理更新後的軌跡資料
            for updated_data in batch_updated_tracklet_data:
                updated_tracklet_embs = updated_data["updated_tracklet_embs"]
                updated_tracklet_scores = updated_data["updated_tracklet_scores"]
                # 在這裡可以進一步處理更新後的軌跡資料，例如儲存或可視化

            # Get sequence name from image path
            tmp_str = sample["img_paths"][0][0].split("/")

            # Setup path for different datasets
            if dataset == "KITTI":
                if sub_class == "car":
                    seq_path = os.path.join(
                        save_tracklet_graph_img_dir, "car", tmp_str[-2]
                    )
                elif sub_class == "person":
                    seq_path = os.path.join(
                        save_tracklet_graph_img_dir, "person", tmp_str[-2]
                    )
            elif dataset == "MOT":
                seq_path = os.path.join(save_tracklet_graph_img_dir, tmp_str[-3])
            elif dataset == "Dance":
                # For DanceTrack, extract sequence name from path (e.g. dancetrack0003)
                seq_name = tmp_str[
                    -3
                ]  # assuming format like /path/to/dancetrack0003/img1/00000001.jpg
                seq_path = os.path.join(save_tracklet_graph_img_dir, seq_name)
            elif dataset == "UADETRAC":
                seq_path = os.path.join(save_tracklet_graph_img_dir, tmp_str[-2])

            # Create directory for saving images
            if save_img and not os.path.exists(seq_path):
                # print("Creating directory: "+seq_path)
                os.makedirs(seq_path, exist_ok=True)

            # Setup txt save path for different datasets
            if dataset == "KITTI":
                if sub_class == "car":
                    save_txt_path = os.path.join(
                        save_txt_dir, "car", tmp_str[-2] + ".txt"
                    )
                elif sub_class == "person":
                    save_txt_path = os.path.join(
                        save_txt_dir, "person", tmp_str[-2] + ".txt"
                    )

                if os.path.exists(save_txt_path):
                    os.remove(save_txt_path)
            elif dataset == "Dance":
                # For DanceTrack, save the result as sequence_name.txt
                seq_name = tmp_str[-3]
                save_txt_path = os.path.join(save_txt_dir, seq_name + ".txt")

                if os.path.exists(save_txt_path):
                    os.remove(save_txt_path)

            # Open txt file for writing results
            if save_txt == True:
                txt_file = open(save_txt_path, "a")

            T = len(sample["img_paths"])

            for n in range(len(sample["img_paths"])):

                img_path = sample["img_paths"][n][0]
                # print("img_path: ", img_path)
                tmp_split = img_path.split("/")

                # Extract frame ID from filename
                if dataset == "Dance":
                    # DanceTrack format is like 00000001.jpg
                    frame_filename = tmp_split[-1]
                    fr_id = int(frame_filename.split(".")[0])
                else:
                    tmp_idx2 = tmp_split[-1].find("0")
                    fr_id = int(tmp_split[-1][tmp_idx2:-4])

                if save_img:
                    if data_dir is not None:
                        if dataset == "Dance":
                            # For DanceTrack, adjust the path according to its structure
                            tt_idx = img_path.find("img1")
                            # img_path = os.path.join(data_dir, img_path[tt_idx:])
                            # print("img_path: ", img_path)
                        else:
                            tt_idx = img_path.find("image_02")
                            img_path = os.path.join(data_dir, img_path[tt_idx:])

                    img = cv2.imread(img_path)

                    if n == 0:
                        height, width, layers = img.shape
                        img_size = (width, height)
                        out_video = cv2.VideoWriter(
                            seq_path + ".avi",
                            cv2.VideoWriter_fourcc(*"DIVX"),
                            10,
                            img_size,
                        )

                tmp_idx = sample["fr_ids"] == n
                tmp_bboxes = sample["boxes"][tmp_idx].detach().numpy()
                tmp_track_ids = batch_updated_tracklet_data[0][
                    "updated_tracklet_scores"
                ][tmp_idx.detach().numpy()[0]]

                for k in range(len(tmp_bboxes)):
                    tmp_id = int(tmp_track_ids[k])

                    if save_img:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        img = cv2.rectangle(
                            img,
                            (int(tmp_bboxes[k][0]), int(tmp_bboxes[k][1])),
                            (int(tmp_bboxes[k][2]), int(tmp_bboxes[k][3])),
                            255 * color_table[tmp_id],
                            2,
                        )
                        img = cv2.putText(
                            img,
                            str(tmp_id),
                            (int(tmp_bboxes[k][0]), int(tmp_bboxes[k][1])),
                            font,
                            1.2,
                            255 * color_table[tmp_id],
                            2,
                        )

                    if save_txt == True:
                        if dataset == "KITTI":
                            if sub_class == "car":
                                txt_file.write(
                                    "%i %i %s %i %i %i %.2f %.2f %.2f %.2f %i %i %i %i %i %i %i %.2f\n"
                                    % (
                                        fr_id,
                                        tmp_id,
                                        "Car",
                                        -1,
                                        -1,
                                        -10,
                                        tmp_bboxes[k][0],
                                        tmp_bboxes[k][1],
                                        tmp_bboxes[k][2],
                                        tmp_bboxes[k][3],
                                        -1,
                                        -1,
                                        -1,
                                        -1000,
                                        -1000,
                                        -1000,
                                        -10,
                                        1.0,
                                    )
                                )
                            if sub_class == "person":
                                txt_file.write(
                                    "%i %i %s %i %i %i %.2f %.2f %.2f %.2f %i %i %i %i %i %i %i %.2f\n"
                                    % (
                                        fr_id,
                                        tmp_id,
                                        "Pedestrian",
                                        -1,
                                        -1,
                                        -10,
                                        tmp_bboxes[k][0],
                                        tmp_bboxes[k][1],
                                        tmp_bboxes[k][2],
                                        tmp_bboxes[k][3],
                                        -1,
                                        -1,
                                        -1,
                                        -1000,
                                        -1000,
                                        -1000,
                                        -10,
                                        1.0,
                                    )
                                )
                        elif dataset == "Dance":
                            # DanceTrack格式: <frame_id>,<track_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<x>,<y>,<z>
                            # 轉換絕對座標到寬高格式
                            bbox_left = int(tmp_bboxes[k][0])
                            bbox_top = int(tmp_bboxes[k][1])
                            bbox_width = int(tmp_bboxes[k][2] - tmp_bboxes[k][0])
                            bbox_height = int(tmp_bboxes[k][3] - tmp_bboxes[k][1])

                            # MOT格式: frame_id,track_id,bbox_left,bbox_top,bbox_width,bbox_height,score,...
                            txt_file.write(
                                "%d,%d,%d,%d,%d,%d,1,-1,-1,-1\n"
                                % (
                                    fr_id,
                                    tmp_id + 1,
                                    bbox_left,
                                    bbox_top,
                                    bbox_width,
                                    bbox_height,
                                )
                            )

                save_path = os.path.join(seq_path, tmp_split[-1])
                if save_img:
                    cv2.imwrite(save_path, img)
                    out_video.write(img)

            if save_txt == True:
                txt_file.close()

            if save_img:
                out_video.release()


if __name__ == "__main__":
    test_tracklet_graph()
