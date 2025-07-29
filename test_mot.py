import einops
import torch
import numpy as np
import cv2
import os
import time
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms


import build_det_graph
import head_utils, head_gnn_v2
import models.gtrt as gtrt
from mot_data_loader import CreateMOTDataset, HFlip, BoxShift, BoxClip, BoxJitter, AddFP
from tqdm import tqdm
from head_utils import get_tracklet_info
from config import *


def test_tracklet_graph():
    # Ensure the device is set correctly
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Remove local device definition

    # color table
    color_table = np.random.rand(5000, 3)
    # tracklet graph model
    tracklet_graph_model = gtrt.GTRT(
        temporal_length=tracklet_temporal_len + 1,
        num_id_vocabulary=num_id_vocabulary,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
        max_length=max_length,
    )
    tracklet_graph_model = tracklet_graph_model.to(device)
    tracklet_graph_checkpoint = torch.load(
        tracklet_graph_model_load_path, map_location=device
    )
    print("tracklet_graph_model_load_path", tracklet_graph_model_load_path)

    # 处理可能包含DataParallel前缀的状态字典
    state_dict = tracklet_graph_checkpoint["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # 移除'module.'前缀
        new_state_dict[name] = v

    # print(new_state_dict)
    tracklet_graph_model.load_state_dict(new_state_dict)
    tracklet_graph_model.eval()

    # data loader
    base_transforms = transforms.Compose([BoxClip()])
    mot_data = CreateMOTDataset(
        data_path=test_data_path,
        temporal_len=tracklet_temporal_len + 1,
        transform=base_transforms,
        stride=T_tracklet_stride,
        random_skip=False,
    )
    # mot_data = CreateMOTDataset(train_data_path, -1, transform=comp_transforms)
    dataloader = DataLoader(mot_data, batch_size=1, shuffle=False, num_workers=4)
    test_seq_info = prepare_test_info(dataloader)
    # print("test_seq_info", test_seq_info[0].keys())
    # print("test_seq_info", len(test_seq_info))
    # for k, v in test_seq_info[0].items():
    #     print(k, v.shape if isinstance(v, torch.Tensor) else v)
    last_id_labels = test_seq_info[0]["history_labels"]

    all_batch_results = []

    for batch in tqdm(
        test_seq_info,
        desc="Predicting batch tracklet labels",
        total=len(test_seq_info),
    ):
        _B, _N = last_id_labels.shape
        B, N, D, T = batch["tracklet_bbox"].shape
        N_min = min(_N, N)
        if "history_labels" not in batch:
            batch["history_labels"] = -torch.ones(
                (B, N), dtype=torch.int64, device=device
            )
            batch["history_labels"][0][:N_min] = last_id_labels[0][:N_min]

        result = tracklet_graph_model(batch, stage="test")
        unknown_id_prob = result["unknown_id_prob"]
        unknown_id_labels = torch.argmax(unknown_id_prob, dim=-1)
        last_id_labels = unknown_id_labels.clone()
        batch_results = batch.copy()
        # print(batch_results["history_time_window"])
        batch_results["unknown_id_prob"] = unknown_id_prob
        batch_results["unknown_id_labels"] = unknown_id_labels
        all_batch_results.append(batch_results)
        # print("unknown_id_prob", unknown_id_prob.shape)
        # print("time_window", batch["time_window"])
        print("history_id_labels", batch["history_labels"])
        print("history_id_mask", batch["history_id_mask"])
        print("unknown_id_labels", unknown_id_labels)
        print("unknown_id_mask", batch["tracklet_id_mask"])
        # print("img_paths", batch["img_paths"])
    print(len(all_batch_results))
    seq_batch_results = {}
    for batch_results in all_batch_results:
        video_name = batch_results["video_name"]
        # print("video_name", video_name)
        if video_name not in seq_batch_results:
            seq_batch_results[video_name] = []
        seq_batch_results[video_name].append(batch_results)

    seq_results = {}
    for video_name, results in seq_batch_results.items():
        print(f"Processing video: {video_name}, number of batches: {len(results)}")
        start_frame = results[0]["history_time_window"][0].item()
        end_frame = results[-1]["time_window"][1].item()
        seq_tracklet_bbox = results[0]["history_bbox"].clone()
        seq_tracklet_mask = results[0]["history_mask"].clone()
        seq_tracklet_id = results[0]["history_labels"].clone()
        seq_tracklet_id_mask = results[0]["history_id_mask"].clone()
        seq_tracklet_id_prob = label_to_one_hot(
            labels=seq_tracklet_id, n_classes=num_id_vocabulary + 1
        )
        seq_tracklet_bbox = einops.rearrange(seq_tracklet_bbox, "1 n d t -> t n d")
        seq_tracklet_mask = einops.rearrange(seq_tracklet_mask, "1 n d t -> t n d")
        seq_tracklet_id = einops.rearrange(seq_tracklet_id, "1 n -> n")
        # print("seq_tracklet_id", seq_tracklet_id)
        seq_tracklet_id_mask = einops.rearrange(seq_tracklet_id_mask, "1 n -> n")
        seq_tracklet_id_prob = einops.rearrange(seq_tracklet_id_prob, "1 n k -> n k")

        current_frames = seq_tracklet_bbox.shape[0]  # t 維度
        pad_frames = end_frame - current_frames
        if pad_frames > 0:
            seq_tracklet_bbox = F.pad(
                seq_tracklet_bbox, (0, 0, 0, 0, 0, pad_frames), value=-1
            )
            seq_tracklet_mask = F.pad(
                seq_tracklet_mask, (0, 0, 0, 0, 0, pad_frames), value=False
            )
        # print(seq_tracklet_bbox.shape, seq_tracklet_mask.shape)
        for res in results:
            window_start = res["time_window"][0].item() - 1  # -1 for zero-based index
            window_end = res["time_window"][1].item()
            # print(
            #     f"Window start: {window_start}, Window end: {window_end}, "
            #     f"Start frame: {start_frame}, End frame: {end_frame}"
            # )

            res_tracklet_bbox = res["tracklet_bbox"].clone()
            res_tracklet_mask = res["tracklet_mask"].clone()
            res_tracklet_bbox = einops.rearrange(res_tracklet_bbox, "1 n d t -> t n d")
            res_tracklet_mask = einops.rearrange(res_tracklet_mask, "1 n d t -> t n d")
            pad_tracklet_bbox = F.pad(
                res_tracklet_bbox,
                (
                    0,
                    0,
                    0,
                    0,
                    window_start,
                    end_frame - window_start - res_tracklet_bbox.shape[0],
                ),
                value=-1,
            )
            pad_tracklet_mask = F.pad(
                res_tracklet_mask,
                (
                    0,
                    0,
                    0,
                    0,
                    window_start,
                    end_frame - window_start - res_tracklet_mask.shape[0],
                ),
                value=False,
            )
            # print("pad_tracklet_bbox", pad_tracklet_bbox.shape)
            seq_tracklet_bbox = torch.cat((seq_tracklet_bbox, pad_tracklet_bbox), dim=1)
            seq_tracklet_mask = torch.cat((seq_tracklet_mask, pad_tracklet_mask), dim=1)
            # print("seq_tracklet_bbox", seq_tracklet_bbox.shape)
            # print("seq_tracklet_mask", seq_tracklet_mask.shape)

            res_tracklet_id_prob = res["unknown_id_prob"]
            res_tracklet_id = res["unknown_id_labels"]
            res_tracklet_id_mask = res["tracklet_id_mask"]
            res_tracklet_id_prob = einops.rearrange(
                res_tracklet_id_prob, "1 n k -> n k"
            )
            res_tracklet_id = einops.rearrange(res_tracklet_id, "1 n -> n")
            res_tracklet_id_mask = einops.rearrange(res_tracklet_id_mask, "1 n -> n")
            # print("res_tracklet_id_prob", res_tracklet_id_prob.shape)
            # print("res_tracklet_id", res_tracklet_id.shape)
            # print("res_tracklet_id_mask", res_tracklet_id_mask.shape)
            seq_tracklet_id_prob = torch.cat(
                (seq_tracklet_id_prob, res_tracklet_id_prob), dim=0
            )
            seq_tracklet_id = torch.cat((seq_tracklet_id, res_tracklet_id), dim=0)
            seq_tracklet_id_mask = torch.cat(
                (seq_tracklet_id_mask, res_tracklet_id_mask), dim=0
            )
            # print("seq_tracklet_id_prob", seq_tracklet_id_prob.shape)
            # print("seq_tracklet_id", seq_tracklet_id.shape)
            # print("seq_tracklet_id_mask", seq_tracklet_id_mask.shape)

        filtered_seq_tracklet_bbox = seq_tracklet_bbox[
            :, seq_tracklet_id_mask, :
        ]  # [112, valid_n, 4]
        filtered_seq_tracklet_mask = seq_tracklet_mask[
            :, seq_tracklet_id_mask, :
        ]  # [112, valid_n, 4]
        filtered_seq_tracklet_id_prob = seq_tracklet_id_prob[
            seq_tracklet_id_mask, :
        ]  # [valid_n, 51]
        filtered_seq_tracklet_id = seq_tracklet_id[seq_tracklet_id_mask]  # [valid_n]

        # 更新後的 mask（全部都是 True，因為已經過濾了）
        filtered_seq_tracklet_id_mask = seq_tracklet_id_mask[
            seq_tracklet_id_mask
        ]  # [valid_n] 全部是 True

        # print("filtered_seq_tracklet_bbox", filtered_seq_tracklet_bbox.shape)
        # print("filtered_seq_tracklet_mask", filtered_seq_tracklet_mask.shape)
        # print("filtered_seq_tracklet_id_prob", filtered_seq_tracklet_id_prob.shape)
        # print("filtered_seq_tracklet_id", filtered_seq_tracklet_id.shape)
        # print("filtered_seq_tracklet_id_mask", filtered_seq_tracklet_id_mask.shape)

        for frame in range(filtered_seq_tracklet_bbox.shape[0]):
            # print("frame: ", frame)
            bboxes = filtered_seq_tracklet_bbox[frame, :, :]
            mask = filtered_seq_tracklet_mask[frame, :, :]
            valid_rows = mask.any(dim=1)
            bboxes = bboxes[valid_rows]
            mask = mask[valid_rows]
            ids = filtered_seq_tracklet_id[valid_rows]
            # print(ids)
            id_prob = filtered_seq_tracklet_id_prob[valid_rows]

            unique_bboxes, inverse_indices, unique_counts = torch.unique(
                bboxes, dim=0, return_inverse=True, return_counts=True
            )
            # print(unique_bboxes)
            # print(inverse_indices)
            # print(unique_counts)
            for i, unique_bbox in enumerate(unique_bboxes):
                match_indices = (inverse_indices == i).nonzero(as_tuple=True)[0]
                corresponding_ids = ids[match_indices]
                # print(f"Unique bbox {i}: {unique_bbox.tolist()}")
                # print(f"  對應的原始索引: {match_indices.tolist()}")
                # print(f"  對應的所有ID: {corresponding_ids.tolist()}")
                # print(f"  出現次數: {unique_counts[i].item()}")

            # print(bboxes.shape)
            # print(unique_bboxes.shape)
            # print(mask.shape)
            # print(ids.shape)
            # print(id_prob.shape)

        # print(f"  Image Paths: {res['img_paths']}")
    # # prediction
    # seq_cnt = 0
    # batch_data = []
    # for i_batch, sample in enumerate(
    #     tqdm(dataloader, desc="Processing batches")
    # ):  # Added tqdm
    #     seq_cnt += 1
    #     # print("sample[boxes]", sample["boxes"].shape)
    #     # print("sample[boxes]", sample["boxes"])

    #     batch_data.append(sample)
    #     st_time = time.time()
    #     # print("device", device)

    #     batch_pred_tracklet_label, new_fr_ids, new_bbox = batch_test_tracklet_graph(
    #         batch_data, tracklet_graph_model, post_precessing, remove_N, device
    #     )
    #     # print("batch_pred_tracklet_label", batch_pred_tracklet_label)
    #     end_time = time.time()
    #     batch_data = []
    #     if len(batch_pred_tracklet_label) == 0:
    #         continue
    #     tmp_label = batch_pred_tracklet_label[0].copy()

    #     if post_precessing:
    #         sample["fr_ids"] = torch.from_numpy(new_fr_ids).unsqueeze(0).int()
    #         sample["boxes"] = torch.from_numpy(new_bbox).unsqueeze(0)

    #     # assign new label
    #     uniq_label = np.unique(tmp_label)
    #     # print("uniq_label", uniq_label.shape)
    #     # print("uniq_label", uniq_label)
    #     for n in range(len(uniq_label)):
    #         batch_pred_tracklet_label[0][tmp_label == uniq_label[n]] = n
    #     final_N = len(uniq_label)

    #     # save tracklet graph results
    #     tmp_str = sample["img_paths"][0][0].split("/")
    #     if dataset == "MOT" or dataset == "Dance":
    #         seq_path = save_tracklet_graph_img_dir + "/" + tmp_str[-3]

    #     if save_img:
    #         if not os.path.exists(seq_path):
    #             os.mkdir(seq_path)

    #     if save_txt:
    #         if not os.path.exists(save_txt_dir):
    #             os.mkdir(save_txt_dir)

    #     if dataset == "Dance":
    #         # For DanceTrack, save the result as sequence_name.txt
    #         seq_name = tmp_str[-3]
    #         save_txt_path = os.path.join(save_txt_dir, seq_name + ".txt")

    #         if os.path.exists(save_txt_path):
    #             os.remove(save_txt_path)

    #     if save_txt == True:
    #         if dataset == "Dance":
    #             txt_file = open(save_txt_path, "a")

    #     T = len(sample["img_paths"])

    #     for n in range(len(sample["img_paths"])):

    #         img_path = sample["img_paths"][n][0]
    #         tmp_split = img_path.split("/")
    #         tmp_idx2 = tmp_split[-1].find("0")
    #         fr_id = int(tmp_split[-1][tmp_idx2:-4])

    #         if save_img:
    #             if data_dir is not None:
    #                 if dataset == "Dance":
    #                     # For DanceTrack, adjust the path according to its structure
    #                     tt_idx = img_path.find("img1")

    #             img = cv2.imread(img_path)

    #             if n == 0:
    #                 height, width, layers = img.shape
    #                 img_size = (width, height)
    #                 out_video = cv2.VideoWriter(
    #                     seq_path + ".avi",
    #                     cv2.VideoWriter_fourcc(*"DIVX"),
    #                     10,
    #                     img_size,
    #                 )

    #         tmp_idx = sample["fr_ids"] == n
    #         # print("sample[fr_ids]", sample["fr_ids"].shape)
    #         # print("sample[fr_ids]", sample["fr_ids"])
    #         # print("sample[boxes]", sample["boxes"].shape)
    #         # print("sample[boxes]", sample["boxes"])
    #         # print("tmp_idx", tmp_idx)
    #         tmp_bboxes = sample["boxes"][tmp_idx].detach().numpy()
    #         # print("tmp_bboxes", tmp_bboxes.shape)
    #         # print("tmp_bboxes", tmp_bboxes)
    #         tmp_track_ids = batch_pred_tracklet_label[0][tmp_idx.detach().numpy()[0]]
    #         # print("tmp_track_ids", tmp_track_ids.shape)
    #         # print("tmp_track_ids", tmp_track_ids)
    #         for k in range(len(tmp_bboxes)):

    #             tmp_id = int(tmp_track_ids[k])
    #             if save_img:
    #                 font = cv2.FONT_HERSHEY_SIMPLEX
    #                 img = cv2.rectangle(
    #                     img,
    #                     (int(tmp_bboxes[k][0]), int(tmp_bboxes[k][1])),
    #                     (int(tmp_bboxes[k][2]), int(tmp_bboxes[k][3])),
    #                     255 * color_table[tmp_id],
    #                     2,
    #                 )
    #                 img = cv2.putText(
    #                     img,
    #                     str(tmp_id),
    #                     (int(tmp_bboxes[k][0]), int(tmp_bboxes[k][1])),
    #                     font,
    #                     1.2,
    #                     255 * color_table[tmp_id],
    #                     2,
    #                 )

    #             if save_txt == True:
    #                 if dataset == "Dance":
    #                     # DanceTrack格式: <frame_id>,<track_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<x>,<y>,<z>
    #                     # 轉換絕對座標到寬高格式
    #                     bbox_left = int(tmp_bboxes[k][0])
    #                     bbox_top = int(tmp_bboxes[k][1])
    #                     bbox_width = int(tmp_bboxes[k][2] - tmp_bboxes[k][0])
    #                     bbox_height = int(tmp_bboxes[k][3] - tmp_bboxes[k][1])

    #                     # MOT格式: frame_id,track_id,bbox_left,bbox_top,bbox_width,bbox_height,score,...
    #                     txt_file.write(
    #                         "%d,%d,%d,%d,%d,%d,1,-1,-1,-1\n"
    #                         % (
    #                             fr_id,
    #                             tmp_id + 1,
    #                             bbox_left,
    #                             bbox_top,
    #                             bbox_width,
    #                             bbox_height,
    #                         )
    #                     )

    #         save_path = seq_path + "/" + tmp_split[-1]
    #         if save_img:
    #             cv2.imwrite(save_path, img)
    #             out_video.write(img)

    #     if save_txt == True:
    #         txt_file.close()

    #     if save_img:
    #         out_video.release()


def prepare_test_info(dataloader):
    all_window_info_list = []
    cnt = 0
    for batch in tqdm(
        dataloader,
        desc="Preparing test info",
        total=len(dataloader),
    ):
        cnt += 1
        if cnt <= 0:
            continue
        if cnt == 300:
            break

        tracklet_dict = {}
        # for k, v in batch.items():
        #     print(k)
        for k, v in batch.items():
            if k != "img_paths" and k != "video_name":
                tracklet_dict[k] = v.squeeze(0).to(device).float()
            else:
                tracklet_dict[k] = v
        tracklet_dict["boxes"][:, 0::2] = tracklet_dict["boxes"][:, 0::2] / float(
            batch["width"].item()
        )
        tracklet_dict["boxes"][:, 1::2] = tracklet_dict["boxes"][:, 1::2] / float(
            batch["height"].item()
        )
        if soft_label:
            score = 0.9 + 0.1 * torch.rand(
                tracklet_dict["boxes"].shape[0], device=device
            )
        else:
            score = torch.ones(tracklet_dict["boxes"].shape[0], device=device)
        window_info = get_tracklet_info(
            det_ids=tracklet_dict["obj_ids"],
            gt_ids=None,
            det_fr_ids=tracklet_dict["fr_ids"],
            gt_fr_ids=None,
            det_bboxes=tracklet_dict["boxes"],
            gt_bboxes=None,
            scores=score,  # 使用生成的 score
            temporal_len=tracklet_temporal_len,
            device=device,
            stage="test",
        )

        # window_info["fr_ids"] = tracklet_dict["fr_ids"]
        window_info["img_paths"] = tracklet_dict["img_paths"]
        window_info["time_window"] = torch.tensor(
            [tracklet_dict["start_frame"].item(), tracklet_dict["end_frame"].item()],
            device=device,
            dtype=torch.int64,
        )

        video_name = tracklet_dict["video_name"]
        if isinstance(video_name, list):
            window_info["video_name"] = video_name[0]  # 提取第一個元素
        else:
            window_info["video_name"] = video_name
        # for k, v in window_info.items():
        #     print(k, v.shape if torch.is_tensor(v) else v)

        all_window_info_list.append(window_info)
    seq_info_list = []
    for i in range(1, len(all_window_info_list)):
        if (
            all_window_info_list[i]["video_name"]
            == all_window_info_list[i - 1]["video_name"]
        ):
            all_window_info_list[i]["history_bbox"] = all_window_info_list[i - 1][
                "tracklet_bbox"
            ]
            all_window_info_list[i]["history_mask"] = all_window_info_list[i - 1][
                "tracklet_mask"
            ]
            all_window_info_list[i]["history_time_window"] = all_window_info_list[
                i - 1
            ]["time_window"]
            seq_info_list.append(all_window_info_list[i])
    # for i in range(len(seq_info_list)):
    #     print("history_time_window", seq_info_list[i]["history_time_window"])
    #     print("time_window", seq_info_list[i]["time_window"])

    # padding
    for seq in seq_info_list:
        N, D, T = seq["tracklet_bbox"].shape
        _N, _D, _T = seq["history_bbox"].shape
        N_pad = max(N, _N)
        trajectory_bbox = -torch.ones((N_pad, D, T), dtype=torch.float32, device=device)
        trajectory_masks = torch.zeros((N_pad, D, T), dtype=torch.bool, device=device)
        history_bbox = -torch.ones((N_pad, D, T), dtype=torch.float32, device=device)
        history_masks = torch.zeros((N_pad, D, T), dtype=torch.bool, device=device)

        # print(data[i]["tracklet_mask"].shape)

        trajectory_bbox[:N, :, :] = seq["tracklet_bbox"]
        trajectory_masks[:N, :, :] = seq["tracklet_mask"]
        history_bbox[:_N, :, :] = seq["history_bbox"]
        history_masks[:_N, :, :] = seq["history_mask"]

        seq["tracklet_bbox"] = einops.rearrange(trajectory_bbox, "n d t -> 1 n d t")
        seq["tracklet_mask"] = einops.rearrange(trajectory_masks, "n d t -> 1 n d t")
        seq["tracklet_id_mask"] = einops.rearrange(
            torch.any(trajectory_masks, dim=(1, 2)), "n -> 1 n"
        )
        seq["history_bbox"] = einops.rearrange(history_bbox, "n d t -> 1 n d t")
        seq["history_mask"] = einops.rearrange(history_masks, "n d t -> 1 n d t")
        seq["history_id_mask"] = einops.rearrange(
            torch.any(history_masks, dim=(1, 2)), "n -> 1 n"
        )
        if seq["history_time_window"][0].item() == 1:

            first_label = torch.arange(0, _N, dtype=torch.int64, device=device)
            print("first_label", first_label)
            seq["history_labels"] = einops.rearrange(first_label, "n -> 1 n")
            # print(seq["history_labels"])

    # first_id_labels = einops.rearrange(
    #     torch.arange(
    #         0, seq_info_list[0]["tracklet_bbox"].shape[1], device=device
    #     ).int(),
    #     "n -> 1 n",
    # )
    # # print("first_labels", first_id_labels.shape)
    # # print(seq_info_list[0]["history_id_mask"].shape)
    # seq_info_list[0]["history_labels"] = torch.where(
    #     seq_info_list[0]["history_id_mask"], first_id_labels, torch.tensor(-1)  # 默认值
    # )
    # print("first_id_labels", seq_info_list[0]["history_labels"])

    return seq_info_list


def label_to_one_hot(labels: torch.Tensor, n_classes: int, dtype=torch.float32):
    # 創建 one-hot tensor，初始化為全0
    one_hot = torch.zeros(
        labels.shape + (n_classes,), dtype=dtype, device=labels.device
    )

    # 找到有效的標籤（>= 0 且 < n_classes）
    valid_mask = (labels >= 0) & (labels < n_classes)

    if valid_mask.any():
        # 只對有效標籤進行 one-hot 編碼
        valid_labels = labels[valid_mask]
        eye_matrix = torch.eye(n_classes, dtype=dtype, device=labels.device)
        one_hot[valid_mask] = eye_matrix[valid_labels]

    # 無效標籤（如 -1）會保持為全0向量
    return one_hot


if __name__ == "__main__":
    test_tracklet_graph()
