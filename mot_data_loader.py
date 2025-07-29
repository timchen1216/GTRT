import numpy as np
import json

import torch.utils.data as data
import torch
import random


class TrackletSplit(object):
    def __init__(self, p0=0.9, p1=0.95, p2=0.9):
        # p0: 初始保持概率 - 決定軌跡是否從一開始就要保持
        # 較高的p0表示大多數軌跡會從頭開始保持
        self.p0 = p0

        # p1: 持續保持概率 - 當前是保持狀態時,下一幀繼續保持的概率
        # 較高的p1使得連續幀更可能保持在一起
        self.p1 = p1

        # p2: 持續移除概率 - 當前是移除狀態時,下一幀繼續移除的概率
        # 較高的p2使得一旦開始移除就會連續移除多幀
        self.p2 = p2

    def __call__(self, sample):
        uniq_ids = np.unique(sample["obj_ids"])
        uniq_ids = uniq_ids[uniq_ids != -1]

        remove_idx = []
        for n in range(len(uniq_ids)):
            tmp_idx = np.where(sample["obj_ids"] == uniq_ids[n])[0]
            rand_tmp = random.random()
            if rand_tmp < self.p0:
                v = True
            else:
                v = False

            for t in range(len(tmp_idx)):
                rand_tmp = random.random()
                if v:
                    if rand_tmp > self.p1:
                        remove_idx.append(tmp_idx[t])
                        v = False
                else:
                    if rand_tmp < self.p2:
                        remove_idx.append(tmp_idx[t])
                    else:
                        v = True

        sample["boxes"] = np.delete(sample["boxes"], remove_idx, 0)
        # sample["gt_boxes"] = np.delete(sample["gt_boxes"], remove_idx, 0)
        sample["classes"] = np.delete(sample["classes"], remove_idx, 0)
        sample["obj_ids"] = np.delete(sample["obj_ids"], remove_idx, 0)
        sample["fr_ids"] = np.delete(sample["fr_ids"], remove_idx, 0)
        return sample


class AddFP(object):
    def __init__(self, temporal_len, fpr=0.1):
        # fpr: false positive rate - 每個真實檢測框可能產生假陽性的概率
        self.fpr = fpr
        # temporal_len: 視頻片段的總幀數
        self.temporal_len = temporal_len

    def __call__(self, sample):
        h = sample["height"]
        w = sample["width"]

        # 對每一幀進行處理
        for t in range(self.temporal_len):
            # 計算當前所有檢測框的寬度和高度
            box_w = (sample["boxes"][:, 2] - sample["boxes"][:, 0]).copy()
            box_h = (sample["boxes"][:, 3] - sample["boxes"][:, 1]).copy()

            # 找出當前幀的所有檢測框索引
            cand_ids = np.where(sample["fr_ids"] == t)[0]
            # 如果當前幀檢測框少於2個,跳過(因為需要計算均值和標準差)
            if len(cand_ids) < 2:
                continue

            # 根據fpr決定要添加多少個假陽性
            fp_num = 0
            for k in range(len(cand_ids)):
                if np.random.rand() < self.fpr:
                    fp_num += 1

            if fp_num == 0:
                continue

            # 計算當前幀中檢測框的統計特徵:
            # 中心點(x,y)和寬高(w,h)的均值和標準差
            mean_x = np.mean(sample["boxes"][cand_ids, 0])
            mean_y = np.mean(sample["boxes"][cand_ids, 1])
            mean_w = np.mean(box_w[cand_ids])
            mean_h = np.mean(box_h[cand_ids])
            std_x = np.std(sample["boxes"][cand_ids, 0])
            std_y = np.std(sample["boxes"][cand_ids, 1])
            std_w = np.std(box_w[cand_ids])
            std_h = np.std(box_h[cand_ids])

            # 根據統計特徵生成假陽性框
            # 使用正態分布生成新的檢測框位置和大小
            xx = np.random.normal(mean_x, std_x, fp_num)  # 中心點x
            yy = np.random.normal(mean_y, std_y, fp_num)  # 中心點y
            ww = np.clip(np.random.normal(mean_w, std_w, fp_num), 0, w)  # 寬度
            hh = np.clip(np.random.normal(mean_h, std_h, fp_num), 0, h)  # 高度

            # 組裝成[x1,y1,x2,y2]格式的框
            fp_box = np.stack([xx, yy, xx + ww, yy + hh], 1).astype(np.int32)

            # concatenate according to fr order
            last_fr_idx = cand_ids[-1] + 1
            sample["boxes"] = np.concatenate(
                [sample["boxes"][:last_fr_idx], fp_box, sample["boxes"][last_fr_idx:]],
                0,
            )
            sample["classes"] = np.concatenate(
                [
                    sample["classes"][:last_fr_idx],
                    np.zeros((fp_num), dtype=np.int32),
                    sample["classes"][last_fr_idx:],
                ],
                0,
            )
            sample["obj_ids"] = np.concatenate(
                [
                    sample["obj_ids"][:last_fr_idx],
                    -np.ones((fp_num), dtype=np.int32),
                    sample["obj_ids"][last_fr_idx:],
                ],
                0,
            )
            sample["fr_ids"] = np.concatenate(
                [
                    sample["fr_ids"][:last_fr_idx],
                    (t * np.ones((fp_num), dtype=np.int32)).astype(np.int32),
                    sample["fr_ids"][last_fr_idx:],
                ],
                0,
            )

        return sample


class BoxJitter(object):
    def __init__(self, jitter_ratio=0.2):
        self.jitter_ratio = jitter_ratio

    def __call__(self, sample):
        sample["gt_boxes"] = sample["boxes"].copy()
        h = sample["height"]
        w = sample["width"]
        box_w = (sample["boxes"][:, 2] - sample["boxes"][:, 0]).copy()
        box_h = (sample["boxes"][:, 3] - sample["boxes"][:, 1]).copy()
        sample["boxes"][:, 0::2] = (
            sample["boxes"][:, 0::2]
            + np.expand_dims(box_w, 1)
            * self.jitter_ratio
            * (np.random.rand(len(sample["boxes"]), 2) - 0.5)
            * 2
        )
        sample["boxes"][:, 1::2] = (
            sample["boxes"][:, 1::2]
            + np.expand_dims(box_h, 1)
            * self.jitter_ratio
            * (np.random.rand(len(sample["boxes"]), 2) - 0.5)
            * 2
        )
        return sample


class BoxShift(object):
    def __init__(self, shift_ratio=0.2):
        self.shift_ratio = shift_ratio

    def __call__(self, sample):
        h = sample["height"]
        w = sample["width"]
        h_shift = self.shift_ratio * (random.random() - 0.5) * 2 * h
        w_shift = self.shift_ratio * (random.random() - 0.5) * 2 * w
        sample["boxes"][:, 0::2] = sample["boxes"][:, 0::2] + w_shift
        sample["boxes"][:, 1::2] = sample["boxes"][:, 1::2] + h_shift
        return sample


class BoxClip(object):
    def __call__(self, sample):
        h = sample["height"]
        w = sample["width"]
        sample["boxes"][:, 0::2] = np.clip(sample["boxes"][:, 0::2], 0, w)
        sample["boxes"][:, 1::2] = np.clip(sample["boxes"][:, 1::2], 0, h)
        remove_idx = np.where(
            (sample["boxes"][:, 0] >= w)
            + (sample["boxes"][:, 1] >= h)
            + (sample["boxes"][:, 2] <= 0)
            + (sample["boxes"][:, 3] <= 0)
            > 0
        )[0]
        sample["boxes"] = np.delete(sample["boxes"], remove_idx, 0)
        if "gt_boxes" in sample.keys():
            sample["gt_boxes"] = np.delete(sample["gt_boxes"], remove_idx, 0)
        sample["classes"] = np.delete(sample["classes"], remove_idx, 0)
        sample["obj_ids"] = np.delete(sample["obj_ids"], remove_idx, 0)
        sample["fr_ids"] = np.delete(sample["fr_ids"], remove_idx, 0)
        return sample


class HFlip(object):

    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = flip_ratio

    def __call__(self, sample):
        h = sample["height"]
        w = sample["width"]
        flip_prob = random.random()
        if flip_prob < self.flip_ratio:
            return sample
        else:
            tmp_x1 = w - sample["boxes"][:, 0].copy()
            tmp_x2 = w - sample["boxes"][:, 2].copy()
            sample["boxes"][:, 0] = tmp_x2
            sample["boxes"][:, 2] = tmp_x1
            return sample


class RandomDelete(object):

    def __init__(self, delete_ratio=0.15, max_bbox=2000):
        self.delete_ratio = delete_ratio
        self.max_bbox = max_bbox

    def __call__(self, sample):
        N = len(sample["obj_ids"])
        idx_arr = np.linspace(0, N - 1, N, dtype=np.int32)
        np.random.shuffle(idx_arr)
        if N > self.max_bbox:
            remove_idx = idx_arr[: N - self.max_bbox]
        else:
            remove_idx = idx_arr[: int(N * self.delete_ratio)]
        sample["boxes"] = np.delete(sample["boxes"], remove_idx, 0)
        sample["classes"] = np.delete(sample["classes"], remove_idx, 0)
        sample["obj_ids"] = np.delete(sample["obj_ids"], remove_idx, 0)
        sample["fr_ids"] = np.delete(sample["fr_ids"], remove_idx, 0)

        return sample


class CreateMOTDataset(data.Dataset):
    # 靜態類變量用於共享隨機狀態
    _shared_rng = random.Random(42)  # 固定種子
    _frame_skips = {}  # 用於存儲每個索引的跳幀狀態

    def __init__(
        self,
        data_path,
        temporal_len=64,
        transform=None,
        stride=4,
        random_skip=True,
    ):
        self.temporal_len = temporal_len
        self.random_skip = random_skip
        self.stride = stride

        with open(data_path) as json_file:
            self.track_data = json.load(json_file)
        self.transform = transform

        # 獲取所有影片序列
        seqs = list(self.track_data["data"].keys())
        self.vids = seqs

        # 建立新的索引映射
        self._build_video_index()

    def _build_video_index(self):
        """建立影片順序的索引映射"""
        self.video_indices = []

        for vid in self.vids:
            video_data = self.track_data["data"][str(vid)]
            video_st_fr = video_data["start_frame"]
            video_end_fr = video_data["end_frame"]

            # 計算該影片可以產生多少個樣本
            # 從start_frame開始，每次stride步長，直到temporal_len範圍內
            current_start = video_st_fr

            while current_start + self.temporal_len - 1 <= video_end_fr + self.stride:
                self.video_indices.append(
                    {"video_name": vid, "start_frame": current_start}
                )
                current_start += self.stride

        # print(f"Total samples: {len(self.video_indices)}")
        # for i, item in enumerate(self.video_indices[:10]):  # 顯示前10個樣本
        #     print(
        #         f"Sample {i}: video {item['video_name']}, start frame {item['start_frame']}"
        #     )

    def __len__(self):
        return len(self.video_indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}

        # 從新的索引映射中獲取影片和起始幀
        video_info = self.video_indices[idx]
        vid = video_info["video_name"]
        st_fr = video_info["start_frame"]
        end_fr = st_fr + self.temporal_len - 1

        # print(f"Processing sample {idx}: video {vid}, frames {st_fr} to {end_fr}")

        seq_boxes = []
        seq_fr_ids = []
        seq_classes = []
        seq_obj_ids = []
        seq_img_paths = []

        video_st_fr = self.track_data["data"][str(vid)]["start_frame"]
        video_end_fr = self.track_data["data"][str(vid)]["end_frame"]

        fr_cnt = 0

        # 為每個新的idx生成跳幀序列
        if idx not in self._frame_skips:
            self._frame_skips[idx] = []
            current_frame = st_fr
            while current_frame <= end_fr:
                skip = self._shared_rng.randint(0, 3) if self.random_skip else 0
                self._frame_skips[idx].extend([current_frame])
                current_frame += skip + 1

        # 使用預先生成的跳幀序列
        selected_frames = self._frame_skips[idx]
        # print("Selected frames:", selected_frames)

        for n in selected_frames:
            cur_fr = n
            if n < video_st_fr:
                cur_fr = video_st_fr
            if n > video_end_fr:
                cur_fr = video_end_fr

            seq_img_paths.append(
                self.track_data["data"][str(vid)]["video_dir"]
                + "/"
                + self.track_data["data"][vid]["images"][str(cur_fr)]["image_name"]
            )

            if "height" not in sample.keys():
                sample["height"] = self.track_data["data"][str(vid)]["height"]
                sample["width"] = self.track_data["data"][str(vid)]["width"]

            boxes = []
            classes = []
            obj_ids = []

            for m in range(
                len(
                    self.track_data["data"][str(vid)]["images"][str(cur_fr)][
                        "annotations"
                    ]
                )
            ):
                boxes.append(
                    self.track_data["data"][str(vid)]["images"][str(cur_fr)][
                        "annotations"
                    ][m]["bbox"]
                )
                classes.append(
                    self.track_data["data"][str(vid)]["images"][str(cur_fr)][
                        "annotations"
                    ][m]["category_id"]
                    - 1
                )
                obj_ids.append(
                    self.track_data["data"][str(vid)]["images"][str(cur_fr)][
                        "annotations"
                    ][m]["object_id"]
                )

            if len(boxes) == 0:
                fr_cnt += 1
                continue

            seq_boxes.append(np.array(boxes))
            seq_classes.append(np.array(classes))
            seq_obj_ids.append(np.array(obj_ids))
            seq_fr_ids.append(fr_cnt * np.ones((len(obj_ids))))
            fr_cnt += 1

        if len(seq_boxes) > 0:
            seq_boxes = np.concatenate(seq_boxes, 0)
            seq_classes = np.concatenate(seq_classes, 0)
            seq_obj_ids = np.concatenate(seq_obj_ids, 0)
            seq_fr_ids = np.concatenate(seq_fr_ids, 0)

        sample["boxes"] = seq_boxes
        sample["classes"] = seq_classes
        sample["obj_ids"] = seq_obj_ids
        if len(seq_fr_ids) > 0:
            sample["fr_ids"] = seq_fr_ids.astype(np.int32)
        sample["img_paths"] = seq_img_paths

        if self.transform and len(seq_boxes) > 0:
            sample = self.transform(sample)

        sample["start_frame"] = (
            max(selected_frames[0], video_st_fr) if selected_frames else st_fr
        )
        sample["end_frame"] = (
            min(selected_frames[-1], video_end_fr) if selected_frames else end_fr
        )
        sample["video_name"] = str(vid)
        # print(str(vid))

        return sample
