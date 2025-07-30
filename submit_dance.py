
import os
import json
import argparse
import time
from copy import deepcopy
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models import build_model
from util.tool import load_model
from main import get_args_parser
from models.structures import Instances


def check_cuda():
    """检查CUDA可用性"""
    assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU environment."


def write_log(log_path, text):
    """将日志信息写入文件"""
    with open(log_path, 'a') as f:
        f.write(text + '\n')


class ListImgDataset(Dataset):
    """自定义图像数据集"""

    def __init__(self, mot_path, img_list, det_db):
        super().__init__()
        self.mot_path = mot_path
        self.img_list = img_list
        self.det_db = det_db
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, f_path):
        cur_img = cv2.imread(os.path.join(self.mot_path, f_path))
        assert cur_img is not None, f"Failed to load image {f_path}"
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        proposals = []
        im_h, im_w = cur_img.shape[:2]
        key = f_path[:-4] + '.txt'
        if key not in self.det_db:
            print(f"Warning: Key '{key}' not found in det_db.")
            return cur_img, torch.zeros((0, 5))
        for line in self.det_db[key]:
            l, t, w, h, s = map(float, line.split(','))
            proposals.append([
                (l + w / 2) / im_w,
                (t + h / 2) / im_h,
                w / im_w,
                h / im_h,
                s
            ])
        return cur_img, torch.as_tensor(proposals).reshape(-1, 5)

    def init_img(self, img, proposals):
        ori_img = img.copy()
        seq_h, seq_w = img.shape[:2]
        scale = self.img_height / min(seq_h, seq_w)
        if max(seq_h, seq_w) * scale > self.img_width:
            scale = self.img_width / max(seq_h, seq_w)
        target_h, target_w = int(seq_h * scale), int(seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std).unsqueeze(0)
        return img, ori_img, proposals

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, proposals = self.load_img_from_file(self.img_list[index])
        return self.init_img(img, proposals)

class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=10):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores >= self.score_thresh)
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.filter_score_thresh)
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance)
        track_instances.obj_idxes[to_del] = -1



class Detector:
    """目标检测器"""

    def __init__(self, args, model, vid):
        self.args = args
        self.detr = model
        self.vid = vid
        self.seq_num = os.path.basename(vid)
        img_dir = os.path.join(args.mot_path, vid, 'img1')
        self.img_list = sorted([
            os.path.join(vid, 'img1', f)
            for f in os.listdir(img_dir)
            if f.endswith('.jpg')
        ])
        self.predict_path = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(self.predict_path, exist_ok=True)

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        keep &= dt_instances.obj_idxes >= 0
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]

    def detect(self, prob_threshold=0.6, area_threshold=100):
        total_dts = 0
        track_instances = None

        with open(os.path.join(self.args.mot_path, self.args.det_db)) as f:
            det_db = json.load(f)

        loader = DataLoader(
            ListImgDataset(self.args.mot_path, self.img_list, det_db),
            batch_size=1,
            num_workers=2
        )

        lines = []
        log_path = os.path.join(self.predict_path, f'{self.seq_num}_log.txt')
        write_log(log_path, f"Start detection for {self.seq_num}")

        frame_times = []

        with torch.no_grad():
            for i, data in enumerate(tqdm(loader)):
                start_time = time.time()

                cur_img, ori_img, proposals = [d[0] for d in data]
                cur_img, proposals = cur_img.cuda(), proposals.cuda()

                if track_instances is not None:
                    track_instances.remove('boxes')
                    track_instances.remove('labels')

                seq_h, seq_w, _ = ori_img.shape
                res = self.detr.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
                track_instances = res['track_instances']

                dt_instances = deepcopy(track_instances)
                dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
                dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

                total_dts += len(dt_instances)
                bbox_xyxy = dt_instances.boxes.tolist()
                identities = dt_instances.obj_idxes.tolist()

                save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
                for xyxy, track_id in zip(bbox_xyxy, identities):
                    if track_id < 0 or track_id is None:
                        continue
                    x1, y1, x2, y2 = xyxy
                    w, h = x2 - x1, y2 - y1
                    lines.append(save_format.format(frame=i + 1, id=track_id, x1=x1, y1=y1, w=w, h=h))

                # FPS 与显存统计 (仅记录，不输出)
                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000
                frame_times.append(elapsed_ms)

        output_file = os.path.join(self.predict_path, f'{self.seq_num}.txt')
        with open(output_file, 'w') as f:
            f.writelines(lines)

        # 统计整体 FPS
        avg_time = sum(frame_times) / len(frame_times)
        avg_fps = 1000.0 / avg_time if avg_time > 0 else 0
        summary = (f"[SUMMARY] {self.seq_num}: "
                   f"Total frames = {len(frame_times)}, "
                   f"Average Time = {avg_time:.2f} ms, "
                   f"Average FPS = {avg_fps:.2f}, "
                   f"Total detections = {total_dts}")
        print(summary)
        write_log(log_path, summary)


if __name__ == '__main__':
    check_cuda()

    parser = argparse.ArgumentParser('DETR GPU inference script', parents=[get_args_parser()])
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--update_score_threshold', default=0.4, type=float)
    parser.add_argument('--miss_tolerance', default=20, type=int)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model, _, _ = build_model(args)
    model.track_embed.score_thr = args.update_score_threshold

    # import RuntimeTrackerBase  
    model.track_base = RuntimeTrackerBase(
        args.score_threshold,
        args.score_threshold,
        args.miss_tolerance
    )

    checkpoint = torch.load(args.resume, map_location='cpu')
    model = load_model(model, args.resume)
    model.eval()
    model = model.cuda()

    sub_dir = 'DanceTrack/val'
    seq_nums = [seq for seq in os.listdir(os.path.join(args.mot_path, sub_dir)) if seq != 'seqmap']
    vids = [os.path.join(sub_dir, seq) for seq in seq_nums]

    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    vids = vids[rank::ws]

    for vid in vids:
        detector = Detector(args, model=model, vid=vid)
        detector.detect(args.score_threshold)
