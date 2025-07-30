
# ------------------------------------------------------------------------
# 版权声明：本代码继承自Deformable DETR和DETR，二次开发版权归megvii-research所有
# ------------------------------------------------------------------------

"""
多目标跟踪数据集加载器（支持MOT格式和CrowdHuman联合训练）
主要功能：
1. 加载DanceTrack等MOT格式数据集
2. 集成CrowdHuman静态检测数据集
3. 实现时序连续帧采样策略
4. 支持检测框提议增强
"""

from collections import defaultdict
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
from PIL import Image, ImageDraw
import copy
import datasets.transforms as T  # 自定义数据增强模块
from models.structures import Instances  # 自定义数据结构容器
from random import choice, randint


def is_crowd(ann):
    """判断CrowdHuman标注是否为拥挤区域（需忽略）"""
    return 'extra' in ann and 'ignore' in ann['extra'] and ann['extra']['ignore'] == 1


class DetMOTDetection(torch.utils.data.Dataset):
    """多目标跟踪数据集类（支持时序连续帧加载）

    核心功能：
    - 从MOT格式数据集中加载连续帧序列
    - 整合CrowdHuman静态检测数据
    - 实现随机间隔采样策略
    - 生成检测提议框（用于训练查询初始化）
    """

    def __init__(self, args, data_txt_path: str, seqs_folder, transform):
        """
        参数:
            args: 配置参数对象（包含数据集路径、采样策略等）
            data_txt_path: 数据集划分文件路径（未实际使用，保留接口兼容）
            seqs_folder: 数据集根目录路径
            transform: 数据增强流水线
        """
        self.args = args
        self.transform = transform
        self.num_frames_per_batch = max(args.sampler_lengths)  # 每个样本的连续帧数（默认为5）
        self.sample_mode = args.sample_mode  # 采样模式（random_interval/fixed_interval）
        self.sample_interval = args.sample_interval  # 最大采样间隔（用于随机模式）
        self.video_dict = {}  # 视频名称到数字ID的映射
        self.mot_path = args.mot_path  # MOT数据集根目录

        # 加载MOT格式标注数据（以DanceTrack为例）
        self.labels_full = defaultdict(lambda: defaultdict(list))  # 三级字典结构：vid -> 帧号 -> 标注列表

        def add_mot_folder(split_dir):
            """加载指定子目录的标注数据"""
            print("Adding", split_dir)
            for vid in os.listdir(os.path.join(self.mot_path, split_dir)):
                if 'seqmap' == vid: continue  # 忽略元数据文件
                vid_path = os.path.join(split_dir, vid)
                if 'DPM' in vid or 'FRCNN' in vid:  # 过滤特定检测器的数据
                    print(f'filter {vid_path}')
                    continue
                gt_path = os.path.join(self.mot_path, vid_path, 'gt', 'gt.txt')
                for line in open(gt_path):
                    # 解析标注行：帧号, 目标ID, 框坐标, 标志位, 类别
                    t, i, *xywh, mark, label = line.strip().split(',')[:8]
                    t, i, mark, label = map(int, (t, i, mark, label))
                    if mark == 0: continue  # 忽略不可见目标
                    if label in [3, 4, 5, 6, 9, 10, 11]: continue  # 过滤非行人类别
                    x, y, w, h = map(float, (xywh))
                    self.labels_full[vid_path][t].append([x, y, w, h, i, False])  # 格式：[x,y,w,h,id,iscrowd]

        add_mot_folder("DanceTrack/train")  # 加载训练集
        vid_files = list(self.labels_full.keys())  # 获取所有视频序列

        # 构建采样索引（video, start_frame）
        self.indices = []  # 存储所有可采样的起始帧
        self.vid_tmax = {}  # 记录每个视频的最大帧号
        for vid in vid_files:
            self.video_dict[vid] = len(self.video_dict)  # 分配视频ID
            t_min = min(self.labels_full[vid].keys())  # 视频起始帧
            t_max = max(self.labels_full[vid].keys()) + 1  # 视频结束帧+1
            self.vid_tmax[vid] = t_max - 1
            # 生成该视频所有可能的采样起始点
            for t in range(t_min, t_max - self.num_frames_per_batch):
                self.indices.append((vid, t))
        print(f"Found {len(vid_files)} videos, {len(self.indices)} frames")

        # 动态采样策略配置（根据训练epoch调整采样长度）
        self.sampler_steps = args.sampler_steps  # 调整采样长度的epoch节点（例如[10, 20]）
        self.lengths = args.sampler_lengths  # 各阶段对应的采样长度（例如[5, 5, 3]）
        print(f"sampler_steps={self.sampler_steps} lengths={self.lengths}")
        self.period_idx = 0  # 当前采样阶段索引

        # 加载CrowdHuman数据集（用于联合训练）
        self.ch_dir = Path(args.mot_path) / 'crowdhuman'
        self.ch_indices = []  # 存储CrowdHuman样本索引
        if args.append_crowd:
            for line in open(self.ch_dir / "annotation_trainval.odgt"):
                datum = json.loads(line)
                # 过滤拥挤区域并收集有效标注
                boxes = [ann['fbox'] for ann in datum['gtboxes'] if not is_crowd(ann)]
                self.ch_indices.append((datum['ID'], boxes))
        print(f"Loaded {len(self.ch_indices)} CrowdHuman images")

        # 加载预生成检测提议框（用于训练时查询初始化）
        if args.det_db:
            with open(os.path.join(args.mot_path, args.det_db)) as f:
                self.det_db = json.load(f)  # 格式：{图像路径: [检测框列表]}
        else:
            self.det_db = defaultdict(list)

    def set_epoch(self, epoch):
        """动态调整采样策略（根据当前训练epoch）"""
        self.current_epoch = epoch
        if not self.sampler_steps: return  # 固定采样长度模式

        # 判断当前所属的训练阶段
        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print(f"Epoch {epoch}: Using sampler period {self.period_idx}")
        self.num_frames_per_batch = self.lengths[self.period_idx]  # 更新采样长度

    def step_epoch(self):
        """训练epoch结束时自动调整采样策略"""
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        """将标注字典转换为Instances对象"""
        gt_instances = Instances(tuple(img_shape))
        n_gt = len(targets['labels'])  # 真实标注数量（检测提议框在后）
        gt_instances.boxes = targets['boxes'][:n_gt]  # 真实框
        gt_instances.labels = targets['labels']  # 类别标签（全0，MOT仅行人）
        gt_instances.obj_ids = targets['obj_ids']  # 跟踪ID（跨帧唯一）
        return gt_instances

    def load_crowd(self, index):
        """加载CrowdHuman样本（用于增强训练数据）"""
        ID, boxes = self.ch_indices[index]
        boxes = copy.deepcopy(boxes)
        img = Image.open(self.ch_dir / 'images' / f'{ID}.jpg')

        # 合并真实框和检测提议框
        w, h = img.size
        n_gts = len(boxes)
        scores = [0.] * n_gts  # 真实框置信度设为1.0
        # 加载该图像的检测提议框（来自预生成文件）
        for line in self.det_db.get(f'crowdhuman/image/{ID}.txt', []):
            *box, s = map(float, line.split(','))  # 格式：x,y,w,h,score
            boxes.append(box)
            scores.append(s)

        # 转换为张量并调整框格式（xywh -> xyxy）
        boxes = torch.tensor(boxes, dtype=torch.float32)
        areas = boxes[..., 2:].prod(-1)  # 计算区域面积（用于损失加权）
        boxes[:, 2:] += boxes[:, :2]  # 转换格式：xywh -> xyxy

        target = {
            'boxes': boxes,  # 真实框在前，检测提议在后
            'scores': torch.tensor(scores),
            'labels': torch.zeros(n_gts, dtype=torch.long),  # 全0表示行人
            'iscrowd': torch.zeros(n_gts, dtype=torch.bool),
            'image_id': torch.tensor([0]),  # 静态图像无时序ID
            'area': areas,
            'obj_ids': torch.arange(n_gts),  # 生成伪ID（仅用于兼容接口）
            'size': torch.tensor([h, w]),  # 图像尺寸（高，宽）
            'orig_size': torch.tensor([h, w]),
            'dataset': "crowdhuman",
        }
        # 应用随机平移增强（模拟视频中的运动模糊）
        rs = T.FixedMotRandomShift(self.num_frames_per_batch)
        return rs([img], [target])

    def _pre_single_frame(self, vid, idx: int):
        """预处理单个视频帧"""
        img_path = os.path.join(self.mot_path, vid, 'img1', f'{idx:08d}.jpg')
        img = Image.open(img_path)
        targets = {}
        w, h = img.size
        assert w > 0 and h > 0, f"Invalid image {img_path}"

        # 生成跨视频唯一的对象ID（video_id * 1e5 + track_id）
        obj_idx_offset = self.video_dict[vid] * 100000

        # 初始化目标字典
        targets = {
            'dataset': 'MOT17',
            'boxes': [],  # 边界框列表（xywh格式）
            'iscrowd': [],  # 是否拥挤区域
            'labels': [],  # 类别标签（全0）
            'obj_ids': [],  # 唯一对象ID
            'scores': [],  # 置信度（真实框为1.0，检测提议为预测值）
            'image_id': torch.tensor(idx),
            'size': torch.tensor([h, w]),
            'orig_size': torch.tensor([h, w]),
        }

        # 加载真实标注
        for *xywh, id, crowd in self.labels_full[vid].get(idx, []):
            targets['boxes'].append(xywh)
            targets['iscrowd'].append(crowd)
            targets['labels'].append(0)
            targets['obj_ids'].append(id + obj_idx_offset)  # 生成全局唯一ID
            targets['scores'].append(1.0)  # 真实框置信度设为1.0

        # 加载检测提议框（来自预生成文件）
        txt_key = os.path.join(vid, 'img1', f'{idx:08d}.txt')
        for line in self.det_db.get(txt_key, []):
            *box, s = map(float, line.split(','))  # 格式：x,y,w,h,score
            targets['boxes'].append(box)
            targets['scores'].append(s)

        # 转换为张量并调整格式
        targets.update({
            'iscrowd': torch.tensor(targets['iscrowd']),
            'labels': torch.tensor(targets['labels']),
            'obj_ids': torch.tensor(targets['obj_ids'], dtype=torch.float64),  # 避免精度丢失
            'scores': torch.tensor(targets['scores']),
            'boxes': torch.tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        })
        targets['boxes'][:, 2:] += targets['boxes'][:, :2]  # xywh -> xyxy
        return img, targets

    def _get_sample_range(self, start_idx):
        """生成连续帧采样范围"""
        assert self.sample_mode in ['fixed_interval', 'random_interval']
        if self.sample_mode == 'fixed_interval':
            interval = self.sample_interval
        else:
            interval = np.random.randint(1, self.sample_interval + 1)
        end_idx = start_idx + (self.num_frames_per_batch - 1) * interval
        return (start_idx, end_idx + 1, interval)

    def pre_continuous_frames(self, vid, indices):
        """加载连续帧序列"""
        return zip(*[self._pre_single_frame(vid, i) for i in indices])

    def sample_indices(self, vid, f_index):
        """随机采样连续帧索引（处理视频边界）"""
        rate = randint(1, self.sample_interval)
        tmax = self.vid_tmax[vid]
        return [min(f_index + rate * i, tmax) for i in range(self.num_frames_per_batch)]

    def __getitem__(self, idx):
        """核心数据加载方法"""
        # 动态选择数据源（MOT视频帧或CrowdHuman静态图）
        if idx < len(self.indices):
            # MOT视频帧采样
            vid, f_index = self.indices[idx]
            indices = self.sample_indices(vid, f_index)
            images, targets = self.pre_continuous_frames(vid, indices)
        else:
            # CrowdHuman静态图采样（索引超出视频样本数时触发）
            images, targets = self.load_crowd(idx - len(self.indices))

        # 应用数据增强
        if self.transform:
            images, targets = self.transform(images, targets)

        # 转换为模型输入格式
        gt_instances, proposals = [], []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)
            # 分离真实框和检测提议框（真实框在前，检测在后）
            n_gt = len(targets_i['labels'])
            proposals.append(torch.cat([
                targets_i['boxes'][n_gt:],  # 检测提议框坐标
                targets_i['scores'][n_gt:, None],  # 检测分数
            ], dim=1))

        return {
            'imgs': images,  # 图像张量列表 [B, C, H, W]
            'gt_instances': gt_instances,  # 真实标注实例列表
            'proposals': proposals,  # 检测提议列表（用于查询初始化）
        }

    def __len__(self):
        """数据集总大小 = 视频帧数 + CrowdHuman样本数"""
        return len(self.indices) + len(self.ch_indices)


class DetMOTDetectionValidation(DetMOTDetection):
    """验证集专用类（继承自训练集类）"""

    def __init__(self, args, seqs_folder, transform):
        args.data_txt_path = args.val_data_txt_path  # 覆盖验证集路径
        super().__init__(args, seqs_folder, transform)


def make_transforms_for_mot17(image_set, args=None):
    """构建MOT数据增强流水线

    参数:
        image_set: 数据集阶段（train/val）
        args: 配置参数（未使用，保留接口）

    返回:
        T.MotCompose: 数据增强操作序列
    """
    normalize = T.MotCompose([
        T.MotToTensor(),  # 转换为张量并归一化[0,1]
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet均值方差
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomHorizontalFlip(),  # 随机水平翻转
            T.MotRandomSelect(  # 随机选择缩放策略
                T.MotRandomResize(scales, max_size=1536),  # 直接多尺度缩放
                T.MotCompose([  # 先裁剪后缩放组合
                    T.MotRandomResize([800, 1000, 1200]),
                    T.FixedMotRandomCrop(800, 1200),  # 固定比例随机裁剪
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            T.MOTHSV(),  # HSV颜色空间增强
            normalize,
        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),  # 固定尺寸缩放
            normalize,
        ])

    raise ValueError(f'Unknown image_set: {image_set}')

def build_transform(args, image_set):
    """构建MOT数据增强流水线

    参数:
        args (argparse.Namespace): 配置参数对象
        image_set (str): 数据集类型（'train'/'val'）

    返回:
        T.MotCompose: 组合后的数据增强操作序列

    实现逻辑:
        - 根据数据集类型调用专用增强函数
        - 统一返回增强后的变换管道
    """
    # 调用MOT专用增强函数生成基础变换流水线
    mot17_train = make_transforms_for_mot17('train', args)
    mot17_test = make_transforms_for_mot17('val', args)

    # 根据数据集类型选择对应流水线
    if image_set == 'train':
        return mot17_train
    elif image_set == 'val':
        return mot17_test
    else:
        raise NotImplementedError(f"不支持的数据集类型: {image_set}")

def build(image_set, args):
    """
     数据集构建入口函数

        参数:
            image_set (str): 数据集类型（'train'/'val'）
            args (argparse.Namespace): 配置参数对象

        返回:
            DetMOTDetection: 构建完成的数据集对象

        执行流程:
            1. 验证数据集路径有效性
            2. 构建数据增强流水线
            3. 根据数据集类型创建对应实例
    """

    # 验证MOT数据集根目录是否存在
    root = Path(args.mot_path)
    assert root.exists(), f"提供的MOT路径 {root} 不存在"

    # 构建数据增强流水线
    transform = build_transform(args, image_set)

    # 根据数据集类型创建实例
    if image_set == 'train':
        data_txt_path = args.data_txt_path_train  # 训练集划分文件路径
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, transform=transform)
    elif image_set == 'val':
        data_txt_path = args.data_txt_path_val  # 验证集划分文件路径
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, transform=transform)
    else:
        raise NotImplementedError(f"不支持的数据集类型: {image_set}")

    return dataset
