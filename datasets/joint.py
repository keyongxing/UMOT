

from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
from PIL import Image, ImageDraw
import copy
import datasets.transforms as T
from models.structures import Instances


class DetMOTDetection:
    """多目标跟踪数据集主类

    参数：
        args (argparse.Namespace): 配置参数对象
        data_txt_path (str): 数据列表文件路径
        seqs_folder (Path): 数据集根目录
        dataset2transform (dict): 数据集名称到变换函数的映射

    核心功能：
        - 支持视频序列加载和静态图像混合训练
        - 动态课程学习采样窗口调整
        - 多尺度数据增强
        - 跨视频ID管理

    属性：
        num_frames_per_batch (int): 当前采样窗口长度
        period_idx (int): 当前训练阶段索引
        video_dict (dict): 视频路径到唯一ID的映射
"""
    def __init__(self, args, data_txt_path: str, seqs_folder, dataset2transform):
        self.video_dict = {}  # 显式初始化视频字典

        self.args = args
        self.dataset2transform = dataset2transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.vis = args.vis
        #
        # # 读取图像文件列表
        # with open(data_txt_path, 'r') as file:
        #     self.img_files = [osp.join(seqs_folder, x.strip()) for x in file.readlines()]
        #     self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))
        #
        # # 对应标签文件路径生成
        # self.label_files = [(x.replace('Images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt'))
        #                     for x in self.img_files]
        #

        # 修改后的文件路径生成逻辑
        self.img_files = []
        self.label_files = []

        # 读取原始文件列表
        with open(data_txt_path, 'r') as file:
            raw_paths = [x.strip() for x in file.readlines() if x.strip()]

        # 同步生成图片和标签路径，并过滤无效文件
        for raw_path in raw_paths:
            img_path = osp.join(seqs_folder, raw_path)

            # 统一路径生成规则
            if 'crowdhuman' in img_path:
                # CrowdHuman 特殊处理
                label_path = img_path.replace('images', 'labels_with_ids').replace('.jpg', '.txt')
            else:
                # MOT17 标准处理
                label_path = img_path.replace('Images', 'labels_with_ids').replace('.jpg', '.txt').replace('.png',
                                                                                                           '.txt')

            # 强制路径存在性检查
            if osp.exists(img_path) and osp.exists(label_path):
                self.img_files.append(img_path)
                self.label_files.append(label_path)
            else:
                print(
                    f"忽略无效文件对: \n  - 图像: {img_path} {'存在' if osp.exists(img_path) else '缺失'} \n  - 标签: {label_path} {'存在' if osp.exists(label_path) else '缺失'}")

        # 验证数据一致性
        assert len(self.img_files) == len(self.label_files), "图像与标签文件数量不一致"
        print(f"有效样本数量: {len(self.img_files)} (过滤掉{len(raw_paths) - len(self.img_files)}个无效样本)")

        # 更新item_num计算逻辑
        min_frames = (self.num_frames_per_batch - 1) * self.sample_interval + 1
        self.item_num = max(0, len(self.img_files) - min_frames + 1)

        # 计算有效样本数量
        # self.item_num = len(self.img_files) - (self.num_frames_per_batch - 1) * self.sample_interval
        min_frames = (self.num_frames_per_batch - 1) * self.sample_interval + 1
        self.item_num = max(0, len(self.img_files) - min_frames + 1)

        # 注册视频信息
        self._register_videos()

        # 课程学习配置
        self.sampler_steps: list = args.sampler_steps
        self.lengths: list = args.sampler_lengths
        print(f"sampler_steps={self.sampler_steps} lengths={self.lengths}")
        if self.sampler_steps and len(self.sampler_steps) > 0:
            assert len(self.lengths) == len(self.sampler_steps) + 1
            for i in range(len(self.sampler_steps) - 1):
                assert self.sampler_steps[i] < self.sampler_steps[i + 1]
            self.item_num = len(self.img_files) - (self.lengths[-1] - 1) * self.sample_interval
            self.period_idx = 0
            self.num_frames_per_batch = self.lengths[0]

        with open(data_txt_path, 'r') as file:
            self.img_files = [osp.join(seqs_folder, x.strip()) for x in file.readlines()]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [...]

    def _register_videos(self):
        """注册视频信息到字典"""
        for label_name in self.label_files:
            video_name = '/'.join(label_name.split('/')[:-1])
            if video_name not in self.video_dict:
                print(f"注册第{len(self.video_dict) + 1}个视频: {video_name}")
                self.video_dict[video_name] = len(self.video_dict)

    def set_epoch(self, epoch):
        """根据当前epoch动态调整采样窗口长度"""
        if not self.sampler_steps or len(self.sampler_steps) == 0:
            return
        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print(f"set epoch={epoch} period_idx={self.period_idx}")
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        """标记一个epoch结束"""
        print(f"Dataset: epoch {self.current_epoch} finishes")
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        """将标注字典转换为Instances对象"""
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = targets['boxes']
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        gt_instances.area = targets['area']
        return gt_instances

    def _pre_single_frame(self, i):
        if i >= len(self.image_files):
            raise IndexError("Frame index exceeds available images.")
        label_path = self.label_files[i]

    def _pre_single_frame(self, idx: int):


            # 添加调试输出
        print(f"Current index: {idx}, Total labels: {len(self.label_files)}")

        """预处理单帧图像和标注"""
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]

        # 新增路径检查
        if not osp.exists(img_path):
            raise FileNotFoundError(f"Image file {img_path} not found")
        if not osp.exists(label_path):
            raise FileNotFoundError(f"Label file {label_path} not found")


        # 处理CrowdHuman数据集的特殊命名
        if 'crowdhuman' in img_path:
            img_path = img_path.replace('.jpg', '.png')
        img = Image.open(img_path)
        w, h = img.size

        # 加载标注文件
        if osp.isfile(label_path):
            labels = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
            # 转换为像素坐标系
            labels[:, 2] = w * (labels[0][:, 2] - labels[0][:, 4] / 2)
            labels[:, 3] = h * (labels[0][:, 3] - labels[0][:, 5] / 2)
            labels[:, 4] = w * (labels[0][:, 2] + labels[0][:, 4] / 2)
            labels[:, 5] = h * (labels[0][:, 3] + labels[0][:, 5] / 2)
        else:
            raise ValueError(f"Invalid label path: {label_path}")

        # 设置视频ID偏移量（每个视频1e6唯一ID）
        video_name = '/'.join(label_path.split('/')[:-1])
        obj_id_offset = self.video_dict[video_name] * 1000000

        # 初始化目标字典
        targets = {
            'dataset': 'MOT17' if 'MOT17' in img_path else 'crowdhuman',
            'boxes': [],
            'area': [],
            'iscrowd': [],
            'labels': [],
            'obj_ids': [],
            'image_id': torch.as_tensor(idx),
            'size': torch.as_tensor([h, w]),
            'orig_size': torch.as_tensor([h, w])
        }



        # 处理每个标注框
        for label in labels:
            # 转换为xyxy格式
            box = [label[2], label[3], label[4], label[5]]
            area = label[4] * label[5]
            iscrowd = 0  # 假设MOT17数据集中没有crowd标注
            obj_id = label[1] + obj_id_offset  # 相对ID

            targets['boxes'].append(box)
            targets['area'].append(area)
            targets['iscrowd'].append(iscrowd)
            targets['labels'].append(0)
            targets['obj_ids'].append(obj_id)

        # 转换为张量
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        targets['area'] = torch.as_tensor(targets['area'])
        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'])
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'])

        return img, targets

    def _get_sample_range(self, start_idx):

        # 新增范围检查
        max_start = len(self.img_files) - (self.num_frames_per_batch - 1) * self.sample_interval - 1
        start_idx = min(start_idx, max_start)

        # 确定样本范围
        assert self.sample_mode in ['fixed_interval', 'random_interval']
        sample_interval = self.sample_interval if self.sample_mode == 'fixed_interval' else np.random.randint(1,
                                                                                                              self.sample_interval + 1)
        return (start_idx, start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1, sample_interval)

    def pre_continuous_frames(self, start, end, interval=1):
        """预加载连续帧"""
        images = []
        targets_list = []
        for i in range(start, end, interval):
            img, targets = self._pre_single_frame(i)
            images.append(img)
            targets_list.append(targets)
        return images, targets_list

    def __getitem__(self, idx):
        """获取单个样本"""
        sample_start, sample_end, sample_interval = self._get_sample_range(idx)
        images, targets_list = self.pre_continuous_frames(sample_start, sample_end, sample_interval)

        # 应用数据增强
        dataset_name = targets_list[0]['dataset']
        transform = self.dataset2transform.get(dataset_name, None)
        if transform is not None:
            images, targets_list = transform(images, targets_list)

        # 转换为模型输入格式
        gt_instances_list = []
        for img, targets in zip(images, targets_list):
            gt_instances = self._targets_to_instances(targets, img.size)
            gt_instances_list.append(gt_instances)

        # 返回结果
        result = {
            'imgs': images,
            'gt_instances': gt_instances_list
        }
        if self.vis:
            result['ori_img'] = [t['ori_img'] for t in targets_list]
        return result

    def __len__(self):
        return self.item_num


class DetMOTDetectionValidation(DetMOTDetection):
    """验证集数据集类"""

    def __init__(self, args, seqs_folder, dataset2transform):
        args.data_txt_path = args.val_data_txt_path
        super().__init__(args, seqs_folder, dataset2transform)


def make_transforms_for_mot17(image_set, args=None):
    """构建MOT17数据增强流水线"""
    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomHorizontalFlip(),
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([400, 500, 600]),
                    T.FixedMotRandomCrop(384, 600),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            normalize,
        ])
    elif image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])
    else:
        raise ValueError(f'Unknown image set: {image_set}')


def make_transforms_for_crowdhuman(image_set, args=None):
    """构建CrowdHuman数据增强流水线"""
    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomHorizontalFlip(),
            T.FixedMotRandomShift(bs=1),
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([400, 500, 600]),
                    T.FixedMotRandomCrop(384, 600),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            normalize,
        ])
    elif image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])
    else:
        raise ValueError(f'Unknown image set: {image_set}')


def build_dataset2transform(args, image_set):
    """构建数据集到变换的映射"""
    mot17_train = make_transforms_for_mot17('train', args)
    mot17_test = make_transforms_for_mot17('val', args)

    crowdhuman_train = make_transforms_for_crowdhuman('train', args)
    dataset2transform_train = {'MOT17': mot17_train, 'crowdhuman': crowdhuman_train}
    dataset2transform_val = {'MOT17': mot17_test, 'crowdhuman': mot17_test}

    return dataset2transform_train if image_set == 'train' else dataset2transform_val


def build(image_set, args):
    """数据集构建入口函数"""
    root = Path(args.mot_path)
    assert root.exists(), f'MOT path {root} does not exist'

    dataset2transform = build_dataset2transform(args, image_set)

    if image_set == 'train':
        data_txt_path = args.data_txt_path_train
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root,
                                  dataset2transform=dataset2transform)
    elif image_set == 'val':
        data_txt_path = args.data_txt_path_val
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root,
                                  dataset2transform=dataset2transform)
    return dataset