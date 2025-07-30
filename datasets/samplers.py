


import os
import math
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class DistributedSampler(Sampler):
    """标准分布式数据采样器

    参数：
        dataset (Dataset): 数据集实例
        num_replicas (int, optional): 参与训练的进程总数
        rank (int, optional): 当前进程的全局编号
        shuffle (bool, optional): 是否启用shuffle

    特性：
        - 自动适应不同规模的分布式环境
        - 支持按epoch动态洗牌
        - 确保数据均匀分配

    方法：
        __iter__: 生成当前进程的数据索引迭代器
        set_epoch: 设置当前epoch编号（用于洗牌控制）
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        # 初始化分布式环境参数
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("分布式训练需要torch.distributed支持")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("分布式训练需要torch.distributed支持")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle

        # 计算每个进程应处理的样本数量
        self.num_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas  # 总样本量（可能包含冗余）

    def __iter__(self):
        """生成当前进程的样本索引迭代器"""
        if self.shuffle:
            # 基于epoch的确定性洗牌
            gen = torch.Generator()
            gen.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=gen).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # 补充样本使总长度达到total_size
        indices += indices[:self.total_size - len(indices)]
        assert len(indices) == self.total_size

        # 获取当前进程的样本范围
        start = self.num_samples * self.rank
        end = start + self.num_samples
        return iter(indices[start:end])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """设置当前epoch编号（用于控制洗牌种子）"""
        self.epoch = epoch


class NodeDistributedSampler(Sampler):
    """节点级分布式数据采样器

    参数：
        dataset (Dataset): 数据集实例
        num_replicas (int, optional): 参与训练的进程总数
        rank (int, optional): 当前进程的全局编号
        local_rank (int, optional): 当前进程在节点内的局部编号
        local_size (int, optional): 每个节点包含的进程数
        shuffle (bool, optional): 是否启用shuffle

    特性：
        - 支持节点级数据划分（local_size控制节点规模）
        - 更细粒度的数据分配策略
        - 保持与DistributedSampler相同的接口

    方法：
        __iter__: 生成当前进程的数据索引迭代器
        set_epoch: 设置当前epoch编号（用于洗牌控制）
    """

    def __init__(self, dataset, num_replicas=None, rank=None, local_rank=None, local_size=None, shuffle=True):
        # 初始化分布式环境参数
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("分布式训练需要torch.distributed支持")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("分布式训练需要torch.distributed支持")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.local_rank = local_rank or int(os.environ.get('LOCAL_RANK', 0))
        self.local_size = local_size or int(os.environ.get('LOCAL_SIZE', 1))
        self.shuffle = shuffle

        # 计算每个进程应处理的样本数量
        self.num_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.total_size_parts = self.num_samples * self.num_replicas // self.local_size  # 节点内总样本量

    def __iter__(self):
        """生成当前进程的样本索引迭代器"""
        if self.shuffle:
            # 基于epoch的确定性洗牌
            gen = torch.Generator()
            gen.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=gen).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # 筛选出属于当前节点的索引
        indices = [i for i in indices if i % self.local_size == self.local_rank]

        # 补充样本使总长度达到total_size_parts
        indices += indices[:self.total_size_parts - len(indices)]
        assert len(indices) == self.total_size_parts

        # 计算全局起始位置
        global_start = (self.rank // self.local_size) * self.num_samples
        return iter(indices[global_start:global_start + self.num_samples])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """设置当前epoch编号（用于控制洗牌种子）"""
        self.epoch = epoch
