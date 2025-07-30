
import torch
from functools import partial
from models.structures import Instances


def to_cuda(samples, targets, device):
    """将数据样本和目标转移到GPU（异步操作）"""
    samples = samples.to(device, non_blocking=True)  # 异步转移样本数据
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]  # 异步转移目标数据
    return samples, targets


def tensor_to_cuda(tensor: torch.Tensor, device):
    """将单个张量转移到指定设备（异步操作）"""
    return tensor.to(device)


def is_tensor_or_instances(data):
    """检查数据是否为张量或Instances对象"""
    return isinstance(data, torch.Tensor) or isinstance(data, Instances)


def data_apply(data, check_func, apply_func):
    """递归遍历数据结构并应用转换函数"""
    if isinstance(data, dict):
        for k in data.keys():
            if check_func(data[k]):
                data[k] = apply_func(data[k])
            elif isinstance(data[k], dict) or isinstance(data[k], list):
                data_apply(data[k], check_func, apply_func)
            else:
                raise ValueError()
    elif isinstance(data, list):
        for i in range(len(data)):
            if check_func(data[i]):
                data[i] = apply_func(data[i])
            elif isinstance(data[i], dict) or isinstance(data[i], list):
                data_apply(data[i], check_func, apply_func)
            else:
                raise ValueError("invalid type {}".format(type(data[i])))
    else:
        raise ValueError("invalid type {}".format(type(data)))
    return data


def data_dict_to_cuda(data_dict, device):
    """将字典中的所有张量和Instances对象转移到GPU"""
    return data_apply(data_dict, is_tensor_or_instances, partial(tensor_to_cuda, device=device))


class data_prefetcher():
    """数据预取器类（实现异步数据加载）"""

    def __init__(self, loader, device, prefetch=True):
        """初始化预取器

        参数:
            loader (iter): 数据加载器迭代器
            device (str/int): 目标设备（'cuda'或0等）
            prefetch (bool): 是否启用预取功能
        """
        self.loader = iter(loader)  # 创建数据加载器迭代器
        self.prefetch = prefetch  # 是否启用预取
        self.device = device  # 目标设备
        if prefetch:
            self.stream = torch.cuda.Stream()  # 创建CUDA流
            self.preload()  # 开始预取数据

    def preload(self):
        """预加载下一批数据到GPU（异步操作）"""
        try:
            self.next_samples, self.next_targets = next(self.loader)  # 获取下一批数据
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return

        # 使用CUDA流异步执行数据转移
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)

            # 注释掉的替代方案（根据实际情况选择使用）：
            # # 方案1：记录流（可能需要特定PyTorch版本支持）
            # self.next_samples.record_stream(torch.cuda.current_stream())
            # self.next_targets.record_stream(torch.cuda.current_stream())

            # # 方案2：手动复制内存（适用于更复杂的类型）
            # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
            # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)

            # # 如果使用混合精度训练（FP16）
            # # if args.fp16:
            # #     self.next_input = self.next_input.half()
            # # else:
            # #     pass

    def next(self):
        """获取下一个数据批次（自动完成预取）"""
        if self.prefetch:
            # 等待预取操作完成
            torch.cuda.current_stream().wait_stream(self.stream)

            # 强制记录当前流，确保数据在正确的流中被使用
            if self.next_samples is not None:
                self.next_samples.record_stream(torch.cuda.current_stream())
            if self.next_targets is not None:
                for t in self.next_targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())

            # 继续预取下一批数据
            self.preload()
        else:
            # 同步加载数据（无预取）
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                print("catch_stop_iter")
                samples = None
                targets = None

        return samples, targets