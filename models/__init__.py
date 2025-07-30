# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

# 导入当前包中motr模块的build函数，重命名为build_motr
# MOTR是Deformable DETR的改进版本，用于多目标跟踪任务
from .motr import build as build_motr


def build_model(args):
    """
    模型构建工厂函数，根据配置参数动态创建指定架构的模型

    Args:
        args (argparse.Namespace):
            包含模型配置参数的对象，必须包含以下属性：
            - meta_arch (str): 指定模型架构名称，当前支持'motr'

    Returns:
        torch.nn.Module: 实例化的PyTorch模型

    Raises:
        AssertionError: 当传入不支持的架构名称时触发

    Example:
        >>> from argparse import Namespace
        >>> args = Namespace(meta_arch='motr')
        >>> model = build_model(args)

    [架构扩展说明]
    新增架构步骤：
    1. 在对应模块实现build函数（如新建faster_rcnn.py并实现build_faster_rcnn）
    2. 在本字典添加键值对，例如：'faster_rcnn': build_faster_rcnn
    3. 通过args.meta_arch参数指定使用新架构
    """

    # 架构构建器注册表（Factory Registry）
    # 键: 架构名称字符串
    # 值: 对应的模型构建函数
    arch_catalog = {
        'motr': build_motr,  # MOTR: Motion Transformer 模型

    }

    # 架构有效性校验
    # 当检测到未注册的架构名称时，触发包含详细错误信息的断言异常
    # 使用显式字符串格式化提高错误可读性（Python 3.6+ f-string）
    assert args.meta_arch in arch_catalog, (
        f"不支持的模型架构: '{args.meta_arch}'. "
        f"当前支持的架构列表: {list(arch_catalog.keys())}"
    )

    # 从注册表中获取对应的模型构建函数
    build_func = arch_catalog[args.meta_arch]

    # 调用构建函数并返回实例化后的模型
    # 注：此处将args参数透传给具体构建函数，用于传递模型超参数
    return build_func(args)
