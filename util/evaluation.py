
import os
import numpy as np
import copy
import motmetrics as mm  # MOT评估指标计算库

mm.lap.default_solver = 'lap'  # 设置线性分配问题求解器
import logging
from typing import Dict


def read_results(filename, data_type: str, is_gt=False, is_ignore=False):
    """读取跟踪结果文件
    Args:
        filename: 文件路径
        data_type: 数据类型，支持'mot'或'lab'
        is_gt: 是否是真实标注数据
        is_ignore: 是否是忽略区域数据
    Returns:
        结果字典 {帧ID: [(坐标, ID, 分数), ...]}
    """
    if data_type in ('mot', 'lab'):
        return read_mot_results(filename, is_gt, is_ignore)
    else:
        raise ValueError(f'未知数据类型: {data_type}')


def read_mot_results(filename, is_gt, is_ignore):
    """读取MOT格式的结果文件
    Args:
        filename: 结果文件路径
        is_gt: 是否处理真实标注
        is_ignore: 是否处理忽略区域
    Returns:
        按帧ID组织的检测结果字典
    """
    # 有效标签集合（1表示行人）
    valid_labels = {1}
    # 需要忽略的标签集合（0: 静态物体, 2: 遮挡区域等）
    ignore_labels = {0, 2, 7, 8, 12}
    results_dict = dict()

    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')
                if len(linelist) < 7:  # 跳过无效行
                    continue

                # 解析帧号
                fid = int(linelist[0])
                if fid < 1:  # 帧号从1开始
                    continue

                # 初始化帧条目
                results_dict.setdefault(fid, [])

                # 处理真实标注
                if is_gt:
                    # MOT16/17的特殊处理
                    if 'MOT16-' in filename or 'MOT17-' in filename:
                        label = int(float(linelist[7]))  # 第8列是标签
                        mark = int(float(linelist[6]))  # 第7列是可见性标记
                        # 过滤无效标注（可见性为0或非有效标签）
                        if mark == 0 or label not in valid_labels:
                            continue
                    score = 1  # 真值分数设为1

                # 处理忽略区域
                elif is_ignore:
                    # MOT16/17的特殊处理
                    if 'MOT16-' in filename or 'MOT17-' in filename:
                        label = int(float(linelist[7]))
                        vis_ratio = float(linelist[8])  # 可见比例
                        # 保留需要忽略的标签且可见度足够的区域
                        if label not in ignore_labels and vis_ratio >= 0:
                            continue
                    # MOT15的特殊处理
                    elif 'MOT15' in filename:
                        label = int(float(linelist[6]))
                        if label not in ignore_labels:
                            continue
                    else:
                        continue
                    score = 1  # 忽略区域分数设为1

                # 处理检测结果
                else:
                    score = float(linelist[6])  # 第7列为检测置信度

                # 解析坐标（top-left width height）
                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])  # 第2列是目标ID

                # 添加到结果字典
                results_dict[fid].append((tlwh, target_id, score))

    return results_dict


def unzip_objs(objs):
    """将对象列表解压为坐标、ID、分数的数组
    Args:
        objs: [(tlwh, id, score), ...]
    Returns:
        tlwhs: Nx4数组
        ids: N维数组
        scores: N维数组
    """
    if len(objs) > 0:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)
    return tlwhs, ids, scores


class Evaluator(object):
    """跟踪结果评估器"""

    def __init__(self, data_root, seq_name, data_type='mot'):
        """
        Args:
            data_root: 数据集根目录
            seq_name: 序列名称
            data_type: 数据类型，默认为'mot'
        """
        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type

        # 加载标注数据
        self.load_annotations()
        # 初始化指标累加器
        self.reset_accumulator()

    def load_annotations(self):
        """加载真实标注和忽略区域标注"""
        assert self.data_type == 'mot'

        # 真实标注文件路径
        gt_filename = os.path.join(self.data_root, self.seq_name, 'gt', 'gt.txt')
        # 读取真实标注 {帧ID: [obj1, obj2...]}
        self.gt_frame_dict = read_results(gt_filename, self.data_type, is_gt=True)
        # 读取忽略区域 {帧ID: [ignore_obj1...]}
        self.gt_ignore_frame_dict = read_results(gt_filename, self.data_type, is_ignore=True)

    def reset_accumulator(self):
        """重置指标累加器"""
        self.acc = mm.MOTAccumulator(auto_id=True)  # 自动生成唯一ID

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, rtn_events=False):
        """单帧评估
        Args:
            frame_id: 当前帧号
            trk_tlwhs: 跟踪框坐标列表
            trk_ids: 跟踪ID列表
            rtn_events: 是否返回事件详情
        Returns:
            匹配事件列表（如果rtn_events=True）
        """
        # 转换为numpy数组
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        # 获取当前帧的真值
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]  # 解压坐标和ID

        # 获取忽略区域
        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = unzip_objs(ignore_objs)[0]

        # 过滤与忽略区域重叠的跟踪框
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        if len(ignore_tlwhs) > 0:
            # 计算忽略区域与跟踪框的IoU
            iou_distance = mm.distances.iou_matrix(ignore_tlwhs, trk_tlwhs, max_iou=0.5)
            # 使用匈牙利算法进行匹配
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            # 过滤IoU>0.5的匹配对
            match_js = match_js[iou_distance[match_is, match_js] <= 0.5]
            keep[match_js] = False

        # 应用过滤
        trk_tlwhs = trk_tlwhs[keep]
        trk_ids = trk_ids[keep]

        # 计算真值与跟踪框的IoU矩阵
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

        # 更新评估指标累加器
        self.acc.update(gt_ids, trk_ids, iou_distance)

        # 返回匹配事件详情
        if rtn_events and hasattr(self.acc, 'last_mot_events'):
            return self.acc.last_mot_events
        return None

    def eval_file(self, filename):
        """评估整个结果文件
        Args:
            filename: 跟踪结果文件路径
        Returns:
            指标累加器
        """
        self.reset_accumulator()

        # 读取跟踪结果
        result_frame_dict = read_results(filename, self.data_type, is_gt=False)
        # 合并所有存在的帧号
        frames = sorted(set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys()))

        # 逐帧处理
        for frame_id in frames:
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            self.eval_frame(frame_id, trk_tlwhs, trk_ids)

        return self.acc

    @staticmethod
    def get_summary(accs, names, metrics=('mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')):
        """生成评估摘要
        Args:
            accs: 多个累加器列表
            names: 对应名称列表
            metrics: 需要计算的指标
        Returns:
            pandas DataFrame格式的摘要
        """
        # 深拷贝避免修改原参数
        names = copy.deepcopy(names)
        metrics = copy.deepcopy(metrics or mm.metrics.motchallenge_metrics)

        # 创建指标计算器
        mh = mm.metrics.create()
        # 计算指标
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True  # 生成总体统计
        )
        return summary

    @staticmethod
    def save_summary(summary, filename):
        """保存评估结果到Excel文件
        Args:
            summary: 评估摘要DataFrame
            filename: 输出文件路径
        """
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer, index=True)
        writer.save()

