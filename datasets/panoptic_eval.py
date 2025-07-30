

import json
import os

import util.misc as utils

try:
    from panopticapi.evaluation import pq_compute
except ImportError:
    pass  # 若评估环境缺失可选依赖则静默跳过


class PanopticEvaluator(object):
    """多目标分割任务评估器类

    主要功能：
        - 收集预测结果并保存为标准格式
        - 多进程训练环境下的结果同步
        - 调用官方评估工具计算Panoptic Quality指标

    参数：
        ann_file (str): 真实标注文件路径
        ann_folder (str): 真实标注文件夹路径
        output_dir (str): 预测结果输出目录（默认'panoptic_eval'）
    """

    def __init__(self, ann_file, ann_folder, output_dir="panoptic_eval"):
        self.gt_json = ann_file  # 真实标注JSON文件
        self.gt_folder = ann_folder  # 真实标注图片文件夹
        # 只有主进程才创建输出目录
        if utils.is_main_process():
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
        self.output_dir = output_dir
        self.predictions = []  # 存储所有预测结果

    def update(self, predictions):
        """接收并存储预测结果

        参数：
            predictions (list): 包含预测结果的字典列表
                每个字典包含：
                    'file_name' (str): 图像文件名
                    'png_string' (bytes): PNG格式的预测掩膜

        功能：
            - 将预测结果按文件名保存到输出目录
            - 清理预测字典中的临时字段
        """
        for p in predictions:
            # 保存PNG掩膜到文件
            with open(os.path.join(self.output_dir, p["file_name"]), "wb") as f:
                f.write(p.pop("png_string"))  # 移除临时字段

            self.predictions.append(p)  # 保留处理后的预测结果

    def synchronize_between_processes(self):
        """多进程同步预测结果

        功能：
            - 收集所有进程的预测结果
            - 合并到主进程的预测列表
        """
        # 使用all_gather收集所有进程的预测数据
        all_predictions = utils.all_gather(self.predictions)
        # 展开合并所有结果
        merged_predictions = []
        for p_list in all_predictions:
            merged_predictions.extend(p_list)
        self.predictions = merged_predictions

    def summarize(self):
        """生成评估报告并计算PQ指标

        返回：
            float: Panoptic Quality指标值（仅主进程有效）

        功能：
            - 将预测结果转换为JSON格式
            - 调用官方评估工具pq_compute计算指标
        """
        if not utils.is_main_process():
            return None  # 非主进程不执行

        # 构建预测结果JSON数据
        json_data = {"annotations": self.predictions}
        predictions_json = os.path.join(self.output_dir, "predictions.json")
        with open(predictions_json, "w") as f:
            f.write(json.dumps(json_data))

        # 调用Panoptic API计算PQ指标
        return pq_compute(
            self.gt_json,  # 真实标注文件
            predictions_json,  # 预测结果文件
            gt_folder=self.gt_folder,  # 真实图像文件夹
            pred_folder=self.output_dir  # 预测图像文件夹
        )
        return None
