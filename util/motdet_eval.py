
import torch
import numpy as np
import time
import cv2


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ 计算每个类别的平均精度（Average Precision）
    Args:
        tp:     True Positive列表（布尔数组）
        conf:   置信度数组（0-1之间的值）
        pred_cls: 预测类别数组
        target_cls: 真实类别数组
    Returns:
        ap: 各类别平均精度
        unique_classes: 唯一类别列表
        r: 召回率数组
        p: 精确率数组
    """
    # 转换输入为numpy数组
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf), np.array(pred_cls), np.array(target_cls)

    # 按置信度降序排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 获取所有唯一类别（合并预测和真实类别）
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    ap, p, r = [], [], []  # 初始化结果列表
    for c in unique_classes:  # 遍历每个类别
        # 获取当前类别的预测索引
        i = pred_cls == c

        # 统计真实和预测数量
        n_gt = sum(target_cls == c)  # 真实目标数
        n_p = sum(i)  # 预测目标数

        # 处理特殊情况
        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # 累积计算FP和TP
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # 计算召回率曲线
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # 计算精确率曲线
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # 计算AP
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """ 根据召回率和精确率曲线计算平均精度（AP）
    Args:
        recall: 召回率数组
        precision: 精确率数组
    Returns:
        ap: 平均精度值
    """
    # 在曲线两端添加哨兵值
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # 计算精确率包络线（确保单调递减）
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 找到召回率变化的点
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # 计算AP（曲线下面积）
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=False):
    """ 计算两组边界框之间的IoU矩阵
    Args:
        box1: 边界框数组1（N x 4）
        box2: 边界框数组2（M x 4）
        x1y1x2y2: 是否使用左上右下坐标格式
    Returns:
        IoU矩阵（N x M）
    """
    N, M = len(box1), len(box2)
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # 将中心坐标转换为左上右下坐标
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # 计算交集区域
    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)

    # 计算并集区域
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).view(-1, 1).expand(N, M)
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).view(1, -1).expand(N, M)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def xyxy2xywh(x):
    """ 将边界框格式从 [x1,y1,x2,y2] 转换为 [x_center,y_center,width,height] """
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    """ 将边界框格式从 [x_center,y_center,width,height] 转换为 [x1,y1,x2,y2] """
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


@torch.no_grad()
def motdet_evaluate(model, data_loader, iou_thres=0.5, print_interval=10):
    """ 单阶段目标检测模型评估函数
    Args:
        model: 待评估的模型
        data_loader: 数据加载器
        iou_thres: IoU阈值
        print_interval: 打印间隔
    Returns:
        mean_mAP: 平均精度均值
        mean_R: 平均召回率
        mean_P: 平均精确率
    """
    model.eval()
    # 初始化评估指标
    mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    # 初始化各种存储容器
    outputs, mAPs, mR, mP, TP, confidence, pred_class, target_class, jdict = \
        [], [], [], [], [], [], [], [], []
    AP_accum, AP_accum_count = np.zeros(1), np.zeros(1)

    for batch_i, data in enumerate(data_loader):
        seen += 1
        if batch_i > 300:  # 限制评估批次数量（调试用）
            break

        # 数据准备
        imgs, _ = data[0].decompose()
        targets = data[1][0]
        height, width = targets['orig_size'].cpu().numpy().tolist()

        # 模型推理
        t = time.time()
        output = model(imgs.cuda())

        # 解析模型输出
        outputs_class = output['pred_logits'].squeeze()
        if outputs_class.ndim == 1:
            outputs_class = outputs_class.unsqueeze(-1)
        outputs_boxes = output['pred_boxes'].squeeze()
        target_boxes = targets['boxes']

        # 处理无目标的情况
        if target_boxes.size(0) == 0:
            mAPs.append(0), mR.append(0), mP.append(0)
            continue

        # 坐标转换
        target_cls = targets['labels']
        target_boxes = xywh2xyxy(target_boxes)
        target_boxes[:, 0] *= width
        target_boxes[:, 2] *= width
        target_boxes[:, 1] *= height
        target_boxes[:, 3] *= height

        outputs_boxes = xywh2xyxy(outputs_boxes)
        outputs_boxes[:, 0] *= width
        outputs_boxes[:, 2] *= width
        outputs_boxes[:, 1] *= height
        outputs_boxes[:, 3] *= height

        # 匹配检测结果与真实框
        detected = set()
        correct = []
        for *pred_bbox, conf in zip(outputs_boxes, outputs_class):
            pred_bbox = torch.FloatTensor(pred_bbox[0]).view(1, -1)
            # 计算IoU
            iou = bbox_iou(pred_bbox, target_boxes, x1y1x2y2=True)[0]
            best_i = np.argmax(iou)
            # 判断是否匹配成功
            if iou[best_i] > iou_thres and best_i.item() not in detected:
                correct.append(1)
                detected.add(best_i.item())
            else:
                correct.append(0)

        # 计算评估指标
        AP, AP_class, R, P = ap_per_class(tp=correct,
                                          conf=outputs_class[:, 0].cpu(),
                                          pred_cls=np.zeros_like(outputs_class[:, 0].cpu()),
                                          target_cls=target_cls)
        # 更新累加器
        AP_accum_count += np.bincount(AP_class, minlength=1)
        AP_accum += np.bincount(AP_class, minlength=1, weights=AP)
        # 更新指标列表
        mAPs.append(AP.mean())
        mR.append(R.mean())
        mP.append(P.mean())

        # 计算运行平均值
        mean_mAP = np.sum(mAPs) / (AP_accum_count + 1E-16)
        mean_R = np.sum(mR) / (AP_accum_count + 1E-16)
        mean_P = np.sum(mP) / (AP_accum_count + 1E-16)

        # 定期打印结果
        if batch_i % print_interval == 0:
            print(('%11s%11s' + '%11.3g' * 4 + 's') %
                  (seen, 100, mean_P, mean_R, mean_mAP, time.time() - t))

    # 最终输出结果
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    print('AP: %-.4f\n\n' % (AP_accum[0] / (AP_accum_count[0] + 1E-16)))
    return mean_mAP, mean_R, mean_P


def init_metrics():
    """ 初始化评估指标容器 """
    return {
        'mean_mAP': 0.0,
        'mean_R': 0.0,
        'mean_P': 0.0,
        'seen': 0,
        'outputs': [],
        'mAPs': [],
        'mR': [],
        'mP': [],
        'TP': [],
        'confidence': [],
        'pred_class': [],
        'target_class': [],
        'jdict': [],
        'AP_accum': np.zeros(1),
        'AP_accum_count': np.zeros(1),
    }


@torch.no_grad()
def detmotdet_evaluate(model, data_loader, device, iou_thres=0.5, print_interval=10):
    """ 多目标跟踪检测评估函数（支持多帧评估）
    Args:
        model: 待评估模型
        data_loader: 数据加载器
        device: 计算设备
        iou_thres: IoU阈值
        print_interval: 打印间隔
    Returns:
        ret: 评估结果列表 [mAP, R, P] * 2
    """
    model.eval()
    print('%11s' * 5 % ('Cur Image', 'Total', 'P', 'R', 'mAP'))
    metrics_list = [init_metrics() for _ in range(10)]  # 初始化多个时间步的指标

    for batch_i, data in enumerate(data_loader):
        if batch_i > 100:  # 限制评估批次数量
            break

        # 数据转移至设备
        for key in data:
            if isinstance(data[key], list):
                data[key] = [img_info.to(device) for img_info in data[key]]
            else:
                data[key] = data[key].to(device)

        # 模型推理
        output = model(data)
        num_frames = len(data['gt_instances'])

        # 逐帧处理
        for i in range(num_frames):
            metrics_i = metrics_list[i]
            metrics_i['seen'] += 1
            gt_instances = data['gt_instances'][i].cpu()

            # 解析模型输出
            outputs_class = output['pred_logits'][i].squeeze()
            outputs_boxes = output['pred_boxes'][i].squeeze()

            # 处理不同维度的输出
            if outputs_class.ndim == 1:
                outputs_class = outputs_class.unsqueeze(-1)

            # 获取真实框信息
            target_boxes = gt_instances.boxes
            target_cls = gt_instances.labels
            height, width = gt_instances.image_size

            # 坐标转换和缩放
            target_boxes = xywh2xyxy(target_boxes)
            target_boxes[:, 0] *= width
            target_boxes[:, 2] *= width
            target_boxes[:, 1] *= height
            target_boxes[:, 3] *= height

            outputs_boxes = xywh2xyxy(outputs_boxes)
            outputs_boxes[:, 0] *= width
            outputs_boxes[:, 2] *= width
            outputs_boxes[:, 1] *= height
            outputs_boxes[:, 3] *= height

            # 匹配检测结果与真实框
            detected = []
            correct = []
            for *pred_bbox, conf in zip(outputs_boxes, outputs_class):
                pred_bbox = torch.FloatTensor(pred_bbox[0]).view(1, -1)
                iou = bbox_iou(pred_bbox, target_boxes, x1y1x2y2=True)[0]
                best_i = np.argmax(iou)
                if iou[best_i] > iou_thres and best_i not in detected:
                    correct.append(1)
                    detected.append(best_i)
                else:
                    correct.append(0)

            # 计算评估指标
            AP, AP_class, R, P = ap_per_class(tp=correct,
                                              conf=outputs_class[:, 0].cpu(),
                                              pred_cls=np.zeros_like(outputs_class[:, 0].cpu()),
                                              target_cls=target_cls)
            # 更新指标
            metrics_i['AP_accum_count'] += np.bincount(AP_class, minlength=1)
            metrics_i['AP_accum'] += np.bincount(AP_class, minlength=1, weights=AP)
            metrics_i['mAPs'].append(AP.mean())
            metrics_i['mR'].append(R.mean())
            metrics_i['mP'].append(P.mean())

            # 计算运行平均值
            metrics_i['mean_mAP'] = np.sum(metrics_i['mAPs']) / (metrics_i['AP_accum_count'] + 1E-16)
            metrics_i['mean_R'] = np.sum(metrics_i['mR']) / (metrics_i['AP_accum_count'] + 1E-16)
            metrics_i['mean_P'] = np.sum(metrics_i['mP']) / (metrics_i['AP_accum_count'] + 1E-16)
