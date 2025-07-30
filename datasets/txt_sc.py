import os


def process_gt_file(base_dir):
    """处理单目录下的标注文件，生成对应帧的标签文件

    参数：
        base_dir (str): 包含gt.txt和图像文件的基准目录

    功能：
        1. 读取gt.txt文件中的标注信息
        2. 按帧索引整理边界框数据
        3. 生成符合格式要求的标签文件

    输出：
        在图像目录下创建与帧索引对应的.txt文件
        （文件名为八位数字，如00000001.txt）
    """
    gt_file_path = os.path.join(base_dir, 'gt/gt.txt')  # 标注文件路径
    img_dir = os.path.join(base_dir, 'img1')  # 图像文件目录

    # 使用字典存储每帧的边界框信息
    frame_bboxes = {}

    # 解析标注文件
    with open(gt_file_path, 'r') as gt_file:
        for line in gt_file:
            line = line.strip()  # 去除首尾空白字符
            if not line:  # 跳过空行
                continue
            parts = line.split(',')  # 按逗号分割数据列

            frame_index = int(parts[0])  # 帧序号
            # 提取边界框坐标和置信度
            # GT格式示例：x1,y1,x2,y2,label,confidence
            # 这里只取前四个坐标和第六个置信度
            x1 = float(parts[2])
            y1 = float(parts[3])
            x2 = float(parts[4])
            y2 = float(parts[5])
            confidence = float(parts[6])

            # 将坐标转换为矩形表示（左上角x,y和宽高）
            width = x2 - x1
            height = y2 - y1

            # 按帧索引整理数据
            if frame_index not in frame_bboxes:
                frame_bboxes[frame_index] = []
            frame_bboxes[frame_index].append((x1, y1, width, height, confidence))

    # 生成标签文件
    for frame_idx, bboxes in frame_bboxes.items():
        output_path = os.path.join(img_dir, f"{frame_idx:06}.txt")  # 八位数字格式

        with open(output_path, 'w') as output:
            for bbox in bboxes:
                x, y, w, h, conf = bbox
                # 格式要求：x,y,width,height,confidence
                output_line = f"{x:.1f},{y:.1f},{w:.1f},{h:.1f},{conf:.1f}\n"
                output.write(output_line)


# 主程序入口
if __name__ == "__main__":
    train_dir = '/home/severs-s/kyx_use/pycharm_xinagmu/MOTRv2-main/data/Dataset/mot/MOT17/images/train'  # 数据集验证目录

    # 遍历每个子目录（每个视频序列）
    for subdir in os.listdir(train_dir):
        current_path = os.path.join(train_dir, subdir)

        # 检查是否为有效目录（包含gt.txt）
        if os.path.isdir(current_path) and os.path.exists(os.path.join(current_path, 'gt/gt.txt')):
            print(f"正在处理目录：{current_path}")
            process_gt_file(current_path)

    print("所有标签文件生成完成！")