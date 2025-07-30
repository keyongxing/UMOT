

# ------------------------------- 导入依赖库 -------------------------------
from collections import defaultdict  # 使用默认字典存储跟踪数据
from glob import glob  # 文件路径模式匹配
import json  # JSON文件处理
import os  # 操作系统接口
import cv2  # OpenCV图像处理库
import subprocess  # 执行外部命令(用于调用ffmpeg)
from tqdm import tqdm  # 进度条显示
import colorsys

# ------------------------------- 工具函数 -------------------------------
# def get_color(i):
#     """根据跟踪ID生成可视化颜色 (固定算法保证颜色区分度)
#     参数：
#         i : 跟踪目标ID (整数)
#     返回：
#         list: BGR格式的颜色值列表
#     """
#     return [(i * 23 * j + 43) % 255 for j in range(3)]  # 通过线性运算生成颜色

def get_color(i):
    """改进颜色分配方法，使不同ID颜色差异更明显
    参数：
        i : 跟踪目标ID (整数)
    返回：
        list: BGR格式的颜色值列表
    """
    hue = (i * 37) % 360  # 使用较大步长跳跃生成色调，避免颜色过于相似
    rgb = colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0)  # 颜色转换 (HSV -> RGB)
    return [int(c * 255) for c in reversed(rgb)]  # 转换为 BGR 格式


# ------------------------------- 全局配置 -------------------------------
# 加载预生成的检测结果数据库（包含所有图像的检测框信息）
with open("/home/severs-s/kyx_use/pycharm_xinagmu/UMOT2/data/Dataset/mot/det_db_motrv2.json") as f:
    det_db = json.load(f)  # 数据结构: {图像路径.txt: [检测框行1, 检测框行2,...]}


# ------------------------------- 核心函数 -------------------------------
def process(trk_path, img_list, output="output.mp4"):
    """视频渲染主函数：将检测框和跟踪结果绘制到视频帧

    参数：
        trk_path : 跟踪结果文件路径 (MOT格式文本文件)
        img_list : 图像路径列表 (按帧顺序排列)
        output   : 输出视频文件路径
    """
    # 获取首帧图像尺寸（用于ffmpeg配置）
    h, w, _ = cv2.imread(img_list[0]).shape  # 假设所有帧尺寸一致

    # 配置ffmpeg命令参数（将图像序列编码为视频）
    command = [
        "/home/severs-s/anaconda3/envs/kyx_umot/bin/ffmpeg",
        '-y',  # 覆盖已存在的输出文件
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',  # 输入帧尺寸
        '-pix_fmt', 'bgr24',
        '-r', '20',  # 输入帧率20fps
        '-i', '-',  # 从标准输入读取数据
        '-c:v', 'libx264',  # 使用x264编码器
        '-crf', '26',  # 控制输出视频质量
        '-preset', 'fast',  # 编码速度预设
        '-vf', f'scale={w // 2 * 2}:{h // 2 * 2}:flags=lanczos',  # 确保输出宽高为偶数，使用高质量缩放滤镜
        '-pix_fmt', 'yuv420p',
        '-an',  # 不处理音频
        '-loglevel', 'error'
    ]

    # 启动ffmpeg进程并准备写入管道
    writing_process = subprocess.Popen(command + [output], stdin=subprocess.PIPE)

    # ------------------------------- 加载跟踪结果 -------------------------------
    tracklets = defaultdict(list)  # 按帧号组织的跟踪结果 {帧号: [(id, x1, y1, x2, y2), ...]}

    # 解析跟踪结果文件（MOT Challenge格式）
    with open(trk_path) as f:
        for line in f:
            # 格式：帧号, 目标ID, x1, y1, 宽度, 高度, 置信度, -1, -1, -1
            t, id, *xywhs = line.split(',')[:7]  # 取前7个字段
            t, id = map(int, (t, id))  # 转换帧号和ID为整数
            x, y, w_box, h_box, s = map(float, xywhs)  # 解析坐标和尺寸
            # 转换为左上角与右下角坐标
            tracklets[t].append((id, *map(int, (x, y, x + w_box, y + h_box))))

    # ------------------------------- 逐帧渲染 -------------------------------
    for i, path in enumerate(tqdm(sorted(img_list))):  # 按帧顺序处理
        # 读取当前帧图像
        im = cv2.imread(path)

        # 绘制检测框（白色矩形）
        det_key = path.replace('.jpg', '.txt')
        det_key = os.path.relpath(det_key, "/home/severs-s/kyx_use/pycharm_xinagmu/UMOT-main/data/Dataset/mot")
        # if det_key in det_db:  # 检查是否存在检测结果
        #     for det in det_db[det_key]:
        #         # 解析检测框：左上坐标 + 宽高 + 置信度
        #         x1, y1, w_box, h_box, _ = map(int, map(float, det.strip().split(',')))
        #         cv2.rectangle(im, (x1, y1), (x1 + w_box, y1 + h_box), (255, 255, 255), 6)

        # 绘制跟踪框（彩色矩形 + ID标签）
        # 注意：跟踪结果文件中的帧号从1开始计数
        for j, x1, y1, x2, y2 in tracklets[i + 1]:
            color = get_color(j)  # 根据目标ID生成颜色
            cv2.rectangle(im, (x1, y1), (x2, y2), color, 4)
            cv2.putText(im, f"{j}", (x1 + 10, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 将处理后的帧写入视频管道
        writing_process.stdin.write(im.tobytes())


# ------------------------------- 主程序入口 -------------------------------
if __name__ == '__main__':
    # ------------------------------- 配置实际路径 -------------------------------
    tracker_dir = "/home/severs-s/kyx_use/pycharm_xinagmu/UMOT2/track_duoren"  # 跟踪结果目录
    output_dir = "/DEMO/demo_motrv2_duoren__tr"  # 输出视频目录
    img_root = "/home/severs-s/kyx_use/pycharm_xinagmu/UMOT2/data/Dataset/mot/DanceTrack/test"  # 图像根目录




    # ------------------------------- 检查路径有效性 -------------------------------
    if not os.path.exists(tracker_dir):
        raise FileNotFoundError(f"跟踪结果目录不存在: {tracker_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------- 分布式任务分配 -------------------------------
    jobs = os.listdir(tracker_dir)
    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    jobs = sorted(jobs)[rank::ws]

    # ------------------------------- 处理每个序列 -------------------------------

    for seq in jobs:
        print(f"Processing: {seq}")

        # 构建完整的跟踪结果文件路径
        trk_path = os.path.join(tracker_dir, seq)
        # 假设跟踪结果文件名格式为 "XXXX.txt"，则序列名称为 "XXXX"
        seq_name = os.path.splitext(seq)[0]

        # 根据序列名称构建对应图像文件夹路径
        # 注意：与上一个代码一致，这里假设图像目录结构为: {img_root}/{seq_name}/img1/
        img_dir = os.path.join(img_root, seq_name, "img1")
        img_list = sorted(glob(os.path.join(img_dir, "*.jpg")))

        if not img_list:
            print(f"警告: 未找到图像文件 {img_dir}")
            continue

        # 生成视频输出路径
        output_path = os.path.join(output_dir, f"{seq_name}.mp4")
        process(trk_path, img_list, output_path)







#
#
# #umot
# # ------------------------------- 导入依赖库 -------------------------------
# from collections import defaultdict  # 使用默认字典存储跟踪数据
# from glob import glob  # 文件路径模式匹配
# import json  # JSON文件处理
# import os  # 操作系统接口
# import cv2  # OpenCV图像处理库
# import subprocess  # 执行外部命令(用于调用ffmpeg)
# from tqdm import tqdm  # 进度条显示
# import colorsys
#
# # ------------------------------- 工具函数 -------------------------------
# def get_color(i):
#     """改进颜色分配方法，使不同ID颜色差异更明显
#     参数：
#         i : 跟踪目标ID (整数)
#     返回：
#         list: BGR格式的颜色值列表
#     """
#     hue = (i * 37) % 360  # 使用较大步长跳跃生成色调，避免颜色过于相似
#     rgb = colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0)  # 颜色转换 (HSV -> RGB)
#     return [int(c * 255) for c in reversed(rgb)]  # 转换为 BGR 格式
#
#
#
#
#
# # ------------------------------- 全局配置 -------------------------------
# # 加载预生成的检测结果数据库（包含所有图像的检测框信息）
# # with open("/home/severs-s/kyx_use/pycharm_xinagmu/UMOT-main/data/Dataset/mot/det_db_motrv2.json") as f:
# #     det_db = json.load(f)  # 数据结构: {图像路径.txt: [检测框行1, 检测框行2,...]}
#
#
# # ------------------------------- 核心函数 -------------------------------
# def process(trk_path, img_list, output="output.mp4"):
#     """视频渲染主函数：将检测框和跟踪结果绘制到视频帧
#
#     参数：
#         trk_path : 跟踪结果文件路径 (MOT格式文本文件)
#         img_list : 图像路径列表 (按帧顺序排列)
#         output   : 输出视频文件路径
#     """
#     # 获取首帧图像尺寸（用于ffmpeg配置）
#     h, w, _ = cv2.imread(img_list[0]).shape  # 假设所有帧尺寸一致
#
#     # 配置ffmpeg命令参数（将图像序列编码为视频）
#     command = [
#         "/home/severs-s/anaconda3/envs/kyx_umot/bin/ffmpeg",
#         '-y',  # 覆盖已存在的输出文件
#         '-f', 'rawvideo',
#         '-vcodec', 'rawvideo',
#         '-s', f'{w}x{h}',  # 输入帧尺寸
#         '-pix_fmt', 'bgr24',
#         '-r', '20',  # 输入帧率20fps
#         '-i', '-',  # 从标准输入读取数据
#         '-c:v', 'libx264',  # 使用x264编码器
#         '-crf', '26',  # 控制输出视频质量
#         '-preset', 'fast',  # 编码速度预设
#         '-vf', f'scale={w // 2 * 2}:{h // 2 * 2}:flags=lanczos',  # 确保输出宽高为偶数，使用高质量缩放滤镜
#         '-pix_fmt', 'yuv420p',
#         '-an',  # 不处理音频
#         '-loglevel', 'error'
#     ]
#
#     # 启动ffmpeg进程并准备写入管道
#     writing_process = subprocess.Popen(command + [output], stdin=subprocess.PIPE)
#
#     # ------------------------------- 加载跟踪结果 -------------------------------
#     tracklets = defaultdict(list)  # 按帧号组织的跟踪结果 {帧号: [(id, x1, y1, x2, y2), ...]}
#
#     # 解析跟踪结果文件（MOT Challenge格式）
#     with open(trk_path) as f:
#         for line in f:
#             # 格式：帧号, 目标ID, x1, y1, 宽度, 高度, 置信度, -1, -1, -1
#             t, id, *xywhs = line.split(',')[:7]  # 取前7个字段
#             t, id = map(int, (t, id))  # 转换帧号和ID为整数
#             x, y, w_box, h_box, s = map(float, xywhs)  # 解析坐标和尺寸
#             # 转换为左上角与右下角坐标
#             tracklets[t].append((id, *map(int, (x, y, x + w_box, y + h_box))))
#
#     # ------------------------------- 逐帧渲染 -------------------------------
#     for i, path in enumerate(tqdm(sorted(img_list))):  # 按帧顺序处理
#         # 读取当前帧图像
#         im = cv2.imread(path)
#
#         # 绘制检测框（白色矩形）
#         det_key = path.replace('.jpg', '.txt')
#         det_key = os.path.relpath(det_key, "/home/severs-s/kyx_use/pycharm_xinagmu/UMOT-main/data/Dataset/mot")
#         # if det_key in det_db:  # 检查是否存在检测结果
#         #     for det in det_db[det_key]:
#         #         # 解析检测框：左上坐标 + 宽高 + 置信度
#         #         x1, y1, w_box, h_box, _ = map(int, map(float, det.strip().split(',')))
#         #         cv2.rectangle(im, (x1, y1), (x1 + w_box, y1 + h_box), (255, 255, 255), 6)
#
#         # 绘制跟踪框（彩色矩形 + ID标签）
#         # 注意：跟踪结果文件中的帧号从1开始计数
#         for j, x1, y1, x2, y2 in tracklets[i + 1]:
#             color = get_color(j)  # 根据目标ID生成颜色
#             cv2.rectangle(im, (x1, y1), (x2, y2), color, 4)
#             # 将ID标签放置于目标框外面的左上角，向上偏移10个像素
#             label_pos = (x1, y1 - 10 if y1 - 10 > 0 else y1 + 30)
#             cv2.putText(im, f"{j}", label_pos,
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#
#         # 将处理后的帧写入视频管道
#         writing_process.stdin.write(im.tobytes())
#
#
# # ------------------------------- 主程序入口 -------------------------------
# if __name__ == '__main__':
#     # ------------------------------- 配置实际路径 -------------------------------
#     tracker_dir = "/home/severs-s/kyx_use/pycharm_xinagmu/UMOT2/tracker5"  # 跟踪结果目录
#     output_dir = "/home/severs-s/kyx_use/pycharm_xinagmu/UMOT2/demo_test_d20"  # 输出视频目录
#     img_root = "/home/severs-s/kyx_use/pycharm_xinagmu/UMOT2/data/Dataset/mot/DanceTrack/test"  # 图像根目录
#
#
#
#
#     # ------------------------------- 检查路径有效性 -------------------------------
#     if not os.path.exists(tracker_dir):
#         raise FileNotFoundError(f"跟踪结果目录不存在: {tracker_dir}")
#     os.makedirs(output_dir, exist_ok=True)
#
#     # ------------------------------- 分布式任务分配 -------------------------------
#     jobs = os.listdir(tracker_dir)
#     rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
#     ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
#     jobs = sorted(jobs)[rank::ws]
#
#     # ------------------------------- 处理每个序列 -------------------------------
#
#     for seq in jobs:
#         print(f"Processing: {seq}")
#
#         # 构建完整的跟踪结果文件路径
#         trk_path = os.path.join(tracker_dir, seq)
#         # 假设跟踪结果文件名格式为 "XXXX.txt"，则序列名称为 "XXXX"
#         seq_name = os.path.splitext(seq)[0]
#
#         # 根据序列名称构建对应图像文件夹路径
#         # 注意：与上一个代码一致，这里假设图像目录结构为: {img_root}/{seq_name}/img1/
#         img_dir = os.path.join(img_root, seq_name, "img1")
#         img_list = sorted(glob(os.path.join(img_dir, "*.jpg")))
#
#         if not img_list:
#             print(f"警告: 未找到图像文件 {img_dir}")
#             continue
#
#         # 生成视频输出路径
#         output_path = os.path.join(output_dir, f"{seq_name}.mp4")
#         process(trk_path, img_list, output_path)
