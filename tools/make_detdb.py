
from glob import glob  # 文件路径模式匹配
import json  # JSON格式处理
from concurrent.futures import ThreadPoolExecutor  # 线程池执行器
from threading import Lock  # 线程互斥锁

from tqdm import tqdm  # 进度条显示

# 初始化全局检测数据库字典（存储所有文件内容）
det_db = {}
# 待缓存文件路径列表（包含来自多个数据集的.txt文件路径）
to_cache = []

# # 收集CrowdHuman数据集检测文件路径（格式：/data2/Dataset/mot/crowdhuman/train_image/*.txt）
# for file in glob("/data2/Dataset/mot/crowdhuman/train_image/*.txt"):
#     to_cache.append(file)

# 收集DanceTrack数据集检测文件路径（格式：/MOTRv2-main/data2/Dataset/mot/DanceTrack_2/*/*/img1/*.txt）
for file in glob("/home/severs-s/kyx_use/pycharm_xinagmu/MOTRv2-main/data/Dataset/mot/DanceTrack/*/*/img1/*.txt"):
    to_cache.append(file)

# # 收集MOT17数据集检测文件路径（格式：/data2/Dataset/mot/MOT17/Images/*/*/img1/*.txt）
# for file in glob("/data2/Dataset/mot/MOT17/Images/*/*/img1/*.txt"):
#     to_cache.append(file)
#
# # 收集MOT20数据集检测文件路径（格式：/data2/Dataset/mot/MOT20/train/*/img1/*.txt）
# for file in glob("/data2/Dataset/mot/MOT20/train/*/img1/*.txt"):
#     to_cache.append(file)
#
# # 收集HIE20数据集检测文件路径（格式：/data2/Dataset/mot/HIE20/train/*/img1/*.txt）
# for file in glob("/data2/Dataset/mot/HIE20/train/*/img1/*.txt"):
#     to_cache.append(file)

# 初始化进度条（总任务数为待处理文件数量）
pbar = tqdm(total=len(to_cache))

# 创建线程互斥锁（保证字典写入操作的线程安全）
mutex = Lock()


def cache(file):
    """文件内容缓存函数（多线程执行）

    Args:
        file (str): 待处理的文件路径
    """
    # 读取文件内容到临时列表（自动关闭文件）
    with open(file) as f:
        tmp = [l for l in f]  # 逐行读取存储为列表

    # 使用互斥锁保护共享资源（det_db字典和进度条更新）
    with mutex:
        det_db[file] = tmp  # 以文件路径为键存储内容
        pbar.update(1)  # 更新进度条（+1）


# 创建线程池执行器（最大48个并发线程）
with ThreadPoolExecutor(max_workers=48) as exe:
    # 提交所有文件处理任务到线程池
    for file in to_cache:
        exe.submit(cache, file)  # 将每个文件交给cache函数处理

# 将所有缓存数据写入JSON文件（永久存储）
with open("/home/severs-s/kyx_use/pycharm_xinagmu/MOTRv2-main/data/Dataset/mot/det_db_dance.json",
          'w') as f:
    json.dump(det_db, f)  # 序列化字典为JSON格式