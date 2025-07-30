# import json
# import os
# from concurrent.futures import ThreadPoolExecutor
# from threading import Lock
# from tqdm import tqdm
#
# # 配置路径：输出根目录使用正确的路径
# json_path = "/home/severs-s/kyx_use/pycharm_xinagmu/UMOT2/data/Dataset/mot/det_db_motrv2.json"
# output_base = "/home/severs-s/kyx_use/pycharm_xinagmu/UMOT2/data/Dataset/mot/DanceTrack"
#
# # 加载缓存文件
# with open(json_path, 'r') as f:
#     det_db = json.load(f)
#
# # 过滤并转换路径，生成映射字典
# # 对于路径中包含 "DanceTrack" 的条目，原路径保持不变，
# # 同时如果文件名为 "det.txt"，在同目录下额外生成一个 "00000001.txt" 的文件
# dance_entries = {}
# for abs_path, content in det_db.items():
#     if "DanceTrack" in abs_path:
#         # 提取 "DanceTrack" 后面的相对路径部分，例如：train/0001/img1/det.txt
#         rel_part = abs_path.split("DanceTrack", 1)[1].lstrip(os.sep)
#         parts = rel_part.split(os.sep)
#         # 原始文件路径（保持原文件名）
#         new_path_original = os.path.join(output_base, *parts)
#         dance_entries[new_path_original] = content
#
#         # 如果文件名为 "det.txt"，则在同目录下额外生成 "00000001.txt"
#         if parts and parts[-1].lower() == "det.txt":
#             new_parts = parts.copy()
#             new_parts[-1] = "00000001.txt"
#             new_path_extra = os.path.join(output_base, *new_parts)
#             dance_entries[new_path_extra] = content
#
# # 初始化进度条和互斥锁
# pbar = tqdm(total=len(dance_entries))
# mutex = Lock()
#
#
# def restore_entry(entry):
#     try:
#         file_path, content_lines = entry
#         if pbar.n < 5:
#             print(f"Generating: {file_path}")
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         with open(file_path, 'w', newline='') as f:
#             f.writelines(content_lines)
#     except Exception as e:
#         print(f"\nError processing {file_path}: {str(e)}")
#     finally:
#         with mutex:
#             pbar.update(1)
#
#
# print(f"[Debug] Found {len(dance_entries)} DanceTrack entries")
# sample_paths = list(dance_entries.keys())[:3]
# print("[Debug] Sample output paths:")
# for p in sample_paths:
#     print(p.replace(output_base, "$DANCE_BASE"))
#
# with ThreadPoolExecutor(max_workers=48) as executor:
#     executor.map(restore_entry, dance_entries.items())
#
# print("\n验证生成结果:")
# # 示例验证两个文件是否生成：
# test_file1 = os.path.join(output_base, "train/0001/img1/det.txt")
# test_file2 = os.path.join(output_base, "train/0001/img1/00000001.txt")
# print(f"检查原始示例文件是否存在: {os.path.exists(test_file1)}")
# print(f"检查新增示例文件是否存在: {os.path.exists(test_file2)}")

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from tqdm import tqdm

# 配置路径
json_path = "/home/severs-s/kyx_use/pycharm_xinagmu/UMOT2/data/Dataset/mot/det_db_motrv2.json"
output_base = "/home/severs-s/kyx_use/pycharm_xinagmu/UMOT2/data/Dataset/mot/MOT17"

# 路径解析正则表达式 (新增对images目录的识别)
mot_pattern = re.compile(
    r".*MOT17[/\\](images)[/\\](train|test)[/\\](MOT17-\d{2}-\w+)[/\\]img1[/\\](.*\.txt)",
    re.IGNORECASE
)

# 加载缓存文件
with open(json_path, 'r') as f:
    det_db = json.load(f)

# 构建路径映射字典
mot_entries = {}
for src_path, content in det_db.items():
    match = mot_pattern.search(src_path.replace("MOT17images", "MOT17/images"))  # 修复路径分隔
    if match:
        # 解析路径组件
        media_type = match.group(1)  # images
        split_type = match.group(2)  # train/test
        seq_name = match.group(3)  # MOT17-01-SDP
        filename = match.group(4)  # 000001.txt 或 det.txt

        # 构建标准路径
        new_path = os.path.join(
            output_base,
            media_type,
            split_type,
            seq_name,
            "img1",
            filename
        )
        mot_entries[new_path] = content

        # 生成逐帧检测文件（如果源文件是det.txt）
        if filename.lower() == "det.txt":
            for frame_id, line in enumerate(content, 1):
                frame_file = os.path.join(
                    output_base,
                    media_type,
                    split_type,
                    seq_name,
                    "img1",
                    f"{frame_id:06d}.txt"
                )
                mot_entries[frame_file] = [line]

# 验证路径转换
print(f"找到 {len(mot_entries)} 个MOT17条目")
print("示例生成路径：")
for p in list(mot_entries.keys())[:3]:
    print(p.replace(output_base, "$MOT17_BASE"))

# 初始化进度条和锁
pbar = tqdm(total=len(mot_entries))
mutex = Lock()


def restore_entry(entry):
    try:
        file_path, content = entry
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', newline='') as f:
            f.writelines(content)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
    finally:
        with mutex:
            pbar.update(1)


# 执行生成
with ThreadPoolExecutor(max_workers=48) as executor:
    executor.map(restore_entry, mot_entries.items())

# 验证文件结构
test_path = os.path.join(
    output_base,
    "images/test/MOT17-01-SDP/img1/000085.txt"
)
print(f"\n验证文件存在: {os.path.exists(test_path)}")