"""""
这段代码的主要目的是为了准备多目标跟踪任务所需的训练或验证数据。
通过读取原始数据集中的图像和标签信息，它能够生成一个文本文件，其中包含了所有参与训练或验证的图像的路径，
这对于后续使用深度学习模型进行训练是非常有帮助的。
"""


import os
from functools import partial
from typing import List


def solve_MOT_train(root, year):
    assert year in [15, 16, 17]
    dataset_path = 'MOT{}/Images/train'.format(year)
    data_root = os.path.join(root, dataset_path)
    if year == 17:
        video_paths = []
        for video_name in os.listdir(data_root):
            if 'SDP' in video_name:
                video_paths.append(video_name)
    else:
        video_paths = os.listdir(data_root)

    frames = []
    for video_name in video_paths:
        files = os.listdir(os.path.join(data_root, video_name, 'img1'))
        files.sort()
        for i in range(1, len(files) + 1):
            frames.append(os.path.join(dataset_path, video_name, 'img1', '%06d.jpg' % i))
    return frames


def solve_CUHK(root):
    dataset_path = 'ethz/CUHK-SYSU'
    data_root = os.path.join(root, dataset_path)
    file_names = os.listdir(os.path.join(data_root, 'Images'))
    file_names.sort()

    frames = []
    for i in range(len(file_names)):
        if os.path.exists(os.path.join(root, 'ethz/CUHK-SYSU/labels_with_ids', f's{i + 1}.txt')):
            if os.path.exists(os.path.join(root, 'ethz/CUHK-SYSU/Images', f's{i + 1}.jpg')):
                frames.append(os.path.join('ethz/CUHK-SYSU/Images', f's{i + 1}.jpg'))
    return frames

def solve_ETHZ(root):
    dataset_path = 'ethz/ETHZ'
    data_root = os.path.join(root, dataset_path)
    video_paths = []
    for name in os.listdir(data_root):
        if name not in ['eth01', 'eth03']:
            video_paths.append(name)

    frames = []
    for video_path in video_paths:
        files = os.listdir(os.path.join(data_root, video_path, 'Images'))
        files.sort()
        for img_name in files:
            if os.path.exists(os.path.join(data_root, video_path, 'labels_with_ids', img_name.replace('.png', '.txt'))):
                if os.path.exists(os.path.join(data_root, video_path, 'Images', img_name)):
                    frames.append(os.path.join('ethz/ETHZ', video_path, 'Images', img_name))
    return frames


def solve_PRW(root):
    dataset_path = 'ethz/PRW'
    data_root = os.path.join(root, dataset_path)
    frame_paths = os.listdir(os.path.join(data_root, 'Images'))
    frame_paths.sort()
    frames = []
    for i in range(len(frame_paths)):
        if os.path.exists(os.path.join(data_root, 'labels_with_ids', frame_paths[i].split('.')[0] + '.txt')):
            if os.path.exists(os.path.join(data_root, 'Images', frame_paths[i])):
                frames.append(os.path.join(dataset_path, 'Images', frame_paths[i]))
    return frames


dataset_catalog = {
    'MOT15': partial(solve_MOT_train, year=15),
    'MOT16': partial(solve_MOT_train, year=16),
    'MOT17': partial(solve_MOT_train, year=17),
    'CUHK-SYSU': solve_CUHK,
    'ETHZ': solve_ETHZ,
    'PRW': solve_PRW,
}


def solve(dataset_list: List[str], root, save_path):
    all_frames = []
    for dataset_name in dataset_list:
        dataset_frames = dataset_catalog[dataset_name](root)
        print("solve {} frames from dataset:{} ".format(len(dataset_frames), dataset_name))
        all_frames.extend(dataset_frames)
    print("totally {} frames are solved.".format(len(all_frames)))
    with open(save_path, 'w') as f:
        for u in all_frames:
            line = '{}'.format(u) + '\n'
            f.writelines(line)

root = '/data/workspace/datasets/mot'
save_path = '/data/workspace/detr-mot/datasets/data_path/mot17.train' # for fangao
dataset_list = ['MOT17', ]

solve(dataset_list, root, save_path)

#
# import os
# from functools import partial
# from typing import List
#
#
# def solve_MOT_train(root, year):
#     assert year in [15, 16, 17]
#     dataset_path = 'MOT{}/Images/train'.format(year)
#     data_root = os.path.join(root, dataset_path)
#
#     if year == 17:
#         video_paths = []
#         for video_name in os.listdir(data_root):
#             if 'SDP' in video_name:
#                 video_paths.append(video_name)
#     else:
#         video_paths = os.listdir(data_root)
#
#     frames = []
#     for video_name in video_paths:
#         files = os.listdir(os.path.join(data_root, video_name, 'img1'))
#         files.sort()
#         for i in range(1, len(files) + 1):
#             frames.append(os.path.join(dataset_path, video_name, 'img1', '%06d.jpg' % i))
#     return frames
#
#
# def solve_MOT_test(root, year):
#     assert year in [15, 16, 17]
#     dataset_path = 'MOT{}/Images/test'.format(year)
#     data_root = os.path.join(root, dataset_path)
#
#     if year == 17:
#         video_paths = []
#         for video_name in os.listdir(data_root):
#             if 'SDP' in video_name:
#                 video_paths.append(video_name)
#     else:
#         video_paths = os.listdir(data_root)
#
#     frames = []
#     for video_name in video_paths:
#         files = os.listdir(os.path.join(data_root, video_name, 'img1'))
#         files.sort()
#         for i in range(1, len(files) + 1):
#             frames.append(os.path.join(dataset_path, video_name, 'img1', '%06d.jpg' % i))
#     return frames
#
#
# def solve_CUHK(root):
#     dataset_path = 'ethz/CUHK-SYSU'
#     data_root = os.path.join(root, dataset_path)
#     file_names = os.listdir(os.path.join(data_root, 'Images'))
#     file_names.sort()
#
#     frames = []
#     for i in range(len(file_names)):
#         if os.path.exists(os.path.join(root, dataset_path, 'labels_with_ids', f's{i + 1}.txt')):
#             if os.path.exists(os.path.join(root, dataset_path, 'Images', f's{i + 1}.jpg')):
#                 frames.append(os.path.join(dataset_path, 'Images', f's{i + 1}.jpg'))
#     return frames
#
#
# def solve_ETHZ(root):
#     dataset_path = 'ethz/ETHZ'
#     data_root = os.path.join(root, dataset_path)
#     video_paths = []
#     for name in os.listdir(data_root):
#         if name not in ['eth01', 'eth03']:
#             video_paths.append(name)
#
#     frames = []
#     for video_path in video_paths:
#         files = os.listdir(os.path.join(data_root, video_path, 'Images'))
#         files.sort()
#         for img_name in files:
#             label_path = os.path.join(data_root, video_path, 'labels_with_ids', img_name.replace('.png', '.txt'))
#             image_path = os.path.join(data_root, video_path, 'Images', img_name)
#             if os.path.exists(label_path) and os.path.exists(image_path):
#                 frames.append(os.path.join(dataset_path, video_path, 'Images', img_name))
#     return frames
#
#
# def solve_PRW(root):
#     dataset_path = 'ethz/PRW'
#     data_root = os.path.join(root, dataset_path)
#     frame_paths = os.listdir(os.path.join(data_root, 'Images'))
#     frame_paths.sort()
#     frames = []
#     for i in range(len(frame_paths)):
#         label_path = os.path.join(data_root, 'labels_with_ids', frame_paths[i].split('.')[0] + '.txt')
#         image_path = os.path.join(data_root, 'Images', frame_paths[i])
#         if os.path.exists(label_path) and os.path.exists(image_path):
#             frames.append(os.path.join(dataset_path, 'Images', frame_paths[i]))
#     return frames
#
#
# dataset_catalog = {
#     'MOT15-train': partial(solve_MOT_train, year=15),
#     'MOT16-train': partial(solve_MOT_train, year=16),
#     'MOT17-train': partial(solve_MOT_train, year=17),
#     'MOT17-test': partial(solve_MOT_test, year=17),  # 添加测试集解析
#     'CUHK-SYSU': solve_CUHK,
#     'ETHZ': solve_ETHZ,
#     'PRW': solve_PRW,
# }
#
#
# def save_frames_to_file(frames, save_path):
#     with open(save_path, 'w') as f:
#         for frame in frames:
#             f.write(f'{frame}\n')
#
#
# def solve_and_save(dataset_list: List[str], root, save_path_train, save_path_test=None):
#     all_frames_train = []
#     all_frames_test = []
#
#     for dataset_name in dataset_list:
#         if 'train' in dataset_name:
#             print(f"Processing training set {dataset_name}")
#             dataset_frames = dataset_catalog[dataset_name](root)
#             print(f"Solved {len(dataset_frames)} frames from dataset: {dataset_name}")
#             all_frames_train.extend(dataset_frames)
#         elif 'test' in dataset_name:
#             print(f"Processing test set {dataset_name}")
#             dataset_frames = dataset_catalog[dataset_name](root)
#             print(f"Solved {len(dataset_frames)} frames from dataset: {dataset_name}")
#             all_frames_test.extend(dataset_frames)
#
#     if all_frames_train:
#         print(f"Totally {len(all_frames_train)} train frames are solved.")
#         save_frames_to_file(all_frames_train, save_path_train)
#
#     if all_frames_test:
#         print(f"Totally {len(all_frames_test)} test frames are solved.")
#         save_frames_to_file(all_frames_test, save_path_test)
#
#
# if __name__ == "__main__":
#     root = '/home/severs-s/kyx_use/pycharm_xinagmu/MOTR/MOTR-main2'
#     save_path_train = '/home/severs-s/kyx_use/pycharm_xinagmu/MOTR/MOTR-main2/datasets/data_path/mot17.train2'
#     save_path_test = '/home/severs-s/kyx_use/pycharm_xinagmu/MOTR-main2/datasets/data_path/mot17.val2'
#     dataset_list = ['MOT17-train', 'MOT17-test']
#
#     solve_and_save(dataset_list, root, save_path_train, save_path_test)