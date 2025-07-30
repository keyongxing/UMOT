import os
import pandas as pd

def convert_gt_to_labels_with_ids(gt_file, output_dir):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 读取 gt.txt 文件
    df = pd.read_csv(gt_file, header=None, names=[
        'frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'
    ])

    # 过滤掉置信度小于 1 的行（可选）
    df = df[df['conf'] == 1]

    # 遍历每一帧并保存到对应的 .txt 文件
    for frame in df['frame'].unique():
        frame_df = df[df['frame'] == frame]
        output_file = os.path.join(output_dir, f'{int(frame):06d}.txt')

        with open(output_file, 'w') as f:
            for _, row in frame_df.iterrows():
                line = ','.join(map(str, [
                    int(row['frame']), int(row['id']), float(row['bb_left']), float(row['bb_top']),
                    float(row['bb_width']), float(row['bb_height']), float(row['conf'])
                ]))
                f.write(line + '\n')

if __name__ == '__main__':
    # 设置输入和输出路径
    input_dir = '/home/severs-s/kyx_use/pycharm_xinagmu/MOTR/MOTR-main2/MOT17/images/train'  # 修改为你的训练集路径
    output_base_dir = '/home/severs-s/kyx_use/pycharm_xinagmu/MOTR/MOTR-main2/MOT17/labels_with_ids/train'  # 修改为你的输出路径

    # 遍历所有子文件夹
    for seq_dir in os.listdir(input_dir):
        seq_path = os.path.join(input_dir, seq_dir)
        if os.path.isdir(seq_path):
            gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
            if os.path.exists(gt_file):
                output_dir = os.path.join(output_base_dir, seq_dir, 'img1')
                print(f"Processing {seq_dir}...")
                convert_gt_to_labels_with_ids(gt_file, output_dir)
            else:
                print(f"gt.txt not found in {seq_dir}")