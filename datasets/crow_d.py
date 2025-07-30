import os
import json
from tqdm import tqdm
from PIL import Image


def convert_crowdhuman_labels(anno_path, img_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)

    with open(anno_path, 'r') as f:
        annotations = [json.loads(line) for line in f.readlines()]

    for anno in tqdm(annotations, desc="Generating labels"):
        # ​**直接使用原始ID作为文件名（保留逗号）​**
        raw_img_id = anno["ID"]
        txt_file = f"{raw_img_id}.txt"
        img_file = f"{raw_img_id}.jpg"

        img_path = os.path.join(img_dir, img_file)

        # ​**检查图片是否存在并获取尺寸**
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_file} not found in {img_dir}")
            continue

        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error reading {img_file}: {str(e)}")
            continue

        labels = []
        for track_id, box in enumerate(anno["gtboxes"]):
            if box["tag"] == "person":
                # ​**使用vbox代替fbox获取可见框**
                vbox = box["vbox"]
                x_center = (vbox[0] + vbox[2] / 2) / width
                y_center = (vbox[1] + vbox[3] / 2) / height
                w = (vbox[2] - vbox[0]) / width
                h = (vbox[3] - vbox[1]) / height

                labels.append(f"0 {track_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        with open(os.path.join(label_dir, txt_file), 'w') as f:
            f.writelines(labels)


if __name__ == "__main__":
    raw_anno_path = "/home/severs-s/kyx_use/pycharm_xinagmu/MOTRv2-main/data/Dataset/mot/crowdhuman/annotation_train.odgt"
    existing_img_dir = "/home/severs-s/kyx_use/pycharm_xinagmu/MOTRv2-main/data/Dataset/mot/crowdhuman/images/train"
    output_label_dir = "/home/severs-s/kyx_use/pycharm_xinagmu/MOTRv2-main/data/Dataset/mot/crowdhuman/labels_with_ids/train"

    convert_crowdhuman_labels(raw_anno_path, existing_img_dir, output_label_dir)
    print("Conversion completed!")