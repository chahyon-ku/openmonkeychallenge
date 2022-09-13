import argparse
import json
import os
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jpgs_dir', type=str, default='data/train')
    parser.add_argument('--json_path', type=str, default='data/train_annotation.json')
    args = parser.parse_args()

    with open(args.json_path, 'r') as f:
        annotations = json.load(f)

    for image_file_i, image_file_name in enumerate(os.listdir(args.jpgs_dir)):
        if image_file_i == 100:
            break
        cv2.imread(os.path.join(args.jpgs_dir, image_file_name))
