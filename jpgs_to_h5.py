import argparse
import json
import h5py
import os
import cv2
import numpy
import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jpgs_dir', type=str, default='data/train')
    parser.add_argument('--json_path', type=str, default='data/train_annotation.json')
    parser.add_argument('--h5_path', type=str, default='data/vpng/train.h5')
    args = parser.parse_args()

    with open(args.json_path, 'r') as f:
        annotations = json.load(f)

    os.makedirs(os.path.dirname(args.h5_path), exist_ok=True)
    with h5py.File(args.h5_path, 'w') as h5f:
        jpgs = list(os.listdir(args.jpgs_dir))
        jpgs = sorted(jpgs)
        for jpg_i, jpg_filename in tqdm.tqdm(enumerate(jpgs), total=len(jpgs)):
            x, y, w, h = annotations['data'][jpg_i]['bbox']

            # print(os.path.join(args.jpgs_dir, jpg_filename))
            image = cv2.imread(os.path.join(args.jpgs_dir, jpg_filename))
            image = image[y:y+h, x:x+w]
            image = cv2.imencode('.png', image)[1]

            jpg_key = '%06d' % (jpg_i)
            jpg_group = h5f.create_group(jpg_key)
            jpg_image = jpg_group.create_dataset('image', data=image)
            jpg_data = jpg_group.create_dataset('data', data=json.dumps(annotations['data'][jpg_i]))

            h5f.flush()
