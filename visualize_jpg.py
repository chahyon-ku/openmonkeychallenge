import argparse
import json
import os
import cv2
import numpy

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
        image = cv2.imread(os.path.join(args.jpgs_dir, image_file_name))

        print(annotations['data'][image_file_i])
        left, top, width, height = annotations['data'][image_file_i]['bbox']
        right = left + width
        bottom = top + height
        cv2.line(image, (left, top), (right, top), (0, 0, 0), 5)
        cv2.line(image, (left, bottom), (right, bottom), (0, 0, 0), 5)
        cv2.line(image, (left, top), (left, bottom), (0, 0, 0), 5)
        cv2.line(image, (right, top), (right, bottom), (0, 0, 0), 5)
        landmarks = numpy.array(annotations['data'][image_file_i]['landmarks'], dtype=int)
        landmarks = numpy.stack((landmarks[0:len(landmarks):2], landmarks[1:len(landmarks):2]), axis=-1)
        for i in range(len(landmarks)):
            cv2.circle(image, (landmarks[i][0], landmarks[i][1]), 5, (0, 0, 0), 5)

        cv2.imshow('image', image)
        cv2.waitKey(0)

