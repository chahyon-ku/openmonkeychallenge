import argparse
import io
import json
import cv2
import h5py
import numpy
import dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str, default='data/train_jpg.h5')
    args = parser.parse_args()

    with h5py.File(args.h5_path, 'r') as h5f:
        print(h5f.keys())
        for key in h5f.keys():
            data = json.loads(h5f[key]['data'][()])
            image = cv2.imdecode(numpy.array(h5f[key]['image']), cv2.IMREAD_COLOR)
            bbox_x, bbox_y, bbox_w, bbox_h = data['bbox']
            print(key, bbox_w, bbox_h, image.shape[1], image.shape[0])
            landmarks = numpy.array(data['landmarks'], dtype=int)
            landmarks = numpy.stack((landmarks[0:len(landmarks):2], landmarks[1:len(landmarks):2]), axis=-1)
            for i in range(len(landmarks)):
                landmark_x, landmark_y = dataset.jpg_xy_to_image_xy(landmarks[i][0], landmarks[i][1], bbox_x, bbox_y, bbox_w, bbox_h)
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 0, 0), 5)
            cv2.imshow('image', image)
            cv2.waitKey()
