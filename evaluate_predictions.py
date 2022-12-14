import argparse
import json

import numpy
import tqdm
import lib.dataset
import torch.utils.data
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_h5_path', type=str, default='data/val.h5')
    parser.add_argument('--pred_path', type=str, default='output/val_preds.json')

    args = parser.parse_args()

    train_dataset = lib.dataset.OMCDataset(args.data_h5_path)

    with open(args.pred_path, 'r') as f:
        results = json.load(f)

    image_iter = tqdm.tqdm(enumerate(train_dataset), total=len(train_dataset))

    with torch.no_grad():
        for image_i, (image, target) in image_iter:
            prediction_landmarks = numpy.reshape(numpy.array(results['data'][image_i]['landmarks']), (17, 2))
            target_landmarks = numpy.argmax(numpy.reshape(target, (target.shape[0], -1)), 1)
            target_landmarks = numpy.stack([target_landmarks // target.shape[1], target_landmarks % target.shape[1]], -1)

            print(image.shape)
            plt.imshow(image.permute(dims=(1, 2, 0)))
            for landmark_i in range(17):
                plt.gcf().gca().add_patch(plt.Circle((prediction_landmarks[landmark_i][1] * 2, prediction_landmarks[landmark_i][0] * 2), color='r', radius=2))
                plt.gcf().gca().add_patch(plt.Circle((target_landmarks[landmark_i][1], target_landmarks[landmark_i][0]), radius=2, color='b'))
                plt.gcf().gca().add_patch(plt.Line2D((target_landmarks[landmark_i][1], prediction_landmarks[landmark_i][1] * 2),
                                                     (target_landmarks[landmark_i][0], prediction_landmarks[landmark_i][0] * 2),))

            plt.savefig()
            if image_i == 99:
                break
