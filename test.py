import argparse
import json
import os

import numpy
import tqdm
import lib.dataset
import torch.utils.data
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_h5_path', type=str, default='data/val.h5')
    parser.add_argument('--truth_path', type=str, default='data/val_annotation.json')
    parser.add_argument('--pred_path', type=str, default='output/hrnet_w48_val.json')
    parser.add_argument('--output_dir', type=str, default='output/hrnet_w48_val_test/')
    parser.add_argument('--n_images', type=int, default=100)
    args = parser.parse_args()

    train_dataset = lib.dataset.OMCDataset(args.data_h5_path)

    with open(args.truth_path, 'r') as f:
        truth = json.load(f)
    with open(args.pred_path, 'r') as f:
        pred = json.load(f)

    image_iter = tqdm.tqdm(enumerate(train_dataset), total=args.n_images)

    os.makedirs(args.output_dir, exist_ok=True)
    with torch.no_grad():
        for image_i, (image, target, bbox) in image_iter:
            bbox = numpy.array(bbox).astype('float')
            prediction_landmarks = numpy.reshape(numpy.array(pred['data'][image_i]['landmarks']), (17, 2))
            prediction_landmarks = prediction_landmarks - numpy.reshape(bbox[:2], (1, 2))
            prediction_landmarks = prediction_landmarks / numpy.reshape(bbox[2:], (1, 2)) * image.shape[1]
            target_landmarks = numpy.reshape(numpy.array(truth['data'][image_i]['landmarks']), (17, 2))
            target_landmarks = target_landmarks - numpy.reshape(bbox[:2], (1, 2))
            target_landmarks = target_landmarks / numpy.reshape(bbox[2:], (1, 2)) * image.shape[1]
            # target_landmarks[:, 0] *= image.shape[1] / bbox[2]
            # target_landmarks[:, 1] *= image.shape[1] / bbox[3]
            # target_landmarks = torch.argmax(torch.flatten(torch.from_numpy(target), -2), -1)
            # target_landmarks = numpy.stack([target_landmarks % target.shape[1], target_landmarks // target.shape[1]], -1)

            plt.imshow(image.permute(dims=(1, 2, 0)))
            for landmark_i in range(17):
                plt.gcf().gca().add_patch(plt.Circle((prediction_landmarks[landmark_i][0], prediction_landmarks[landmark_i][1]), color='r', radius=2))
                plt.gcf().gca().add_patch(plt.Circle((target_landmarks[landmark_i][0], target_landmarks[landmark_i][1]), radius=2, color='b'))
                plt.gcf().gca().add_patch(plt.Line2D((target_landmarks[landmark_i][0], prediction_landmarks[landmark_i][0]),
                                                     (target_landmarks[landmark_i][1], prediction_landmarks[landmark_i][1]),))

            plt.savefig(f'{args.output_dir}/{image_i:06d}.png', bbox_inches='tight')
            plt.close()