import argparse
import json
import os

import cv2
import numpy
import numpy as np
import tqdm
import lib.dataset
import torch.utils.data
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='data/val')
    parser.add_argument('--truth_path', type=str, default='data/val_annotation.json')
    parser.add_argument('--pred_path', type=str, default='output/val_preds_w18.json')
    parser.add_argument('--output_dir', type=str, default='output/val/')
    parser.add_argument('--n_images', type=int, default=100)
    args = parser.parse_args()

    with open(args.truth_path, 'r') as f:
        truth = json.load(f)
    with open(args.pred_path, 'r') as f:
        pred = json.load(f)

    truth_np = np.array([image['landmarks'] for image in truth['data']])
    truth_np = np.reshape(truth_np, (truth_np.shape[0], -1, 2))
    pred_np = np.array([image['landmarks'] for image in pred['data']])
    pred_np = np.reshape(pred_np, (pred_np.shape[0], -1, 2))

    os.makedirs(args.output_dir, exist_ok=True)
    with torch.no_grad():
        for image_i, dir_entry in tqdm.tqdm(enumerate(os.scandir(args.image_dir))):
            x, y, w, h = truth['data'][image_i]['bbox']
            image = cv2.imread(dir_entry.path)
            truth_landmarks = truth_np[image_i]
            pred_landmarks = pred_np[image_i]

            plt.imshow(image)
            for landmark_i in range(17):
                plt.gcf().gca().add_patch(plt.Circle((pred_landmarks[landmark_i][0], pred_landmarks[landmark_i][1]), color='r', radius=2))
                plt.gcf().gca().add_patch(plt.Circle((truth_landmarks[landmark_i][0], truth_landmarks[landmark_i][1]), radius=2, color='b'))
                plt.gcf().gca().add_patch(plt.Line2D((truth_landmarks[landmark_i][0], pred_landmarks[landmark_i][0]),
                                                     (truth_landmarks[landmark_i][1], pred_landmarks[landmark_i][1]),))

            plt.axis('off')
            plt.savefig(f'{args.output_dir}/{image_i:06d}.png', bbox_inches='tight')
            plt.close()

