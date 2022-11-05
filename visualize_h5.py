import argparse
import os

import cv2
import numpy
import torch.utils.data
import torchvision.transforms.functional_tensor
import tqdm
import matplotlib.pyplot as plt
import lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str, default='data/val.h5')
    parser.add_argument('--output_dir', type=str, default='output/val_h5/')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--n_images', type=int, default=100)
    args = parser.parse_args()

    dataset = lib.dataset.OMCDataset(args.h5_path, args.image_size, sigma=1)
    dataloader = torch.utils.data.DataLoader(dataset, 32)

    os.makedirs(args.output_dir, exist_ok=True)
    n_images = 0
    for i_batch, (images, targets, bboxes) in tqdm.tqdm(enumerate(dataloader)):
        for i_image, image in enumerate(images):
            if n_images == args.n_images:
                return

            landmarks = torch.argmax(torch.flatten(targets[i_image], -2), -1)
            landmarks = torch.stack([landmarks % image.shape[-1], landmarks // image.shape[-1]], -1)

            image = torch.permute(image, (1, 2, 0))
            plt.imshow(image)
            for landmark_i in range(17):
                plt.gcf().gca().add_patch(plt.Circle((landmarks[landmark_i][0], landmarks[landmark_i][1]), radius=2, color='b'))

            plt.axis('off')
            plt.savefig(f'{args.output_dir}/{n_images:06d}.png', bbox_inches='tight')
            plt.close()
            n_images += 1


if __name__ == '__main__':
    main()
