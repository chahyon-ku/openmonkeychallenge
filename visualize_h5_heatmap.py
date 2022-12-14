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
    parser.add_argument('--h5_path', type=str, default='data/v2/val.h5')
    parser.add_argument('--output_dir', type=str, default='output/heatmap_s8_m4/')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--target_size', type=int, default=56)
    parser.add_argument('--n_images', type=int, default=100)
    args = parser.parse_args()

    unnormalize = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0., 0., 0.],
                                                                                   std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                                  torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                                                   std=[1., 1., 1.]),
                                                  ])

    dataset = lib.dataset_context.OMCDataset(args.h5_path, args.image_size, args.target_size, sigma=8, magnitude=4)
    dataloader = torch.utils.data.DataLoader(dataset, 32)

    os.makedirs(args.output_dir, exist_ok=True)
    n_images = 0
    for i_batch, (images, targets, bboxes, cls) in tqdm.tqdm(enumerate(dataloader)):
        images = unnormalize(images)
        for i_image, image in enumerate(images):
            if n_images == args.n_images:
                return
                
            heatmap = torch.sum(targets[i_image], 0, keepdim=True)
            heatmap = heatmap / torch.max(heatmap)
            heatmap = torchvision.transforms.functional_tensor.resize(heatmap, (args.image_size, args.image_size))
            image = image * heatmap
            image = torch.permute(image, (1, 2, 0))
            plt.imshow(image)

            plt.axis('off')
            plt.savefig(f'{args.output_dir}/{n_images:06d}.png', bbox_inches='tight')
            plt.close()
            n_images += 1


if __name__ == '__main__':
    main()
