import argparse
import cv2
import numpy
import torch.utils.data
import tqdm
from lib import dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str, default='data/train.h5')
    args = parser.parse_args()

    train_dataset = dataset.OMCDataset(args.h5_path, sigma=10)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 32)

    for batch_i, (images, targets) in tqdm.tqdm(enumerate(train_dataloader)):
        for image_i, image in enumerate(images):
            for target_i, target in enumerate(targets[image_i]):
                masked_image = image * target
                cv2.imshow('image', numpy.array(torch.permute(masked_image, (1, 2, 0))))
                cv2.waitKey(-1)
