import argparse
import matplotlib.pyplot as plt
import cv2
import numpy
import torch.utils.data
import torchvision.transforms.functional_tensor
import tqdm
import dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str, default='data/train.h5')
    args = parser.parse_args()

    train_dataset = dataset.OMCDataset(args.h5_path, sigma=10)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 32)

    for batch_i, (image, target) in tqdm.tqdm(enumerate(train_dataloader)):
        mask = torch.sum(target, dim=1, keepdim=True)
        mask = torch.clamp(mask, 0, 1)
        masked = image * mask
        print(masked.shape)
        for image_i in range(len(image)):
            cv2.imshow('image', numpy.array(torch.permute(masked[image_i], (1, 2, 0))))
            cv2.waitKey()
