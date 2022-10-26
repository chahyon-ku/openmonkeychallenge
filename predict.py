import argparse
import json
import os

import cv2
import numpy as np
import tensorboardX
import torchvision
import timm
import tqdm
import lib.pose_model
import lib.dataset
import torch.utils.data
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data_h5_path', type=str, default='data/val.h5')

    # testing
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--resume_path', type=str, default='logs/agate/hrnet_w32/10.pt')
    parser.add_argument('--output_path', type=str, default='output/val_preds_w32.json')
    args = parser.parse_args()

    model = timm.create_model('hrnet_w32', pretrained=False, features_only=True)
    model = lib.pose_model.PoseModel(model).to('cuda')
    model.load_state_dict(torch.load(args.resume_path))

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                                                ),])
    data = lib.dataset.OMCDataset(args.data_h5_path, transform=transform)
    loader = torch.utils.data.DataLoader(data, args.batch_size)

    batch_iter = tqdm.tqdm(enumerate(loader), total=len(loader))
    result = {'data': []}
    with torch.no_grad():
        for batch_i, (images, _, bbox) in batch_iter:
            images = images.to('cuda')
            heatmap = model(images)

            for prediction_i, prediction_landmarks in enumerate(model.get_landmarks(heatmap, bbox)):
                result['data'].append({'landmarks': np.round(np.array(prediction_landmarks)).astype(int).tolist()})

    with open(args.output_path, 'w') as f:
        json.dump(result, f)
