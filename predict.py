import argparse
import json
import os

import cv2
import numpy as np
import tensorboardX
import torchvision
import timm
import tqdm
import lib.hrnet
import lib.dataset
import torch.utils.data
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data_h5_path', type=str, default='data/v2/val.h5')
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--target_size', type=int, default=112)

    # model
    parser.add_argument('--model_name', type=str, default='hrnet_w18',
                        choices=('hrnet_w18', 'hrnet_w32', 'hrnet_w48', 'hrnet_w64',
                                 'vit_small_patch8_224_dino', 'vit_base_patch8_224_dino', 'vit_small_patch16_224_dino',
                                 'vit_base_patch16_224_dino'))
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--resume_path', type=str, default='logs/20/hrnet_w48/8.pt')

    # testing
    parser.add_argument('--output_path', type=str, default='output/hrnet_w48_10_val.json')
    args = parser.parse_args()

    if args.model_name.startswith('hrnet'):
        model = lib.hrnet.HRNet(args.model_name, args.pretrained, args.image_size).to('cuda')
    elif args.model_name.startswith('vit'):
        embed_dim = 768 if 'base' in args.model_name else 384
        patch_size = 8 if 'p8' in args.model_name else 16
        model = lib.vitpose.ViTPose(args.model_name, args.pretrained, args.image_size, patch_size, embed_dim).to('cuda')
    model.load_state_dict(torch.load(args.resume_path))

    data = lib.dataset.OMCDataset(args.data_h5_path, args.image_size, args.target_size)
    loader = torch.utils.data.DataLoader(data, args.batch_size, num_workers=args.n_workers)

    batch_iter = tqdm.tqdm(enumerate(loader), total=len(loader))
    result = {'data': []}
    with torch.no_grad():
        for batch_i, (images, _, bbox) in batch_iter:
            images = images.to('cuda')
            heatmap = model(images)

            for prediction_i, prediction_landmarks in enumerate(model.get_landmarks(heatmap, bbox)):
                result['data'].append({'landmarks': np.round(np.array(prediction_landmarks)).astype(int).tolist()})

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(result, f)
