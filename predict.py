import argparse
import json
import os

import cv2
import numpy as np
import tensorboardX
import torchvision
import torchvision.transforms.functional
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
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--sigma', type=int, default=32)

    # model
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224',
                        choices=('hrnet_w18', 'hrnet_w32', 'hrnet_w48', 'hrnet_w64',
                                 'vit_small_patch16_384', 'vit_base_patch16_384', 'vit_large_patch16_384',
                                 'vit_small_patch16_224', 'vit_base_patch16_224', 'vit_large_patch16_224'))
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--resume_path', type=str, default='logs/base/vit_base_patch16_224_16/best.pt')
    parser.add_argument('--n_upscales', type=int, default=3)
    parser.add_argument('--context', type=bool, default=False)

    # testing
    parser.add_argument('--device', type=str, default='cuda:0', choices=('cpu', 'cuda:0', 'cuda:1'))
    parser.add_argument('--output_path', type=str, default='output/base/vit_base_patch16_224_16.json')
    args = parser.parse_args()

    # model
    if args.model_name.startswith('hrnet'):
        model = lib.hrnet.HRNet(args.model_name, args.pretrained, args.image_size)
    elif args.model_name.startswith('vit'):
        if 'large' in args.model_name:
            embed_dim = 1024
        elif 'base' in args.model_name:
            embed_dim = 768
        elif 'small' in args.model_name:
            embed_dim = 384
        if 'patch16' in args.model_name:
            patch_size = 16
        elif 'patch8' in args.model_name:
            patch_size = 8
        model = lib.vitpose_context.ViTPose(args.model_name, args.pretrained, args.image_size, patch_size, embed_dim, args.n_upscales, 26 if args.context else 1)
    model.load_state_dict(torch.load(args.resume_path))
    model.to(args.device)

    args.target_size = args.image_size // 16 * (2 ** args.n_upscales) if 'patch16' in args.model_name else args.image_size // 8 * (2 ** args.n_upscales)
    data = lib.dataset_context.OMCDataset(args.data_h5_path, args.image_size, args.target_size, args.sigma)
    loader = torch.utils.data.DataLoader(data, args.batch_size, num_workers=args.n_workers)

    batch_iter = tqdm.tqdm(enumerate(loader), total=len(loader))
    result = {'data': []}
    with torch.no_grad():
        model.eval()
        for batch_i, (images, _, bbox, cls) in batch_iter:
            images = images.to(args.device)
            cls = cls.to(args.device)
            heatmap = model(images, cls)

            for prediction_i, prediction_landmarks in enumerate(model.get_landmarks(heatmap, bbox)):
                result['data'].append({'landmarks': np.round(np.array(prediction_landmarks)).astype(int).tolist()})

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(result, f, indent=1)
