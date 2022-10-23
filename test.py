import argparse
import json
import os
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
    parser.add_argument('--train_h5_path', type=str, default='data/val.h5')
    parser.add_argument('--val_h5_path', type=str, default='data/val.h5')

    # testing
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--resume_path', type=str, default='logs/hrnet_w18/40.pt')
    parser.add_argument('--output_path', type=str, default='output/val_preds.json')
    args = parser.parse_args()

    model = timm.create_model('hrnet_w18', pretrained=False, features_only=True)
    model = lib.pose_model.PoseModel(model).to('cuda')
    model.load_state_dict(torch.load(args.resume_path))

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                                                ),])
    train_dataset = lib.dataset.OMCDataset(args.train_h5_path, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size)

    batch_iter = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    result = {'data': []}
    with torch.no_grad():
        for batch_i, (images, targets) in batch_iter:
            images = images.to('cuda')
            predictions = model(images)

            predictions_landmarks = torch.argmax(torch.flatten(predictions, -2), -1)
            predictions_landmarks = torch.stack([predictions_landmarks // predictions.shape[-2],
                                                 predictions_landmarks % predictions.shape[-2]],
                                                -1)
            predictions_landmarks = torch.flatten(predictions_landmarks, 1)
            predictions_landmarks = predictions_landmarks.cpu()

            for prediction_i, prediction_landmarks in enumerate(predictions_landmarks):
                result['data'].append({'landmarks': [landmark.item() * 2 for landmark in prediction_landmarks]})

    with open(args.output_path, 'w') as f:
        json.dump(result, f)