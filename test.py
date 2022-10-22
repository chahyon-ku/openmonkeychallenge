import argparse
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
    parser.add_argument('--train_h5_path', type=str, default='data/train.h5')
    parser.add_argument('--val_h5_path', type=str, default='data/train.h5')

    # testing
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--resume_path', type=str, default='logs/hrnet_w18/40.pt')
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

    with torch.no_grad():
        for batch_i, (images, targets) in batch_iter:
            images = images.to('cuda')
            targets = targets.to('cuda')
            predictions = model(images)
            loss = torch.nn.functional.mse_loss(predictions, torchvision.transforms.Resize(128)(targets))

            images = images.to('cpu')
            targets = targets.to('cpu')
            predictions = predictions.to('cpu')
            for pred_i, image_predictions in enumerate(predictions):
                image = torch.clip(images[pred_i], 0, 1)
                image_targets = targets[pred_i]
                plt.imshow(image.permute(dims=(1, 2, 0)))

                target_landmarks = torch.argmax(torch.flatten(image_targets, 1), 1)
                target_landmarks = torch.stack([target_landmarks // image_targets.shape[1], target_landmarks % image_targets.shape[1]], -1)
                prediction_landmarks = torch.argmax(torch.flatten(image_predictions, 1), 1)
                prediction_landmarks = torch.stack([prediction_landmarks // image_predictions.shape[1], prediction_landmarks % image_predictions.shape[1]], -1)
                for landmark_i in range(17):
                    plt.gcf().gca().add_patch(plt.Circle((target_landmarks[landmark_i][0], target_landmarks[landmark_i][1]), color='b'))
                    plt.gcf().gca().add_patch(plt.Circle((prediction_landmarks[landmark_i][0] * 2, prediction_landmarks[landmark_i][1] * 2), color='r'))
                    plt.gcf().gca().add_patch(plt.Line2D((target_landmarks[landmark_i][0], prediction_landmarks[landmark_i][0] * 2),
                                                         (target_landmarks[landmark_i][1], prediction_landmarks[landmark_i][1] * 2),))

                plt.show()
