import argparse
import os

import tensorboardX
import torch
import torchvision
import timm
import tqdm
import dataset
import torch.utils.data
import pose_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--train_h5_path', type=str, default='data/train.h5')
    parser.add_argument('--val_h5_path', type=str, default='data/train.h5')

    # training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_path', type=str, default='models/hrnet_w18.pth')
    parser.add_argument('--log_dir', type=str, default='logs/hrnet_w18')
    args = parser.parse_args()

    model = timm.create_model('hrnet_w18', pretrained=True, features_only=True)
    model = pose_model.PoseModel(model).to('cuda')
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                                                ),])
    train_dataset = dataset.OMCDataset(args.train_h5_path, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size)

    summary_writer = tensorboardX.SummaryWriter(args.log_dir)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    batch_iter = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for batch_i, (image, target) in batch_iter:
        image = image.to('cuda')
        target = target.to('cuda')
        optim.zero_grad()
        prediction = model(image)
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        optim.step()

        batch_iter.set_postfix_str(f'train_batch_loss: {loss.item():.3e}')
        summary_writer.add_scalar('train_batch_loss', loss.item(), batch_i)
        summary_writer.flush()

    torch.save(model.state_dict(), args.model_path)
