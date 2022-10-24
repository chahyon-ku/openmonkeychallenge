import argparse
import collections
import os
import time

import tensorboardX
import torchvision.transforms.functional
import timm
import tqdm
import lib
import torch.utils.data
import numpy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--train_h5_path', type=str, default='data/train.h5')
    parser.add_argument('--val_h5_path', type=str, default='data/train.h5')
    parser.add_argument('--n_workers', type=int, default=1)

    # training
    parser.add_argument('--f_save', type=int, default=2)
    parser.add_argument('--f_val', type=int, default=2)
    parser.add_argument('--n_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--log_dir', type=str, default='logs/hrnet_w18')
    parser.add_argument('--pretrained_model', type=str, default='hrnet_w18')
    args = parser.parse_args()

    model = timm.create_model(args.pretrained_model, pretrained=True, features_only=True)
    model = lib.pose_model.PoseModel(model).to('cuda')
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                                                ),])
    train_dataset = lib.dataset.OMCDataset(args.train_h5_path, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.n_workers)
    val_dataset = lib.dataset.OMCDataset(args.val_h5_path, transform=transform)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, args.batch_size)

    os.makedirs(args.log_dir, exist_ok=True)
    summary_writer = tensorboardX.SummaryWriter(args.log_dir)

    global_step = 0
    postfix = collections.OrderedDict()
    epoch_iter = tqdm.tqdm(range(1, args.n_epochs + 1))
    for epoch_i in epoch_iter:
        batch_iter = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)
        start = time.time()
        for batch_i, (image, target) in batch_iter:
            image = image.to('cuda')
            target = target.to('cuda')
            postfix['data_time'] = time.time() - start
            start = time.time()

            optim.zero_grad()
            prediction = model(image)
            loss = torch.nn.functional.mse_loss(prediction, torchvision.transforms.functional.resize(target, prediction.shape[-2:]))
            loss.backward()
            optim.step()
            postfix['model_time'] = time.time() - start
            start = time.time()

            global_step += 1
            postfix['train_batch_loss'] = loss.item()
            batch_iter.set_postfix(postfix)
            summary_writer.add_scalar('train_batch_loss', loss.item(), global_step)
            summary_writer.flush()

        if epoch_i % args.f_save == 0:
            torch.save(model.state_dict(), os.path.join(args.log_dir, f'{epoch_i}.pt'))

        if epoch_i % args.f_val == 0:
            with torch.no_grad():
                val_losses = []
                for batch_i, (image, target) in enumerate(val_dataloader):
                    image = image.to('cuda')
                    target = target.to('cuda')
                    prediction = model(image)
                    loss = torch.nn.functional.mse_loss(prediction, torchvision.transforms.functional.resize(target, prediction.shape[-2:]))

                    postfix['val_batch_loss'] = loss.item()
                    val_losses.append(loss.item())
                    batch_iter.set_postfix(postfix)
                summary_writer.add_scalar('val_loss', numpy.mean(numpy.array(val_losses)), global_step)
                summary_writer.flush()
