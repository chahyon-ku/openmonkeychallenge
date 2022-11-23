import argparse
import collections
import os
import time

import tensorboardX
import torchvision.transforms.functional
import timm
import tqdm

import evaluate
import lib
import torch.utils.data
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--train_h5_path', type=str, default='data/v2/train.h5')
    parser.add_argument('--val_h5_path', type=str, default='data/v2/val.h5')
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--target_size', type=int, default=112)
    # model
    parser.add_argument('--model_name', type=str, default='hrnet_w18',
                        choices=('hrnet_w18', 'hrnet_w32', 'hrnet_w48', 'hrnet_w64',
                                 'vit_small_patch8_224_dino', 'vit_base_patch8_224_dino', 'vit_small_patch16_224_dino',
                                 'vit_base_patch16_224_dino',
                                 'vit_small_patch16_384', 'vit_base_patch16_384', 'vit_large_patch16_384',
                                 'vit_small_patch16_224', 'vit_base_patch16_224', 'vit_large_patch16_224'))
    parser.add_argument('--pretrained', type=bool, default=True)

    # optim
    parser.add_argument('--lr', type=float, default=1e-3)

    # train
    parser.add_argument('--f_save', type=int, default=999)
    parser.add_argument('--f_val', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=2)
    parser.add_argument('--log_dir', type=str, default='logs/hrnet_w18')
    args = parser.parse_args()

    # data
    train_dataset = lib.dataset.OMCDataset(args.train_h5_path, args.image_size, args.target_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.n_workers)
    val_dataset = lib.dataset.OMCDataset(args.val_h5_path, args.image_size, args.target_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, args.batch_size, num_workers=args.n_workers)

    # model
    if args.model_name.startswith('hrnet'):
        model = lib.hrnet.HRNet(args.model_name, args.pretrained, args.image_size).to('cuda')
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
        model = lib.vitpose.ViTPose(args.model_name, args.pretrained, args.image_size, patch_size, embed_dim).to('cuda')

    # optim
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # train
    os.makedirs(args.log_dir, exist_ok=True)
    summary_writer = tensorboardX.SummaryWriter(args.log_dir)

    min_val_loss = 999999
    global_step = 0
    postfix = collections.OrderedDict()
    epoch_iter = tqdm.tqdm(range(1, args.n_epochs + 1))
    for epoch_i in epoch_iter:
        start = time.time()
        for batch_i, (image, target, bbox) in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False):
            image = image.to('cuda')
            target = target.to('cuda')
            postfix['data_time'] = time.time() - start
            start = time.time()

            optim.zero_grad()
            prediction = model(image)
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            optim.step()
            postfix['model_time'] = time.time() - start
            start = time.time()

            metrics = {}
            with torch.no_grad():
                prediction_landmarks = np.round(np.array(model.get_landmarks(prediction, bbox)))
                prediction_landmarks = np.reshape(prediction_landmarks, (prediction_landmarks.shape[0], -1, 2))
                target_landmarks = np.round(np.array(model.get_landmarks(target, bbox)))
                target_landmarks = np.reshape(target_landmarks, (target_landmarks.shape[0], -1, 2))

                w = np.reshape(np.array(bbox[:, 2]), (-1, 1))
                mpjpe = evaluate.get_mpjpe(prediction_landmarks, target_landmarks, w)
                pck = evaluate.get_pck(prediction_landmarks, target_landmarks, w)
                aps = [evaluate.get_ap(prediction_landmarks, target_landmarks, w, eps=e)
                       for e in np.linspace(0.5, 0.95, 10)]
                ap = sum(aps) / len(aps)

                global_step += 1
                postfix['train_loss'] = loss.item()
                postfix['train_mpjpe'] = mpjpe
                postfix['train_pck'] = pck
                postfix['train_ap'] = ap
                epoch_iter.set_postfix(postfix)

                for metric_name, metric_value in postfix.items():
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append(metric_value)

        metrics = {metric_name: sum(metric_values) / len(metric_values) for metric_name, metric_values in metrics}
        for metric_name, metric_value in postfix.items():
            postfix[metric_name] = metric_value
            summary_writer.add_scalar(metric_name, metric_value)
        epoch_iter.set_postfix(postfix)
        summary_writer.flush()

        with torch.no_grad():
            if epoch_i % args.f_save == 0:
                torch.save(model.state_dict(), os.path.join(args.log_dir, f'{epoch_i}.pt'))

            if epoch_i % args.f_val == 0:
                val_pred_lands = []
                val_targ_lands = []
                val_w = []
                val_losses = []
                for batch_i, (image, target, bbox) in tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader),
                                                                leave=False):
                    image = image.to('cuda')
                    target = target.to('cuda')
                    prediction = model(image)
                    loss = torch.nn.functional.mse_loss(prediction, target)

                    pred_lands = np.round(np.array(model.get_landmarks(prediction, bbox)))
                    pred_lands = np.reshape(pred_lands, (pred_lands.shape[0], -1, 2))
                    targ_lands = np.round(np.array(model.get_landmarks(target, bbox)))
                    targ_lands = np.reshape(targ_lands, (targ_lands.shape[0], -1, 2))

                    val_pred_lands.append(pred_lands)
                    val_targ_lands.append(targ_lands)
                    val_w.append(np.reshape(np.array(bbox[:, 2]), (-1, 1)))
                    val_losses.append(loss.item())

                val_pred_lands = np.concatenate(val_pred_lands)
                val_targ_lands = np.concatenate(val_targ_lands)
                val_w = np.concatenate(val_w)

                mpjpe = evaluate.get_mpjpe(val_pred_lands, val_targ_lands, val_w)
                pck = evaluate.get_pck(val_pred_lands, val_targ_lands, val_w)
                aps = [evaluate.get_ap(val_pred_lands, val_targ_lands, val_w, eps=e)
                       for e in np.linspace(0.5, 0.95, 10)]
                ap = sum(aps) / len(aps)
                val_loss = np.mean(np.array(val_losses))

                postfix['val_mpjpe'] = mpjpe
                postfix['val_pck'] = pck
                postfix['val_ap'] = ap
                postfix['val_loss'] = val_loss
                epoch_iter.set_postfix(postfix)
                summary_writer.add_scalar('val_mpjpe', postfix['val_mpjpe'], global_step)
                summary_writer.add_scalar('val_pck', postfix['val_pck'], global_step)
                summary_writer.add_scalar('val_ap', postfix['val_ap'], global_step)
                summary_writer.add_scalar('val_loss', postfix['val_loss'], global_step)
                summary_writer.flush()

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(args.log_dir, f'best.pt'))

