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
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--sigma', type=int, default=16)
    # model
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224',
                        choices=('hrnet_w18', 'hrnet_w32', 'hrnet_w48', 'hrnet_w64',
                                 'vit_small_patch8_224_dino', 'vit_base_patch8_224_dino', 'vit_small_patch16_224_dino',
                                 'vit_base_patch16_224_dino',
                                 'vit_small_patch16_384', 'vit_base_patch16_384', 'vit_large_patch16_384',
                                 'vit_small_patch16_224', 'vit_base_patch16_224', 'vit_large_patch16_224'))
    parser.add_argument('--pretrained', type=bool, default=True)

    # optim
    parser.add_argument('--lr', type=float, default=1e-3)

    # train
    parser.add_argument('--device', type=str, default='cuda:0', choices=('cpu', 'cuda:0', 'cuda:1'))
    parser.add_argument('--f_save', type=int, default=999)
    parser.add_argument('--f_val', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=40)
    parser.add_argument('--log_dir', type=str, default='logs/base/vit_base_patch16_224_16')
    args = parser.parse_args()

    args.target_size = args.image_size // 4 if 'patch16' in args.model_name else args.image_size // 2

    # data
    train_dataset = lib.dataset.OMCDataset(args.train_h5_path, args.image_size, args.target_size, args.sigma)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.n_workers)
    val_dataset = lib.dataset.OMCDataset(args.val_h5_path, args.image_size, args.target_size, args.sigma)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, args.batch_size, num_workers=args.n_workers)

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
        model = lib.vitpose.ViTPose(args.model_name, args.pretrained, args.image_size, patch_size, embed_dim)
    model.to(args.device)

    # optim
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # train
    os.makedirs(args.log_dir, exist_ok=True)
    summary_writer = tensorboardX.SummaryWriter(args.log_dir)
    with open(os.path.join(args.log_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f)

    min_val_loss = 999999
    global_step = 0
    postfix = collections.OrderedDict()
    epoch_iter = tqdm.tqdm(range(1, args.n_epochs + 1))
    for epoch_i in epoch_iter:
        global_step += 1
        start = time.time()
        metrics = collections.defaultdict(list)
        for batch_i, (image, target, bbox) in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False):
            image = image.to(args.device)
            target = target.to(args.device)
            postfix['data_time'] = time.time() - start
            start = time.time()

            optim.zero_grad()
            prediction = model(image)
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            optim.step()
            postfix['model_time'] = time.time() - start
            start = time.time()

            with torch.no_grad():
                pred_lands = np.round(np.array(model.get_landmarks(prediction, bbox)))
                pred_lands = np.reshape(pred_lands, (pred_lands.shape[0], -1, 2))
                targ_lands = np.round(np.array(model.get_landmarks(target, bbox)))
                targ_lands = np.reshape(targ_lands, (targ_lands.shape[0], -1, 2))
                w = np.reshape(np.array(bbox[:, 2]), (-1, 1))

                metrics['pred_lands'].append(pred_lands)
                metrics['targ_lands'].append(targ_lands)
                metrics['w'].append(w)
                metrics['loss'].append(loss.item())

                mpjpe = evaluate.get_mpjpe(pred_lands, targ_lands, w)
                pck = evaluate.get_pck(pred_lands, targ_lands, w)
                aps = [evaluate.get_ap(pred_lands, targ_lands, w, eps=e)
                       for e in np.linspace(0.5, 0.95, 10)]
                ap = sum(aps) / len(aps)

                postfix['train_loss'] = loss.item()
                postfix['train_mpjpe'] = mpjpe
                postfix['train_pck'] = pck
                postfix['train_ap'] = ap
                epoch_iter.set_postfix(postfix)

        for k, v in metrics.items():
            if k != 'loss':
                metrics[k] = np.concatenate(v, axis=0)

        mpjpe = evaluate.get_mpjpe(metrics['pred_lands'], metrics['targ_lands'], metrics['w'])
        pck = evaluate.get_pck(metrics['pred_lands'], metrics['targ_lands'], metrics['w'])
        aps = [evaluate.get_ap(metrics['pred_lands'], metrics['targ_lands'], metrics['w'], eps=e)
               for e in np.linspace(0.5, 0.95, 10)]
        ap = sum(aps) / len(aps)
        loss = np.mean(metrics['loss'])

        postfix['train_mpjpe'] = mpjpe
        postfix['train_pck'] = pck
        postfix['train_ap'] = ap
        postfix['train_loss'] = loss
        epoch_iter.set_postfix(postfix)

        summary_writer.add_scalar('train_mpjpe', mpjpe, global_step)
        summary_writer.add_scalar('train_pck', pck, global_step)
        summary_writer.add_scalar('train_ap', ap, global_step)
        summary_writer.add_scalar('train_loss', loss, global_step)
        summary_writer.flush()

        with torch.no_grad():
            if epoch_i % args.f_val == 0:
                val_pred_lands = []
                val_targ_lands = []
                val_w = []
                val_losses = []
                metrics = collections.defaultdict(list)
                for batch_i, (image, target, bbox) in tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader),
                                                                leave=False):
                    image = image.to(args.device)
                    target = target.to(args.device)
                    prediction = model(image)
                    loss = torch.nn.functional.mse_loss(prediction, target)

                    pred_lands = np.round(np.array(model.get_landmarks(prediction, bbox)))
                    pred_lands = np.reshape(pred_lands, (pred_lands.shape[0], -1, 2))
                    targ_lands = np.round(np.array(model.get_landmarks(target, bbox)))
                    targ_lands = np.reshape(targ_lands, (targ_lands.shape[0], -1, 2))
                    w = np.reshape(np.array(bbox[:, 2]), (-1, 1))

                    metrics['pred_lands'].append(pred_lands)
                    metrics['targ_lands'].append(targ_lands)
                    metrics['w'].append(w)
                    metrics['loss'].append(loss.item())

                    epoch_iter.set_postfix(postfix)

                for k, v in metrics.items():
                    if k != 'loss':
                        metrics[k] = np.concatenate(v, axis=0)

                mpjpe = evaluate.get_mpjpe(metrics['pred_lands'], metrics['targ_lands'], metrics['w'])
                pck = evaluate.get_pck(metrics['pred_lands'], metrics['targ_lands'], metrics['w'])
                aps = [evaluate.get_ap(metrics['pred_lands'], metrics['targ_lands'], metrics['w'], eps=e)
                    for e in np.linspace(0.5, 0.95, 10)]
                ap = sum(aps) / len(aps)
                valid_loss = np.mean(metrics['loss'])

                postfix['valid_mpjpe'] = mpjpe
                postfix['valid_pck'] = pck
                postfix['valid_ap'] = ap
                postfix['valid_loss'] = valid_loss
                epoch_iter.set_postfix(postfix)

                summary_writer.add_scalar('valid_mpjpe', mpjpe, global_step)
                summary_writer.add_scalar('valid_pck', pck, global_step)
                summary_writer.add_scalar('valid_ap', ap, global_step)
                summary_writer.add_scalar('valid_loss', valid_loss, global_step)
                summary_writer.flush()

            torch.save(model.state_dict(), os.path.join(args.log_dir, f'last.pt'))

            if valid_loss < min_val_loss:
                min_val_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(args.log_dir, f'best.pt'))

            if epoch_i % args.f_save == 0:
                torch.save(model.state_dict(), os.path.join(args.log_dir, f'{epoch_i}.pt'))

