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
                                 'vit_base_patch16_224_dino'))
    parser.add_argument('--pretrained', type=bool, default=True)

    # optim
    parser.add_argument('--lr', type=float, default=1e-3)

    # train
    parser.add_argument('--f_save', type=int, default=4)
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
        embed_dim = 768 if 'base' in args.model_name else 384
        patch_size = 8 if 'p8' in args.model_name else 16
        model = lib.vitpose.ViTPose(args.model_name, args.pretrained, args.image_size, patch_size, embed_dim).to('cuda')

    # optim
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # train
    os.makedirs(args.log_dir, exist_ok=True)
    summary_writer = tensorboardX.SummaryWriter(args.log_dir)

    global_step = 0
    postfix = collections.OrderedDict()
    epoch_iter = tqdm.tqdm(range(1, args.n_epochs + 1))
    for epoch_i in epoch_iter:
        start = time.time()
        for batch_i, (image, target, _) in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False):
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

            global_step += 1
            postfix['train_loss'] = loss.item()
            epoch_iter.set_postfix(postfix)
            summary_writer.add_scalar('train_loss', loss.item(), global_step)
            summary_writer.flush()

        with torch.no_grad():
            if epoch_i % args.f_save == 0:
                torch.save(model.state_dict(), os.path.join(args.log_dir, f'{epoch_i}.pt'))

            if epoch_i % args.f_val == 0:
                val_losses = []
                for batch_i, (image, target, _) in enumerate(val_dataloader):
                    image = image.to('cuda')
                    target = target.to('cuda')
                    prediction = model(image)
                    loss = torch.nn.functional.mse_loss(prediction, target)

                    val_losses.append(loss.item())
                val_loss = numpy.mean(numpy.array(val_losses))
                postfix['val_loss'] = val_loss
                epoch_iter.set_postfix(postfix)
                summary_writer.add_scalar('val_loss', val_loss, global_step)
                summary_writer.flush()
