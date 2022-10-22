import argparse
import os
import tensorboardX
import torchvision.transforms.functional
import timm
import tqdm
import lib
import torch.utils.data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--train_h5_path', type=str, default='data/train.h5')
    parser.add_argument('--val_h5_path', type=str, default='data/train.h5')

    # training
    parser.add_argument('--f_save', type=int, default=2)
    parser.add_argument('--n_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--log_dir', type=str, default='logs/hrnet_w18')
    args = parser.parse_args()

    model = timm.create_model('hrnet_w18', pretrained=True, features_only=True)
    model = lib.pose_model.PoseModel(model).to('cuda')
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                                                ),])
    train_dataset = lib.dataset.OMCDataset(args.train_h5_path, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size)

    os.makedirs(args.log_dir, exist_ok=True)
    summary_writer = tensorboardX.SummaryWriter(args.log_dir)

    epoch_iter = tqdm.tqdm(range(1, args.n_epochs + 1))
    batch_iter = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    for epoch_i in epoch_iter:
        for batch_i, (image, target) in batch_iter:
            image = image.to('cuda')
            target = target.to('cuda')
            optim.zero_grad()
            prediction = model(image)
            loss = torch.nn.functional.mse_loss(prediction, torchvision.transforms.functional.resize(target, prediction.shape[-2:]))
            loss.backward()
            optim.step()

            batch_iter.set_postfix_str(f'train_batch_loss: {loss.item():.3e}')
            summary_writer.add_scalar('train_batch_loss', loss.item(), batch_i)
            summary_writer.flush()

        if epoch_i % args.f_save == 0:
            torch.save(model.state_dict(), os.path.join(args.log_dir, f'{epoch_i}.pt'))
