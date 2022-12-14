
import argparse
import lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--n_upscales', type=int, default=2)
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
        model = lib.vitpose_context.ViTPose(args.model_name, args.pretrained, args.image_size, patch_size, embed_dim, args.n_upscales, 26)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


if __name__ == '__main__':
    main()