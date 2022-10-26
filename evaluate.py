import argparse
import json
import numpy as np


def get_mpjpe(truth_np, pred_np, w_np):
    dist = np.sum((truth_np - pred_np) ** 2, -1)
    mpjpe = np.sum(dist / w_np, 0)
    return mpjpe


def get_pck(truth_np, pred_np, w_np, eps):
    dist = np.sum((truth_np - pred_np) ** 2, -1)
    pck = np.sum((dist / w_np < eps).astype(int)) / len(truth_np) / 17
    return pck


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--truth_path', type=str, default='data/val_annotation.json')
    parser.add_argument('--pred_path', type=str, default='output/val_preds_w32.json')
    args = parser.parse_args()

    with open(args.truth_path, 'r') as f:
        truth = json.load(f)
    with open(args.pred_path, 'r') as f:
        pred = json.load(f)

    truth_np = np.array([image['landmarks'] for image in truth['data']])
    truth_np = np.reshape(truth_np, (truth_np.shape[0], -1, 2))
    pred_np = np.array([image['landmarks'] for image in pred['data']])
    pred_np = np.reshape(pred_np, (pred_np.shape[0], -1, 2))
    w_np = np.reshape(np.array([image['bbox'][2] for image in truth['data']]), (-1, 1))
    eps = 0.2

    mpjpe = get_mpjpe(truth_np, pred_np, w_np)
    pck = get_pck(truth_np, pred_np, w_np, eps)
    print(mpjpe, pck)


if __name__ == '__main__':
    main()
