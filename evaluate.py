import argparse
import json
import numpy as np


def get_mpjpe(truth_np, pred_np, w_np):
    dist = np.sqrt(np.sum((truth_np - pred_np) ** 2, -1))
    mpjpe = np.mean(np.mean(dist / w_np, 0))
    return mpjpe


def get_pck(truth_np, pred_np, w_np, eps):
    dist = np.sqrt(np.sum((truth_np - pred_np) ** 2, -1))
    pck = np.mean((dist / w_np < eps).astype(int))
    return pck


def get_ap(truth_np, pred_np, w_np, k_np, eps):
    dist2 = np.sum((truth_np - pred_np) ** 2, -1)
    oks = np.exp(-dist2 / 2 / (w_np ** 2) / (k_np ** 2))
    ap = np.mean((oks >= eps).astype(int))
    return ap



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--truth_path', type=str, default='data/val_annotation.json')
    parser.add_argument('--pred_path', type=str, default='output/hrnet_w48_5_val.json')
    args = parser.parse_args()

    k_np = np.array([[.025, .025, .026, .035, .035, 0.079, 0.072, 0.062, 0.079, 0.072, 0.062, 0.107,
                      0.087, 0.089, 0.087, 0.089, 0.062]]) * 2

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
    aps = [get_ap(truth_np, pred_np, w_np, k_np, e) for e in np.linspace(0.5, 0.95, 10)]
    print(np.round(mpjpe, 4), np.round(pck, 4), np.round(np.mean(aps), 4), np.round(aps, 4))


if __name__ == '__main__':
    main()
