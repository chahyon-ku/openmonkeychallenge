import json
import h5py
import numpy
import cv2
import torch.utils.data
import torchvision.transforms.functional as F


class OMCDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, image_size, target_size, sigma=1):
        super(OMCDataset, self).__init__()
        self.h5f = None
        self.h5_path = h5_path
        self.image_size = image_size
        self.target_size = target_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.sigma = sigma
        with h5py.File(h5_path, 'r') as h5f:
            self.len = len(h5f.keys())

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if self.h5f is None:
            self.h5f = h5py.File(self.h5_path, 'r')
            g_size = 6 * self.sigma + 1
            x = numpy.arange(0, g_size, dtype=float)
            y = x[:, numpy.newaxis]
            x0 = g_size // 2
            y0 = g_size // 2
            self.g = torch.from_numpy(numpy.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2)))

        key = '%06d' % item

        data = json.loads(self.h5f[key]['data'][()])
        box_x, box_y, box_w, box_h = data['bbox']
        box_s = max(box_w, box_h)

        image = cv2.imdecode(self.h5f[key]['image'][()], cv2.IMREAD_COLOR)
        image = F.to_tensor(image)
        image = F.resize(image, [box_h * self.image_size // box_s, box_w * self.image_size // box_s])
        image = F.normalize(image, self.mean, self.std)
        pad_x = (self.image_size - image.shape[2]) // 2
        pad_y = (self.image_size - image.shape[1]) // 2
        image = F.pad(image, [pad_x, pad_y,
                              self.image_size - image.shape[2] - pad_x,
                              self.image_size - image.shape[1] - pad_y])

        metadata = torch.from_numpy(numpy.array([box_x, box_y, box_w, box_h, pad_x, pad_y]))

        if len(data['landmarks']) == 0:
            target = torch.zeros(0)
        else:
            target = torch.zeros(17, self.target_size, self.target_size, dtype=torch.float32)
            landmarks = numpy.array(data['landmarks'], dtype=int)
            landmarks = numpy.stack((landmarks[0:len(landmarks):2], landmarks[1:len(landmarks):2]), axis=-1)

            for i, (x, y) in enumerate(landmarks):
                x = (x - box_x) * self.target_size // box_s + pad_x * self.target_size // self.image_size
                y = (y - box_y) * self.target_size // box_s + pad_y * self.target_size // self.image_size
                l = max(x - self.g.shape[0] // 2, 0)
                t = max(y - self.g.shape[0] // 2, 0)
                r = min(x + self.g.shape[0] // 2 + 1, self.target_size)
                b = min(y + self.g.shape[0] // 2 + 1, self.target_size)
                w = r - l
                h = b - t
                g_l = l - x + self.g.shape[0] // 2
                g_t = t - y + self.g.shape[0] // 2
                g_r = g_l + w
                g_b = g_t + h
                if target[i, t:b, l:r].shape != self.g[g_t:g_b, g_l:g_r].shape:
                    print(box_x, box_y, box_w, box_h, pad_x, pad_y)
                    print(l, t, r, b, g_l, g_t, g_r, g_b)
                    print(target[i, t:b, l:r].shape, self.g[g_t:g_b, g_l:g_r].shape)
                target[i, t:b, l:r] = self.g[g_t:g_b, g_l:g_r]

        return image, target, metadata

