import json
import h5py
import numpy
import cv2
import torch.utils.data
import torchvision.transforms


class OMCDatasetv2(torch.utils.data.Dataset):
    def __init__(self, h5_path, sigma=1):
        super(OMCDatasetv2, self).__init__()
        self.h5f = None
        self.h5_path = h5_path
        self.transform = transform
        self.sigma = sigma
        with h5py.File(h5_path, 'r') as h5f:
            self.len = len(h5f.keys())

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if self.h5f is None:
            self.h5f = h5py.File(self.h5_path, 'r')
            size = 6 * self.sigma + 1
            x = numpy.arange(0, size, dtype=float)
            y = x[:, numpy.newaxis]
            x0 = size // 2
            y0 = size // 2
            self.g = numpy.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

        key = '%06d' % item
        image = cv2.imdecode(self.h5f[key]['image'][()], cv2.IMREAD_COLOR)
        image = self.transform(image)
        data = json.loads(self.h5f[key]['data'][()])

        if len(data['landmarks']) == 0:
            target = torch.zeros(0)
        else:
            target = numpy.zeros((17, image.shape[1], image.shape[2]), dtype=numpy.float32)
            landmarks = numpy.array(data['landmarks'], dtype=int)
            landmarks = numpy.stack((landmarks[0:len(landmarks):2], landmarks[1:len(landmarks):2]), axis=-1)
            bbox_x, bbox_y, bbox_w, bbox_h = data['bbox']

            pad_x = (bbox_h - bbox_w) // 2 if bbox_h > bbox_w else 0
            pad_y = (bbox_w - bbox_h) // 2 if bbox_w > bbox_h else 0
            bbox_s = max(bbox_w, bbox_h)
            for i in range(len(landmarks)):
                landmark_x = (landmarks[i][0] - bbox_x + pad_x) * image.shape[2] // bbox_s
                landmark_y = (landmarks[i][1] - bbox_y + pad_y) * image.shape[1] // bbox_s
                landmark_l = max(landmark_x - self.g.shape[0] // 2, 0)
                landmark_t = max(landmark_y - self.g.shape[0] // 2, 0)
                landmark_r = min(landmark_x + self.g.shape[0] // 2, image.shape[2])
                landmark_b = min(landmark_y + self.g.shape[0] // 2, image.shape[1])
                g_l = landmark_l - (landmark_x - self.g.shape[0] // 2)
                g_t = landmark_t - (landmark_y - self.g.shape[0] // 2)
                g_r = landmark_r - (landmark_x + self.g.shape[0] // 2) + self.g.shape[0] - 1
                g_b = landmark_b - (landmark_y + self.g.shape[0] // 2) + self.g.shape[0] - 1
                target[i, landmark_t:landmark_b, landmark_l:landmark_r] = self.g[g_t:g_b, g_l:g_r]

        return image, target, torch.from_numpy(numpy.array(data['bbox']))

