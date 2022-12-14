import json
import h5py
import numpy
import cv2
import torch.utils.data
import torchvision
import torchvision.transforms.functional as F
import math

def _apply_op2(img, op_name, magnitude, interpolation, fill=None):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        pass
    elif op_name == "Color":
        pass
    elif op_name == "Contrast":
        pass
    elif op_name == "Sharpness":
        pass
    elif op_name == "Posterize":
        pass
    elif op_name == "Solarize":
        pass
    elif op_name == "AutoContrast":
        pass
    elif op_name == "Equalize":
        pass
    elif op_name == "Invert":
        pass
    elif op_name == "Identity":
        pass
    # removed the error raised here because we may want to remove more code
    return img


class OMCDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, image_size, target_size, sigma=8, magnitude=0):
        super(OMCDataset, self).__init__()
        self.h5f = None
        self.h5_path = h5_path
        self.image_size = image_size
        self.target_size = target_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.sigma = sigma
        self.magnitude = magnitude
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
        image = torch.from_numpy(image)
        image = torch.permute(image, (2, 0, 1))

        image = F.resize(image, [box_h * self.image_size // box_s, box_w * self.image_size // box_s])
        pad_x = (self.image_size - image.shape[2]) // 2
        pad_y = (self.image_size - image.shape[1]) // 2
        image = F.pad(image, [pad_x, pad_y,
                              self.image_size - image.shape[2] - pad_x,
                              self.image_size - image.shape[1] - pad_y])

        # get landmarks before any changes made
        target = torch.zeros(17, self.image_size, self.image_size, dtype=torch.float32)

        if len(data['landmarks']):
            # landmarks already assigned
            landmarks = numpy.array(data['landmarks'], dtype=int)
            landmarks = numpy.stack((landmarks[0:len(landmarks):2], landmarks[1:len(landmarks):2]), axis=-1)
            for i, (x, y) in enumerate(landmarks):
                x = (x - box_x) * self.image_size // box_s + pad_x * self.image_size // self.image_size
                y = (y - box_y) * self.image_size // box_s + pad_y * self.image_size // self.image_size
                l = max(x - self.g.shape[0] // 2, 0)
                t = max(y - self.g.shape[0] // 2, 0)
                r = min(x + self.g.shape[0] // 2 + 1, self.image_size)
                b = min(y + self.g.shape[0] // 2 + 1, self.image_size)
                w = r - l
                h = b - t
                g_l = l - x + self.g.shape[0] // 2
                g_t = t - y + self.g.shape[0] // 2
                g_r = g_l + w
                g_b = g_t + h
                target[i, t:b, l:r] = self.g[g_t:g_b, g_l:g_r]
        else:
            print(data)
        
        # https://pytorch.org/vision/main/_modules/torchvision/transforms/autoaugment.html#RandAugment
        # Number of different magnitudes available to choose from
        num_bins = 10
        op_meta = {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

        num_ops = 3
        for _ in range(num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude_value = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude_value *= -1.0
            image = torchvision.transforms.autoaugment._apply_op(image, op_name, magnitude_value, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, fill=None)
            # apply_op2 across last two dimensions of target (operation for the underlying heat maps)
            # might be able to send all 17 channels though
            target = _apply_op2(target, op_name, magnitude_value, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, fill=None)
            
        image = image.to(torch.get_default_dtype()).div(255)
        image = F.normalize(image, self.mean, self.std)
        target = F.resize(target, [self.target_size, self.target_size])
        target = target / torch.sum(target, dim=(1, 2), keepdim=True)
        
        # Suggestion: check for landmarks earlier (only should be applied to training data)

        metadata = torch.from_numpy(numpy.array([box_x, box_y, box_w, box_h, pad_x, pad_y]))

        return image, target, metadata
