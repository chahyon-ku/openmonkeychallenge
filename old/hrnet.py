import cv2
import numpy as np
import torch.nn
import timm


class HRNet(torch.nn.Module):
    def __init__(self, hrnet: timm.models.hrnet.HighResolutionNetFeatures):
        super(HRNet, self).__init__()

        self.add_module('hrnet', hrnet)
        self.final = torch.nn.Conv2d(64, 17, 1, 1, 0)

    def forward(self, image):
        out = self.get_submodule('hrnet')(image)
        out = self.final(out[0])
        return out

    @staticmethod
    def get_landmarks(heatmap, bbox):
        landmarks = torch.argmax(torch.flatten(heatmap, -2), -1)
        landmarks = torch.stack([landmarks % heatmap.shape[-1], landmarks // heatmap.shape[-1]], -1)
        landmarks = landmarks.cpu()

        offset = torch.unsqueeze(torch.stack((bbox[:, 0], bbox[:, 1]), -1), 1)
        scale = torch.unsqueeze(torch.unsqueeze(torch.maximum(bbox[:, 2], bbox[:, 3]) / heatmap.shape[-1], 1), 1)
        scale = torch.unsqueeze(bbox[:, 2:] / heatmap.shape[-1], 1)
        offset = torch.unsqueeze(bbox[:, :2], 1)
        landmarks = landmarks * scale + offset
        # landmarks[:, :, 0] -= torch.unsqueeze(torch.maximum(bbox[:, 3] - bbox[:, 2], torch.tensor(0)), 1) / 2
        # landmarks[:, :, 1] -= torch.unsqueeze(torch.maximum(bbox[:, 2] - bbox[:, 3], torch.tensor(0)), 1) / 2
        landmarks = torch.flatten(landmarks, 1)

        return landmarks

