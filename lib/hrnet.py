import torch.nn
import timm


class HRNet(torch.nn.Module):
    def __init__(self, model_name, pretrained, image_size):
        super(HRNet, self).__init__()

        self.hrnet = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.pool = torch.nn.AdaptiveAvgPool2d((image_size // 2, image_size // 2))
        self.final = torch.nn.Conv2d(1984, 17, 1, 1, 0)

    def forward(self, image):
        out = self.get_submodule('hrnet')(image)
        for i, branch in enumerate(out):
            out[i] = self.pool(out[i])
        out = torch.cat(out, dim=1)
        out = self.final(out)
        return out

    @staticmethod
    def get_landmarks(heatmap, bbox):
        landmarks = torch.argmax(torch.flatten(heatmap, -2), -1)
        landmarks = torch.stack([landmarks % heatmap.shape[-1], landmarks // heatmap.shape[-1]], -1)
        landmarks = landmarks.cpu()

        offset = torch.unsqueeze(torch.stack((bbox[:, 0], bbox[:, 1]), -1), 1)
        scale = torch.unsqueeze(torch.unsqueeze(torch.maximum(bbox[:, 2], bbox[:, 3]) / heatmap.shape[-1], 1), 1)
        landmarks = landmarks * scale + offset
        landmarks[:, :, 0] -= torch.unsqueeze(torch.maximum(bbox[:, 3] - bbox[:, 2], torch.tensor(0)), 1) / 2
        landmarks[:, :, 1] -= torch.unsqueeze(torch.maximum(bbox[:, 2] - bbox[:, 3], torch.tensor(0)), 1) / 2
        landmarks = torch.flatten(landmarks, 1)

        return landmarks
