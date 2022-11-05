import torch.nn
import timm


class ViTPose(torch.nn.Module):
    def __init__(self, model_name, pretrained, image_size, patch_size, embed_dim):
        super(ViTPose, self).__init__()

        self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.head = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(self.embed_dim, self.embed_dim, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(self.embed_dim, self.embed_dim, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(self.embed_dim, self.embed_dim, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.embed_dim, 17, 1, 1, 0)
        )

    def forward(self, x):
        x = self.vit.forward_features(x)[:, :-1]
        x = torch.reshape(x, (x.shape[0], self.image_size // self.patch_size, self.image_size // self.patch_size,
                              x.shape[-1]))
        x = torch.permute(x, (0, 3, 1, 2))
        x = self.head(x)
        return x

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
