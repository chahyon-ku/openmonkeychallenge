import torch.nn
import timm


class ViTPose(torch.nn.Module):
    def __init__(self, model_name, pretrained, image_size):
        super(ViTPose, self).__init__()

        self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.image_size = image_size
        self.head = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(384, 17, 1, 1, 0)
        )

    def backbone_forward(self, x):
        x = self.backbone.patch_embed(x)
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1,
                                                   -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.backbone.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.backbone.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        return x

    def forward(self, x):
        x = self.backbone_forward(x)[:, :-1]
        x = torch.reshape(x, (x.shape[0], self.image_size // 8, self.image_size // 8, x.shape[-1]))
        x = self.head(x)

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

        timm.models.HighResolutionNetFeatures

        return landmarks
