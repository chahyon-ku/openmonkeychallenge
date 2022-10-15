import timm.models
import torch.nn
import timm


class PoseModel(torch.nn.Module):
    def __init__(self, hrnet: timm.models.hrnet.HighResolutionNetFeatures):
        super(PoseModel, self).__init__()

        self.add_module('hrnet', hrnet)
        self.pool = torch.nn.AdaptiveAvgPool2d((128, 128))
        self.final = torch.nn.Conv2d(1984, 17, 1, 1, 0)

    def forward(self, image):
        out = self.get_submodule('hrnet')(image)
        for i, branch in enumerate(out):
            out[i] = self.pool(out[i])
        out = torch.cat(out, dim=1)
        out = self.final(out)
        return out
