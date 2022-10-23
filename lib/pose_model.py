import torch.nn
import timm


class PoseModel(torch.nn.Module):
    def __init__(self, hrnet: timm.models.hrnet.HighResolutionNetFeatures):
        super(PoseModel, self).__init__()

        self.add_module('hrnet', hrnet)
        self.final = torch.nn.Conv2d(64, 17, 1, 1, 0)

    def forward(self, image):
        out = self.get_submodule('hrnet')(image)
        out = self.final(out[0])
        return out
