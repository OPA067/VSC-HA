import torch
import torch.nn as nn
from config.base_config import Config

class AbsolutePool(nn.Module):
    def __init__(self, config: Config):
        super(AbsolutePool, self).__init__()

        self.config = config
        self.config.alpha = config.alpha
        self.config.beta = config.beta
        self.config.embed_dim = config.embed_dim
        self.config.center = config.center
        self.config.temp = 5

    def forward(self, text_features, video_features):

        v_weight = torch.einsum('ad,bvd->abv', [text_features, video_features])
        v_weight = torch.softmax(v_weight / 5, dim=-1)
        v_feat = torch.einsum('abv,bvd->abd', [v_weight, video_features])

        return v_feat

