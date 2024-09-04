import torch
import torch.nn as nn
from config.base_config import Config

class AdaptivePool(nn.Module):
    def __init__(self, config: Config):
        super(AdaptivePool, self).__init__()

        self.config = config
        self.config.alpha = config.alpha
        self.config.beta = config.beta
        self.config.embed_dim = config.embed_dim
        self.config.center = config.center
        self.config.temp = 5

        self.linear_layer_text = nn.Linear(self.config.embed_dim, self.config.center * (self.config.embed_dim // self.config.center))
        self.linear_layer_video = nn.Linear(self.config.embed_dim, self.config.center * (self.config.embed_dim // self.config.center))

        width = int(self.config.embed_dim // self.config.center)
        self.weight_fc = nn.Sequential(
            nn.Linear(2 * width, 4 * width),
            nn.ReLU(inplace=True),
            nn.Linear(4 * width, 1))

    def forward(self, text_features, video_features):

        v_weight = torch.einsum('ad,bvd->abv', [text_features, video_features])
        v_weight = torch.softmax(v_weight / 5, dim=-1)
        v_feat = torch.einsum('abv,bvd->abd', [v_weight, video_features])

        text_mean = text_features
        video_mean = v_feat

        t_feat = text_mean.view(text_mean.shape[0], self.config.center, -1)
        v_feat = video_mean.view(text_mean.shape[0], video_mean.shape[1], self.config.center, -1)

        temp = torch.cat([t_feat.unsqueeze(1).repeat(1, v_feat.shape[1], 1, 1), v_feat], dim=-1)

        weight = self.weight_fc(temp).squeeze(3)

        _t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        _v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)

        retrieve_logits = torch.einsum('acd,abcd->abc', [_t_feat, _v_feat])

        retrieve_logits = torch.einsum('abc,abc->ab', [retrieve_logits, weight])

        return retrieve_logits






