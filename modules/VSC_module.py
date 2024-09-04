import numpy as np
import torch
import torch.nn as nn
from config.base_config import Config

class VSC_module(nn.Module):
    def __init__(self, config: Config):
        super(VSC_module, self).__init__()
        self.num_frames = config.num_frames
        self.embed_dim = config.embed_dim

        self.linear_proj = nn.Linear(self.num_frames, self.embed_dim)
        self.learnable_scalar = nn.Parameter(torch.Tensor(1))

        self.video_mlp = nn.Sequential(
            nn.Linear(self.num_frames, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim))

        self._init_parameters()
        self.config = config

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, text_embeds, video_embeds):

        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        video_embeds = video_embeds / video_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds.unsqueeze(1).repeat(1, self.config.num_frames, 1)

        sims = torch.matmul(text_embeds, video_embeds.permute(0, 2, 1))
        sims = torch.mean(sims, dim=1)

        # sims_out = self.linear_proj(sims)
        sims_out = self.video_mlp(sims)

        return sims, sims_out

class CompressionVideo(nn.Module):
    def __init__(self, config: Config):
        super(CompressionVideo, self).__init__()

        self.config = config

        self.VSC_module = VSC_module(config)

    def forward(self, text_features, video_features):

        sims, log_var = self.VSC_module(text_features, video_features)

        video_std = torch.exp(log_var).unsqueeze(1).expand(-1, self.config.num_frames, -1)

        sigma = torch.rand_like(video_features)
        # sigma = torch.normal(mean=0., std=1.0, size=text_features.shape).to(video_std.device)

        video_r_features = torch.einsum('bf,bfd->bfd', [(1 - sims), video_std])
        video_r_features = video_r_features * sigma
        # video_r_features = video_std * sigma

        video_c_features = video_features - video_r_features

        return text_features, video_c_features