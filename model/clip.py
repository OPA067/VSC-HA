import torch
import torch.nn as nn
from config.base_config import Config
from modules.Absolute_module import AbsolutePool
from modules.Adaptive_module import AdaptivePool
from modules.Attention_module import AttentionPool
from modules.VSC_module import CompressionVideo


class CLIPStochastic(nn.Module):
    def __init__(self, config: Config):
        super(CLIPStochastic, self).__init__()
        self.config = config
        
        from transformers import CLIPModel
        if config.clip_arch == 'ViT-B/32':
            self.clip = CLIPModel.from_pretrained("./openai/clip-vit-base-patch32")
        elif config.clip_arch == 'ViT-B/16':
            self.clip = CLIPModel.from_pretrained("./openai/clip-vit-base-patch16")
        else:
            raise ValueError

        config.pooling_type = 'transformer'
        self.Compression = CompressionVideo(config)
        self.Absolute = AbsolutePool(config)
        self.Attention = AttentionPool(config)
        self.Adaptive = AdaptivePool(config)

    def forward(self, data, is_train=True):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)

        text_features = self.clip.get_text_features(**text_data)
        video_features = self.clip.get_image_features(video_data)
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        if is_train:
            text_features, video_c_features = self.Compression(text_features, video_features)

            video_cg_features = self.Absolute(text_features, video_c_features)
            video_mg_features = self.Attention(text_features, video_c_features)
            fine_logits = self.Adaptive(text_features, video_c_features)

            return text_features, video_cg_features, video_mg_features, fine_logits

        else:
            return text_features, video_features