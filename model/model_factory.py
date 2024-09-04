from config.base_config import Config
from model.clip import CLIPStochastic

class ModelFactory:
    @staticmethod
    def get_model(config: Config):
        if config.arch == 'clip_stochastic':
            return CLIPStochastic(config)
        else:
            raise NotImplementedError
