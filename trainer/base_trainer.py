import torch
import os
from abc import abstractmethod
from config.base_config import Config

class BaseTrainer:

    def __init__(self, model, loss, metrics, optimizer, config: Config, writer=None):
        self.config = config

        self.device = self._prepare_device()
        self.model = model.to(self.device)

        self.loss = loss.to(self.device)
        self.metrics = metrics
        self.optimizer = optimizer
        self.start_epoch = 1
        self.global_step = 0

        self.num_epochs = config.num_epochs
        self.writer = writer
        self.checkpoint_dir = config.model_path

        self.log_step = config.log_step
        self.evals_per_epoch = config.evals_per_epoch

    @abstractmethod
    def _train_epoch(self, epoch):

        raise NotImplementedError

    @abstractmethod
    def _valid_epoch_step(self, epoch, step, num_steps):

        raise NotImplementedError

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self._train_epoch(epoch)
            if epoch % self.config.save_every == 0:
                    self._save_checkpoint(epoch, save_best=False)

    def validate(self):
        self._valid_epoch_step(0,0,0)

    def _prepare_device(self):

        use_gpu = torch.cuda.is_available()
        device = torch.device('cuda:0' if use_gpu else 'cpu')
        return device

    def _save_checkpoint(self, epoch, save_best=False):

        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")
        else:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            print("Saving checkpoint: {} ...".format(filename))

    def load_checkpoint(self, model_name):

        # checkpoint_path = os.path.join(self.checkpoint_dir, model_name)
        checkpoint_path = os.path.join(model_name)
        print("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.start_epoch = checkpoint['epoch'] + 1 if 'epoch' in checkpoint else 1
        state_dict = checkpoint['state_dict']
        
        missing_key, unexpected_key = self.model.load_state_dict(state_dict, strict=False)
        print(f'missing_key={missing_key}')
        print(f'unexpected key={unexpected_key}')

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded")

