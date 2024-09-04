import gc
import time
import torch
import numpy as np
from tqdm import tqdm
from config.all_config import gen_log
from config.base_config import Config
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, sim_matrix_inference, np_softmax, generate_embeds_per_video_id, \
    sim_matrix_inference_light_allops


class Trainer(BaseTrainer):

    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader,
                 test_data_loader, tokenizer, lr_scheduler=None, writer=None):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best = -1.0

        self.alpha = config.alpha
        self.bata = config.beta

    def _train_epoch(self, epoch):

        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps - 1, self.evals_per_epoch + 1, dtype=int)[1:]

        for batch_idx, data in enumerate(self.train_data_loader):
            if self.tokenizer is not None:
                data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
            if isinstance(data['text'], torch.Tensor):
                data['text'] = data['text'].to(self.device)
            else:
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}

            data['video'] = data['video'].to(self.device)

            text_features, video_cg_features, video_mg_features, fine_logits = self.model(data, is_train=True)

            # Video-Sentence alignment
            output = sim_matrix_training(text_features, video_cg_features, self.pooling_type)
            loss_vsa = self.loss(output, self.model.clip.logit_scale)

            # Frame-Sentence Alignment
            output = sim_matrix_training(text_features, video_mg_features, self.pooling_type)
            loss_fsa = self.loss(output, self.model.clip.logit_scale)

            # Entity-Word Alignment
            loss_ewa = self.loss(fine_logits, self.model.clip.logit_scale)

            loss_vsa = loss_vsa * self.alpha
            loss_fsa = loss_fsa
            loss_ewa = loss_ewa * self.bata

            loss_all = loss_vsa + loss_fsa + loss_ewa

            loss_all.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))

            self.global_step += 1

            total_loss += loss_all.detach().item()

            if batch_idx % self.log_step == 0:
                msg = (
                'Train epoch: {} dl:{}/{} total_loss:{:.10f}, loss_vsa:{:.5f}, loss_fsa:{:.5f}, loss_ewa:{:.5f}'.format(
                    epoch,
                    batch_idx,
                    num_steps - 1,
                    loss_all.detach().item(),
                    loss_vsa.detach().item(),
                    loss_fsa.detach().item(),
                    loss_ewa.detach().item()
                ))
                gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)

            if batch_idx in eval_steps:

                if self.config.skip_eval:
                    msg = '\nSkip eval due to long time usage!\n'
                    gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)

                else:
                    test_res, R1 = self._valid_epoch_step(epoch, batch_idx, num_steps - 1)
                    self.model.train()

                    if R1 > self.best:
                        self.best = R1
                        self._save_checkpoint(epoch, save_best=True)

                    msg = (" Current Best Text-Video R@1 is {}".format(self.best))
                    gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)

        res = {
            'loss_train': total_loss / num_steps
        }

        return res

    def _valid_epoch_step(self, epoch, step, num_steps):

        self.model.eval()
        text_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []

        start_selection_time = time.time()

        with torch.no_grad():
            for idx, data in tqdm(enumerate(self.test_data_loader)):
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.device)
                else:
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}

                data['video'] = data['video'].to(self.device)

                text_features, video_features = self.model(data, is_train=False)

                text_embed_arr.append(text_features.cpu())
                vid_embed_arr.append(video_features.cpu())

                for v_id in data['video_id']:
                    all_vid_ids.append(v_id)

            text_embeds = torch.cat(text_embed_arr)     # [N, D]
            vid_embeds = torch.cat(vid_embed_arr)       # [N, F, D]

            vid_embeds_per_video_id = {}
            for idx, v_id in enumerate(all_vid_ids):
                if v_id not in vid_embeds_per_video_id:
                    vid_embeds_per_video_id[v_id] = vid_embeds[idx]

            vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])   # [N, F, D]

            # self.model.Compression.cpu()
            # text_embeds, vid_embeds = self.model.Compression(text_embeds, vid_embeds)
            # self.model.Compression.cuda()

            # self.model.Absolute.cpu()
            # self.model.Absolute(text_embeds, vid_embeds)
            # self.model.Absolute.cuda()

            self.model.Attention.cpu()
            vid_embeds_pooled = self.model.Attention(text_embeds, vid_embeds)
            self.model.Attention.cuda()

            # self.model.Adaptive.cpu()
            # self.model.Adaptive(text_embeds, vid_embeds)
            # self.model.Adaptive.cuda()

            # Based on T-Mass
            text_embeds_allpairs = torch.zeros(size=(vid_embeds.shape[0], text_embeds.shape[0], text_embeds.shape[1]))
            for idx_vid, single_vid in tqdm(enumerate(vid_embeds)):
                text_embeds_allpairs[idx_vid, :, :] = text_embeds  # [N, N ,D]

            del text_embeds, vid_embeds
            gc.collect()

            # [N, N, 1, D], [N, N, 1, D]
            text_embeds_per_video_id, vid_embeds_pooled_per_video_id = generate_embeds_per_video_id(text_embeds_allpairs, vid_embeds_pooled, all_vid_ids, self.pooling_type)

            # save_memory_mode
            if self.config.save_memory_mode:
                sims = sim_matrix_inference_light_allops(text_embeds_per_video_id,
                                                         vid_embeds_pooled_per_video_id,
                                                         self.pooling_type,
                                                         self.config.batch_size_split,
                                                         self.config)
            else:
                sims = sim_matrix_inference(text_embeds_per_video_id,
                                            vid_embeds_pooled_per_video_id,
                                            self.pooling_type)

            del text_embeds_per_video_id, vid_embeds_pooled_per_video_id
            gc.collect()

            ''' Here we conduct text-video retrieval '''
            if self.config.DSL:
                sims_t2v = sims * np_softmax(sims * 100, axis=0)
            else:
                sims_t2v = sims
            metrics = self.metrics
            res = metrics(sims_t2v)
            R1 = res['R1']
            msg = (f"--text-video--Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                   f"R@1: {res['R1']:.1f}",
                   f"R@5: {res['R5']:.1f}",
                   f"R@10: {res['R10']:.1f} ",
                   f"R@50: {res['R50']:.1f} ",
                   f"MedR: {res['MdR']:.1f}",
                   f"MeanR: {res['MnR']:.1f}",
                   )
            gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)

            ''' Here we conduct video-text retrieval (.T)'''
            sims = sims.permute(2, 1, 0)
            if self.config.DSL:
                sims_v2t = sims * np_softmax(sims * 100, axis=0)
            else:
                sims_v2t = sims
            res = metrics(sims_v2t)
            msg = (f"--video-text--Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                   f"R@1: {res['R1']:.1f}",
                   f"R@5: {res['R5']:.1f}",
                   f"R@10: {res['R10']:.1f} ",
                   f"R@50: {res['R50']:.1f} ",
                   f"MedR: {res['MdR']:.1f}",
                   f"MeanR: {res['MnR']:.1f}",
                   )
            gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)

            end_selection_time = time.time()

            msg = (
                f'To compute all video-text embeddings for the whole dataset, the time usage is {end_selection_time - start_selection_time}\n')
            gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)

            return res, R1
