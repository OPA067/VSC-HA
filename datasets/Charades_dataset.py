import csv
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class CharadesDataset(Dataset):

    def __init__(self, config: Config, split_type='train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type

        dir = './Charades'
        train_csv = dir + '/Charades_v1_train.csv'
        test_csv = dir + '/Charades_v1_test.csv'

        if split_type == 'train':
            self._construct_all_train_pairs(train_csv)
        else:
            self._construct_all_test_pairs(test_csv)

    def __getitem__(self, index):

        if self.split_type == 'train':
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)
            imgs, idxs = VideoCapture.load_frames_from_video(video_path, self.config.num_frames, self.config.video_sample_type)

            if self.img_transforms is not None:
                imgs = self.img_transforms(imgs)

            return {
                'video_id': video_id,
                'video': imgs,
                'text': caption,
            }
        else:
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)
            imgs, idxs = VideoCapture.load_frames_from_video(video_path, self.config.num_frames,self.config.video_sample_type)

            if self.img_transforms is not None:
                imgs = self.img_transforms(imgs)

            return {
                'video_id': video_id,
                'video': imgs,
                'text': caption,
            }

    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        else:
            return len(self.all_test_pairs)

    def _get_vidpath_and_caption_by_index(self, index):
        if self.split_type == 'train':
            vid, caption = self.all_train_pairs[index]
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
            return video_path, caption, vid
        else:
            vid, caption = self.all_test_pairs[index]
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
            return video_path, caption, vid

    def _construct_all_train_pairs(self, file):
        self.all_train_pairs = []
        with open(file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                id, descriptions = row["id"], row["descriptions"]
                self.all_train_pairs.append([id, descriptions])
        print("train len is", len(self.all_train_pairs))

    def _construct_all_test_pairs(self, file):
        self.all_test_pairs = []
        with open(file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                id, descriptions = row["id"], row["descriptions"]
                self.all_test_pairs.append([id, descriptions])
        print("test len is", len(self.all_test_pairs))



