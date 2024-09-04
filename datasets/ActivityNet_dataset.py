import os
import random

import numpy as np
import pandas as pd
from collections import defaultdict

from datasets.rawvideo_util import RawVideoExtractor
from modules.basic_utils import load_json, read_lines
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class ActivityNetDataset(Dataset):

    def __init__(self, config: Config, split_type='train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type

        dir = './ActivityNet'
        db_file_train = dir + '/train.json'
        train_ids = dir + '/train_list.txt'

        db_file_test = dir + '/test.json'
        test_ids = dir + '/test_list.txt'

        self.db_train = load_json(db_file_train)
        self.db_test = load_json(db_file_test)

        self.num_frames = config.num_frames

        if split_type == 'train':
            train_id = read_lines(train_ids)
            self.train_vids = train_id
            self._construct_all_train_pairs()
        else:
            test_id = read_lines(test_ids)
            self.test_vids = test_id
            self._construct_all_test_pairs()

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
        return len(self.all_test_pairs)

    def _get_vidpath_and_caption_by_index(self, index):
        if self.split_type == 'train':
            vid, caption = self.all_train_pairs[index]
            video_path_mp4 = os.path.join(self.videos_dir, vid + '.mp4')
            video_path_mkv = os.path.join(self.videos_dir, vid + '.mkv')
            video_path_webm = os.path.join(self.videos_dir, vid + '.webm')

            if os.path.exists(video_path_mp4):
                video_path = video_path_mp4
                return video_path, caption, vid
            if os.path.exists(video_path_mkv):
                video_path = video_path_mkv
                return video_path, caption, vid
            if os.path.exists(video_path_webm):
                video_path = video_path_webm
                return video_path, caption, vid

        else:
            vid, caption = self.all_test_pairs[index]
            video_path_mp4 = os.path.join(self.videos_dir, vid + '.mp4')
            video_path_mkv = os.path.join(self.videos_dir, vid + '.mkv')
            video_path_webm = os.path.join(self.videos_dir, vid + '.webm')

            if os.path.exists(video_path_mp4):
                video_path = video_path_mp4
                return video_path, caption, vid
            if os.path.exists(video_path_mkv):
                video_path = video_path_mkv
                return video_path, caption, vid
            if os.path.exists(video_path_webm):
                video_path = video_path_webm
                return video_path, caption, vid

    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        for vid in self.train_vids:
            video_info = self.db_train[vid]
            caption_str = ''
            for time, caption in zip(video_info["timestamps"], video_info["sentences"]):
                caption_str = caption_str + caption
            self.all_train_pairs.append([vid, caption_str])
        print("all_train_pairs len is ", len(self.all_train_pairs))

    def _construct_all_test_pairs(self):
        self.all_test_pairs = []
        for vid in self.test_vids:
            video_info = self.db_test[vid]
            caption_str = ''
            for time, caption in zip(video_info["timestamps"], video_info["sentences"]):
                caption_str = caption_str + caption
            self.all_test_pairs.append([vid, caption_str])
        print("all_test_pairs len is ", len(self.all_test_pairs))
