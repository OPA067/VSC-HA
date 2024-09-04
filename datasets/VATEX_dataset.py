import os
import random

from modules.basic_utils import load_json, read_lines
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class VATEXDataset(Dataset):

    def __init__(self, config: Config, split_type='train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        db_train_file = './VATEX/vatex_training_v1.0.json'
        db_test_file = './VATEX/vatex_testing_v1.0.json'
        self.vid2caption_train = load_json(db_train_file)
        self.vid2caption_test = load_json(db_test_file)

        uncan_read_vid = './VATEX/nocan_read_list.txt'
        with open(uncan_read_vid, 'r') as file:
            self.un_read_vid_list = [line.strip() for line in file.readlines()]

        train_vid_map_list_file = './VATEX/train_vid_map_list.txt'
        self.train_vid_map_list = self._construct_vid(train_vid_map_list_file)

        test_vid_map_list_file = './VATEX/test_vid_map_list.txt'
        self.test_vid_map_list = self._construct_vid(test_vid_map_list_file)

        if split_type == 'train':
            self._construct_all_train_pairs()
        else:
            self._construct_all_test_pairs()

    def __getitem__(self, index):
        if self.split_type == 'train':
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index_train(index)
        else:
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index_test(index)

        imgs, idxs = VideoCapture.load_frames_from_video(video_path, self.config.num_frames, self.config.video_sample_type)

        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        ret = {
            'video_id': video_id,
            'video': imgs,
            'text': caption
        }

        return ret

    def _get_vidpath_and_caption_by_index_train(self, index):
        vid, caption = self.all_train_pairs[index]
        video_path = os.path.join(self.videos_dir, vid + '.mp4')
        return video_path, caption, vid

    def _get_vidpath_and_caption_by_index_test(self, index):
        vid, caption = self.all_test_pairs[index]
        video_path = os.path.join(self.videos_dir, vid + '.mp4')
        return video_path, caption, vid

    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.all_test_pairs)

    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        for data in self.vid2caption_train:
            vid = data['videoID']
            if vid in self.train_vid_map_list and self.train_vid_map_list[vid] not in self.un_read_vid_list:
                vid = self.train_vid_map_list[vid]
                caption_list = data['enCap']
                str_caption = ''
                for caption in caption_list:
                    str_caption = str_caption + caption
                self.all_train_pairs.append([vid, str_caption])
        print("len of all_train_pairs: ", len(self.all_train_pairs))

    def _construct_all_test_pairs(self):
        self.all_test_pairs = []
        for data in self.vid2caption_test:
            vid = data['videoID']
            if vid in self.test_vid_map_list and self.test_vid_map_list[vid] not in self.un_read_vid_list:
                vid = self.test_vid_map_list[vid]
                caption_list = data['enCap']
                str_caption = ''
                for caption in caption_list:
                    str_caption = str_caption + caption
                self.all_test_pairs.append([vid, str_caption])
        print("len of all_test_pairs: ", len(self.all_test_pairs))

    def _construct_vid(self, map_file_path):
        video_id_map = {}
        with open(map_file_path, 'r', encoding='utf-8') as map_file:
            for line in map_file:
                original_id, mapped_id = line.strip().split(" ")
                video_id_map[original_id] = mapped_id
        return video_id_map


