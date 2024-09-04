import cv2
import torch
import random
import numpy as np

class VideoCapture:

    @staticmethod
    def load_frames_from_video(video_path, num_frames, sample='rand'):

        cap = cv2.VideoCapture(video_path)
        assert (cap.isOpened()), video_path

        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []

        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))

        if sample == 'rand':
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        else:
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

        frames = []
        for index in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()

            if not ret:
                n_tries = 5
                for _ in range(n_tries):
                    ret, frame = cap.read()
                    if ret:
                        break

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)
            else:
                raise ValueError

        while len(frames) < num_frames:
            frames.append(frames[-1].clone())
            
        frames = torch.stack(frames).float() / 255
        cap.release()
        return frames, frame_idxs
