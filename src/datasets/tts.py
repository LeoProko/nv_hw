import time
import os

import torch
from tqdm import tqdm
import numpy as np

from text import text_to_sequence


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def get_data_to_buffer(
    data_path, mel_ground_truth, alignment_path, text_cleaners, limit=None
):
    buffer = list()
    text = process_text(data_path)

    if limit is None:
        limit = len(text)

    start = time.perf_counter()
    for i in tqdm(range(limit)):
        mel_gt_name = os.path.join(mel_ground_truth, "ljspeech-mel-%05d.npy" % (i + 1))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(alignment_path, str(i) + ".npy"))
        character = text[i][0 : len(text[i]) - 1]
        character = np.array(text_to_sequence(character, text_cleaners))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)

        buffer.append(
            {
                "src_seq": character,
                "length_target": duration,
                "mel_target": mel_gt_target,
            }
        )

    end = time.perf_counter()
    print("cost {:.2f}s to load {} data into buffer.".format(end - start, len(buffer)))

    return buffer


class BufferDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path, mel_ground_truth, alignment_path, text_cleaners, limit=None
    ):
        self.buffer = get_data_to_buffer(
            data_path, mel_ground_truth, alignment_path, text_cleaners, limit
        )
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]
