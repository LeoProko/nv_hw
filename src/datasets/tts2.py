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
    data_path,
    mel_ground_truth,
    energy_ground_truth,
    pitch_ground_truth,
    alignment_path,
    text_cleaners,
    limit=None,
):
    buffer = list()
    text = process_text(data_path)

    if limit is None:
        limit = len(text)

    start = time.perf_counter()
    names = os.listdir(mel_ground_truth)
    for i in tqdm(range(limit)):
        mel_gt_target = np.load(os.path.join(mel_ground_truth, names[i]))
        energy_gt_target = np.load(os.path.join(energy_ground_truth, names[i]))
        pitch_gt_target = np.load(os.path.join(pitch_ground_truth, names[i]))
        duration = np.load(os.path.join(alignment_path, str(i) + ".npy"))
        character = text[i][0 : len(text[i]) - 1]
        character = np.array(text_to_sequence(character, text_cleaners))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = np.rot90(mel_gt_target, 3)
        mel_gt_target = torch.from_numpy(mel_gt_target.copy())
        energy_gt_target = torch.from_numpy(energy_gt_target)
        pitch_gt_target = torch.from_numpy(pitch_gt_target)

        buffer.append(
            {
                "src_seq": character,
                "length_target": duration,
                "mel_target": mel_gt_target,
                "energy_target": energy_gt_target,
                "pitch_target": pitch_gt_target,
            }
        )

    end = time.perf_counter()
    print("cost {:.2f}s to load {} data into buffer.".format(end - start, len(buffer)))

    return buffer


class BufferDataset2(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        mel_ground_truth,
        energy_ground_truth,
        pitch_ground_truth,
        alignment_path,
        text_cleaners,
        limit=None,
    ):
        self.buffer = get_data_to_buffer(
            data_path,
            mel_ground_truth,
            energy_ground_truth,
            pitch_ground_truth,
            alignment_path,
            text_cleaners,
            limit,
        )
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]
