import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

from src.utils import make_mel

mel_maker = make_mel.MelSpectrogram()


def collate_fn(batch: list[dict[str, torch.Tensor]]):
    specs = [item["spectrogram"].squeeze() for item in batch]
    max_spec_len = max([spec.size(-1) for spec in specs])
    specs = pad_sequence(
        [pad(spec, (0, max_spec_len - spec.size(-1))) for spec in specs],
        batch_first=True,
    )

    return {
        "audio": pad_sequence(
            [item["audio"].squeeze() for item in batch], batch_first=True
        ),
        "spectrogram": specs,
        "spectrogram_length": torch.tensor(
            [item["spectrogram"].squeeze().size(-1) for item in batch],
        ),
        "duration": [item["duration"] for item in batch],
    }
