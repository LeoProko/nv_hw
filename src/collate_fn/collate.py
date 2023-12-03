import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad


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
    }
