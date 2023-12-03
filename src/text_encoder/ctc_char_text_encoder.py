from typing import List, NamedTuple
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        return "".join(
            [
                self.ind2char[ind]
                for i, ind in enumerate(inds)
                if (i == 0 or ind != inds[i - 1])
                and ind != self.char2ind[self.EMPTY_TOK]
            ]
        )

    def ctc_beam_search(
        self, probs: torch.tensor, probs_length: int, beam_size: int = 100
    ) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        hypos = {("", self.EMPTY_TOK): 1}

        for char_probs in probs[:probs_length]:
            char_probs, indices = char_probs.sort()
            char_probs = char_probs[-beam_size:]
            indices = indices[-beam_size:]

            new_hypos = defaultdict(float)

            for next_char_prob, next_char_index in zip(char_probs, indices):
                next_char = self.ind2char[next_char_index.item()]

                for (prefix, prev_char), prob in hypos.items():
                    new_prefix = prefix
                    if next_char != self.EMPTY_TOK and next_char != prev_char:
                        new_prefix = prefix + next_char
                    new_hypos[(new_prefix, next_char)] += prob * next_char_prob.item()

            hypos = dict(
                list(sorted(new_hypos.items(), key=lambda x: -x[1]))[:beam_size]
            )

        return list(
            map(
                lambda x: Hypothesis(text=x[0][0], prob=x[1]),
                sorted(hypos.items(), key=lambda x: -x[1]),
            )
        )
