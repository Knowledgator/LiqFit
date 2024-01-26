from collections import defaultdict
import abc
from typing import Union


class Collator(abc.ABC):
    def __init__(
        self,
        tokenizer,
        max_length: int,
        padding: Union[bool, str],
        truncation: bool,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    @abc.abstractmethod
    def collate(self, batch):
        raise NotImplementedError("Should be implemented in a subclass.")

    def __call__(self, batch):
        grouped_batch = defaultdict(list)
        for example in batch:
            for k, v in example.items():
                grouped_batch[k].append(v)
        output = self.collate(grouped_batch)
        return output
