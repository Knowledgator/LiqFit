from typing import Callable
import torch

from .base_collator import Collator
from typing import Union


class NLICollator(Collator):
    def __init__(
        self,
        tokenizer: Callable,
        max_length: int,
        padding: Union[bool, str],
        truncation: bool,
    ):
        super().__init__(
            tokenizer,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
        )

    def _tokenize_and_align_labels(self, batch):
        texts = batch.get("texts", None)
        if texts is None:
            raise ValueError(
                "Expected to find a key with name 'texts' that "
                "contains a list of tuples where each tuple "
                "contains the hypothesis and the premise. "
                f"Received: {batch.keys()}"
            )
        tokenized_input = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )
        labels = torch.tensor(batch["labels"])
        tokenized_input.update({"labels": labels})
        return tokenized_input

    def collate(self, batch):
        tokenized_input = self._tokenize_and_align_labels(batch)
        return tokenized_input
