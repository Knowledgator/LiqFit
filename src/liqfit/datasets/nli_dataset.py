from __future__ import annotations

from typing import Optional, List
from datasets import Dataset, load_dataset

from .transform import transform_dataset


class NLIDataset:
    def __init__(self, hypothesis: List, premises: List, labels: List):
        """LiqFitDataset used for NLI training.

        Args:
            hypothesis (List): List of hypothesis texts.
            premises (List): List of premises texts.
            labels (List): List of labels for each example.
        """
        self.hypothesis = hypothesis
        self.premises = premises
        self.labels = labels

    def __len__(self):
        equal_lengths = (
            len(self.hypothesis) == len(self.premises) == len(self.labels)
        )
        if not equal_lengths:
            raise ValueError(
                "Expected equal lengths between `self.hypothesis`"
                ", `self.premises` and `self.labels`. "
                f"Received: {len(self.hypothesis)} "
                f"- {len(self.premises)} - {len(self.labels)}."
            )
        return len(self.hypothesis)

    def __getitem__(self, idx):
        return {
            "texts": (self.hypothesis[idx], self.premises[idx]),
            "labels": self.labels[idx],
        }

    @classmethod
    def load_dataset(
        cls,
        dataset: Optional[Dataset] = None,
        dataset_name: Optional[str] = None,
        classes: Optional[List[str]] = None,
        text_column: Optional[str] = "text",
        label_column: Optional[str] = "label",
        template: Optional[str] = "This example is {}.",
        normalize_negatives: bool = False,
        positives: int = 1,
        negatives: int = -1,
        multi_label: bool = False,
    ) -> NLIDataset:
        """Returns a `NLIDataset` instance.

        Args:
            dataset (Optional[Dataset], optional): Instance of Huggingface
                Dataset class. Defaults to None.
            dataset_name (Optional[str], optional): Dataset name to load from
                Huggingface datasets. Defaults to None.
            classes (Optional[List[str]], optional): List of classes.
                Defaults to None.
            text_column (Optional[str], optional): Text column name.
                Defaults to 'text'.
            label_column (Optional[str], optional): Label column name.
                Defaults to 'label'.
            template (Optional[str], optional): Template string that will be
                used for Zero-Shot training/prediction. Defaults to
                'This example is {}.'.
            normalize_negatives (bool, optional): Whether to normalize amount
                of negative examples per each positive example of a class.
                Defaults to False.
            positives (int, optional): Number of positive examples to generate
                per source. Defaults to 1.
            negatives (int, optional): Number of negative examples to generate
                per source. Defaults to -1.
            multi_label (bool, optional): Whether each example has multiple
                labels or not. Defaults to False.

        Raises:
            TypeError: if `dataset_name` is `None` while `dataset` instance is
                not passed.
            TypeError: if `label_name` is `None`.
            TypeError: if `text_column` is `None` while `dataset` instance is
                not passed.
            TypeError: if `label_column` is `None` while `classes` is `None`.

        Returns:
            LiqFitDataset: An instance of LiqFitDataset.
        """
        if dataset is None:
            if dataset_name is None:
                raise TypeError(
                    "If dataset object is not provided you need to"
                    " specify dataset_name."
                )
            else:
                dataset = load_dataset(dataset_name)["train"]

        if label_column not in dataset.features:
            raise TypeError(f"Expected to find {label_column} in the dataset.")

        if text_column not in dataset.features:
            raise TypeError(f"Expected to find {text_column} in the dataset.")

        if classes is None:
            raise ValueError(
                f"Expected to have a list classes. Received: {classes}."
            )

        processed_data = transform_dataset(
            dataset,
            classes,
            text_column,
            label_column,
            template,
            normalize_negatives,
            positives,
            negatives,
            multi_label,
        )

        return cls(
            processed_data["sources"],
            processed_data["targets"],
            processed_data["labels"],
        )
