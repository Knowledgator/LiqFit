from typing import List, Tuple, Optional
from collections import defaultdict
from datasets import Dataset
import numpy as np
import random


def get_labels_stat(labels: List[str]) -> Tuple[List[str], List[float]]:
    """Calculates the number of occurrences and probability of each unique
    label in the provided list of labels.

    Args:
        labels (List[str]): List of label strings

    Returns:
        unique_labels (List[str]): Unique label values
        probs (List[float]): Probability of each label
    """
    # count occurrences of each label
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[label] += 1

    # calculate probabilities
    count = len(labels)
    label_probs = {
        label: label_count / count
        for label, label_count in label_counts.items()
    }

    # extract labels and probabilities
    unique_labels = list(label_probs.keys())
    probs = list(label_probs.values())

    return unique_labels, probs


def transform_dataset(
    dataset: Dataset,
    classes: List[str],
    text_column: Optional[str] = "text",
    label_column: Optional[str] = "label",
    template: Optional[str] = "This example is {}.",
    normalize_negatives: bool = False,
    positives: int = 1,
    negatives: int = -1,
    multi_label: bool = False,
) -> Dataset:
    """Transform a dataset into a format suitable for training.

    Args:
        dataset (Dataset): Input dataset.
        classes (List[str]): List of possible class labels.
        template (str, optional): Template string for generating examples.
        normalize_negatives (bool, optional): Whether to normalize amount of
                                negative examples per each positive example of a class.
        positives (int, optional): Number of positive examples to generate per source.
        negatives (int, optional): Number of negative examples to generate per source.


    Returns:
        Dataset: Transformed dataset.

    This function transforms the input dataset into a format suitable for
    multi-label discriminative training. For each source text, it generates
    positive examples using the provided labels, and negative examples by
    sampling random incorrect labels.
    """
    new_dataset = {"sources": [], "targets": [], "labels": []}

    texts = dataset[text_column]

    if label_column == "all_labels":
        labels = dataset["all_labels"]
        multi_label = True
    elif label_column in dataset.features:
        labels = dataset[label_column]
        if type(labels[0]) == int:
            labels = [classes[idx] for idx in labels]
    else:
        raise NotImplementedError(
            'Dataset should contains "label" or "all_labels" columns'
        )

    if normalize_negatives:
        unique_labels, probs = get_labels_stat(labels)

    if positives == -1:
        positives = len(classes) - 1
    if negatives == -1:
        negatives = len(classes) - 1

    for text, label in zip(texts, labels):
        if multi_label:
            curr_labels = label
        else:
            curr_labels = [label]

        for label in curr_labels:
            for i in range(positives):
                new_dataset["sources"].append(text)
                new_dataset["targets"].append(template.format(label))
                new_dataset["labels"].append(1)

            for _ in range(len(classes) - 1):
                neg_class_ = label

                while neg_class_ in curr_labels:
                    if normalize_negatives:
                        neg_class_ = np.random.choice(unique_labels, p=probs)
                    else:
                        neg_class_ = random.sample(classes, k=1)[0]

                new_dataset["sources"].append(text)
                new_dataset["targets"].append(template.format(neg_class_))
                new_dataset["labels"].append(0)

    return Dataset.from_dict(new_dataset)
