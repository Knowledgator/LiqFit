from typing import Callable, Dict
from datasets import Dataset
from ..datasets import transform_dataset


def tokenize_and_align_label(
    example: Dict,
    tokenizer: Callable,
    sources_column_name: str = "sources",
    targets_column_name: str = "targets",
):
    """Tokenizes Source and Target sequences and concatenates them for NLI training task.

    Args:
        example (Dict): Dictionary that contains the sources and target sequences.
        tokenizer (Callable): Tokenizer function, if you are using Huggingface
            tokenizer, you can wrap it with your configuration using
            `functools.partial`. Example:
            tokenizer_wrapped_function = \
                functools.partial(tokenizer.batch_encode_plus, padding=True,
                truncation=True, max_length=512) then pass
                `tokenizer_wrapped_function` to this function.
        sources_column_name (str, optional): Sources key name in the
            dictionary. Defaults to "sources".
        targets_column_name (str, optional): Targets key name in the
            dictionary. Defaults to "targets".

    Returns:
        torch.Tensor: A tensor of your tokenized input.
    """
    hypothesis = example[targets_column_name]
    seq = example[sources_column_name]
    tokenized_input = tokenizer([seq, hypothesis])
    return tokenized_input


def transform(
    dataset: Dataset,
    classes: list,
    template: str,
    normalize_negatives: bool,
    positives: int,
    negatives: int,
):
    """Transforms the dataset for NLI training task.

    Args:
        dataset (Dataset): Hugginface Dataset instance
        classes (List[str]): List of possible class labels.
        template (str, optional): Template string for generating examples.
        normalize_negatives (bool, optional): Whether to normalize amount of 
                                negative examples per each positive example of a class.
        positives (int, optional): Number of positive examples to generate per source.
        negatives (int, optional): Number of negative examples to generate per source.

    Raises:
        ValueError: If there is no "{}" in the template. It should exist in
            order to format the template with the labels.

    Returns:
        Dataset: Transformed dataset.
    """
    if "{}" not in template:
        raise ValueError(
            "Cannot apply `.format()` function on the template. "
            'Expected template to have "{}". '
            f"Received: {template}."
        )

    transformed_dataset = transform_dataset(
        dataset, classes, template, normalize_negatives, positives, negatives
    )
    tokenized_dataset = transformed_dataset.map(tokenize_and_align_label)
    return tokenized_dataset
