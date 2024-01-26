import evaluate
import numpy as np
from transformers import EvalPrediction


class Accuracy:
    def __init__(self):
        """Simple wrapper class around `evaluate.load("accuracy")`.
        """
        self.accuracy = evaluate.load("accuracy")

    def __call__(self, eval_pred: EvalPrediction):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.accuracy.compute(
            predictions=predictions, references=labels
        )
