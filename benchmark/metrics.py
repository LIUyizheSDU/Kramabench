import math

import nltk
from rouge_score import rouge_scorer

def percentage_to_float(percentage_string):
    cleaned_string = percentage_string.strip("%")
    float_value = float(cleaned_string) / 100
    return float_value

class Metric:
    name = "Metric"

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, predicted: str, target: str):
        raise NotImplementedError("Metric must implement __call__ method!")


class Accuracy(Metric):
    name = "accuracy"

    def __call__(self, predicted: str, target: str):
        return 0.0
        # return accuracy_score(predicted["answer"], target["answer"])


class Precision(Metric):
    name = "precision"

    def __call__(self, predicted: str, target: str):
        # Figure out how to compute precision of individual queries
        return 0
        # return precision_score(predicted["answer"], target["answer"])


class Recall(Metric):
    name = "recall"

    def __call__(self, predicted: str, target: str):
        return 0.0
        # return recall_score(predicted["answer"], target["answer"])


class F1(Metric):
    name = "f1"

    def __call__(self, predicted: str, target: str):
        return 0.0
        # return f1_score(predicted["answer"], target["answer"])

class MeanSquaredError(Metric):
    # This method computes the squared error. The evaluation script is responsible for aggregating.
    name = "mean_squared_error"

    def __call__(self, predicted: str | int | float, target: str | int | float):
        # TODO: account for percentage
        return (float(predicted) - float(target)) * (float(predicted) - float(target))
    
class MeanAbsoluteError(Metric):
    # This method computes the squared error. The evaluation script is responsible for aggregating.
    name = "mean_absolute_error"

    def __call__(self, predicted: str | int | float, target: str | int | float):
        # TODO: account for percentage
        return abs(float(predicted) - float(target))

class MeanRelativeAbsoluteError(Metric):
    # This method computes the squared error. The evaluation script is responsible for aggregating.
    name = "mean_relative_absolute_error"

    def __call__(self, predicted: str | int | float, target: str | int | float):
        # TODO: account for percentage
        return abs(float(predicted) - float(target)) / float(target)


class BleuScore(Metric):
    name = "bleu"

    def __call__(self, predicted: str, target: str):
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([target], predicted)
        return BLEUscore


class RougeScore(Metric):
    name = "rouge"

    def __call__(self, predicted: str, target: str):
        # Using Rouge-1, the overlap of words
        rouge = rouge_scorer.RougeScorer(['rouge1'])
        results = rouge.score(target=target, prediction=predicted)
        precision = results['rouge1'].precision
        recall = results['rouge1'].recall
        f1 = results['rouge1'].fmeasure
        return f1


class Success(Metric):
    name = "success"

    def __call__(self, predicted: str, target: str):
        return int(predicted == target)

def metric_factory(metric_name: str):
    metrics = {
        "precision": Precision,
        "recall": Recall,
        "f1": F1,
        "bleu": BleuScore,
        "rouge": RougeScore,
        "success": Success,
        "mean_squared_error": MeanSquaredError,
        "mean_absolute_error": MeanAbsoluteError,
        "mean_relative_absolute_error": MeanRelativeAbsoluteError,
    }
    if metric_name not in metrics:
        raise ValueError(f"Metric '{metric_name}' not found.")
    return metrics[metric_name]()
