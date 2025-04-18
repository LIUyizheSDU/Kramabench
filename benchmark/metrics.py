import nltk
from rouge_score import rouge_scorer

class Metric:
    name = "Metric"

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, predicted: str, target: str):
        raise NotImplementedError("Metric must implement __call__ method!")


class Accuracy(Metric):
    name = "Accuracy"

    def __call__(self, predicted: str, target: str):
        return 0.0
        # return accuracy_score(predicted["answer"], target["answer"])


class Precision(Metric):
    name = "Precision"

    def __call__(self, predicted: str, target: str):
        # Figure out how to compute precision of individual queries
        return 0
        # return precision_score(predicted["answer"], target["answer"])


class Recall(Metric):
    name = "Recall"

    def __call__(self, predicted: str, target: str):
        return 0.0
        # return recall_score(predicted["answer"], target["answer"])


class F1(Metric):
    name = "F1"

    def __call__(self, predicted: str, target: str):
        return 0.0
        # return f1_score(predicted["answer"], target["answer"])


class BleuScore(Metric):
    name = "BLEU"

    def __call__(self, predicted: str, target: str):
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([target], predicted)
        return BLEUscore


class RougeScore(Metric):
    name = "ROUGE"

    def __call__(self, predicted: str, target: str):
        # Using Rouge-1, the overlap of words
        rouge = rouge_scorer.RougeScorer(['rouge1'])
        results = rouge.score(target=target, prediction=predicted)
        precision = results['rouge1'].precision
        recall = results['rouge1'].recall
        f1 = results['rouge1'].fmeasure
        return f1


class Success(Metric):
    name = "Success"

    def __call__(self, predicted: str, target: str):
        return int(predicted == target)

def metric_factory(metric_name: str):
    metrics = {
        "Precision": Precision,
        "Recall": Recall,
        "F1": F1,
        "BLEU": BleuScore,
        "ROUGE": RougeScore,
        "Success": Success,
    }
    if metric_name not in metrics:
        raise ValueError(f"Metric '{metric_name}' not found.")
    return metrics[metric_name]()
