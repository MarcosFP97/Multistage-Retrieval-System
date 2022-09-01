from typing import Dict, Any

import datasets
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
from torch import Tensor


def acc_and_f1(preds: Tensor, labels):
    acc = float((preds == labels).mean())
    f1_macro = float(f1_score(y_true=labels, y_pred=preds, average='macro'))
    return {
        "accuracy_macro": acc,
        "f1_macro": f1_macro,
        "f1_correct": float(f1_score(y_true=labels, y_pred=preds)),
        "f1_incorrect": float(f1_score(y_true=labels, y_pred=preds, pos_label=0)),
        "precision_correct": float(precision_score(y_true=labels, y_pred=preds)),
        "precision_incorrect": float(precision_score(y_true=labels, y_pred=preds, pos_label=0)),
        "recall_correct": float(recall_score(y_true=labels, y_pred=preds)),
        "recall_incorrect": float(recall_score(y_true=labels, y_pred=preds, pos_label=0)),
    }


class TrecMetric(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="""TREC metrics for the Health Misinformation task.""",
            citation="stub",
            inputs_description="""
                    Args:
                        predictions: list of predictions to score.
                            Each translation should be tokenized into a list of tokens.
                        references: list of lists of references for each translation.
                            Each reference should be tokenized into a list of tokens.
                    Returns:
                        "accuracy": Accuracy
                        "f1": F1 score
                    """,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int64"),
                    "references": datasets.Value("int64"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
            format="numpy",
        )

    def _compute(self, predictions, references) -> Dict[str, Any]:
        if self.config_name == "classification":
            return acc_and_f1(predictions, references)
        elif self.config_name == "ranking":
            return {
                "weee": 0,
            }
        else:
            raise KeyError(
                "You should supply a configuration name selected in "
                '["classification", "ranking"]'
            )
