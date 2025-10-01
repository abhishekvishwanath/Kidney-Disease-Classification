import sys
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from kidneyclassification.entity.artifacts_entity import ClassificationMetricArtifact
from kidneyclassification.exception.exception import CustomException
from kidneyclassification.logging.logger import logging


def get_classification_metrics(y_true, y_pred, average: str = "macro") -> ClassificationMetricArtifact:
    try:
        logging.info("Calculating multi-class classification metrics")
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        logging.info(
            f"Metrics calculated - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f} (avg={average})"
        )
        return ClassificationMetricArtifact(precision=precision, recall=recall, f1_score=f1)
    except Exception as e:
        raise CustomException(e, sys)
