"""
Evaluation metrics for Spotify Recommendation System.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, average_precision_score
)
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, 
                           y_pred_proba: np.ndarray = None) -> Dict:
    """
    Evaluate binary classification performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)
    
    # ROC-AUC (requires probabilities)
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    metrics['tn'] = cm[0, 0]
    metrics['fp'] = cm[0, 1]
    metrics['fn'] = cm[1, 0]
    metrics['tp'] = cm[1, 1]
    
    # Accuracy
    metrics['accuracy'] = (metrics['tp'] + metrics['tn']) / (metrics['tp'] + metrics['tn'] + metrics['fp'] + metrics['fn'])
    
    return metrics


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 10) -> float:
    """
    Calculate Precision@K.
    
    Args:
        y_true: True labels
        y_scores: Predicted scores
        k: Number of top recommendations
        
    Returns:
        Precision@K score
    """
    # Get top-k indices
    top_k_idx = np.argsort(y_scores)[-k:]
    
    # Count relevant items in top-k
    relevant_in_top_k = y_true[top_k_idx].sum()
    
    return relevant_in_top_k / k


def recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 10) -> float:
    """
    Calculate Recall@K.
    
    Args:
        y_true: True labels
        y_scores: Predicted scores
        k: Number of top recommendations
        
    Returns:
        Recall@K score
    """
    # Get top-k indices
    top_k_idx = np.argsort(y_scores)[-k:]
    
    # Count relevant items in top-k
    relevant_in_top_k = y_true[top_k_idx].sum()
    
    # Total relevant items
    total_relevant = y_true.sum()
    
    if total_relevant == 0:
        return 0.0
    
    return relevant_in_top_k / total_relevant


def mean_average_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate Mean Average Precision (MAP).
    
    Args:
        y_true: True labels
        y_scores: Predicted scores
        
    Returns:
        MAP score
    """
    # Sort by scores (descending)
    sorted_idx = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_idx]
    
    # Calculate precision at each position where relevant item appears
    precisions = []
    num_relevant = 0
    
    for i, label in enumerate(y_true_sorted):
        if label == 1:
            num_relevant += 1
            precision_at_i = num_relevant / (i + 1)
            precisions.append(precision_at_i)
    
    if len(precisions) == 0:
        return 0.0
    
    return np.mean(precisions)


def evaluate_ranking(y_true: np.ndarray, y_scores: np.ndarray, 
                     k_values: List[int] = [5, 10, 20]) -> Dict:
    """
    Evaluate ranking performance.
    
    Args:
        y_true: True labels
        y_scores: Predicted scores
        k_values: List of K values for Precision@K and Recall@K
        
    Returns:
        Dictionary with ranking metrics
    """
    metrics = {}
    
    # Precision@K and Recall@K for different K values
    for k in k_values:
        metrics[f'precision@{k}'] = precision_at_k(y_true, y_scores, k)
        metrics[f'recall@{k}'] = recall_at_k(y_true, y_scores, k)
    
    # MAP
    metrics['map'] = mean_average_precision(y_true, y_scores)
    
    return metrics


def print_evaluation_report(metrics: Dict, title: str = "Evaluation Results"):
    """
    Print formatted evaluation report.
    
    Args:
        metrics: Dictionary with metrics
        title: Report title
    """
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)
    
    # Classification metrics
    if 'accuracy' in metrics:
        print(f"\nClassification Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Avg Precision: {metrics['avg_precision']:.4f}")
    
    # Confusion matrix
    if 'confusion_matrix' in metrics:
        print(f"\nConfusion Matrix:")
        print(f"  TN: {metrics['tn']:>6}  FP: {metrics['fp']:>6}")
        print(f"  FN: {metrics['fn']:>6}  TP: {metrics['tp']:>6}")
    
    # Ranking metrics
    ranking_keys = [k for k in metrics.keys() if '@' in k or k == 'map']
    if ranking_keys:
        print(f"\nRanking Metrics:")
        for key in sorted(ranking_keys):
            print(f"  {key.upper()}: {metrics[key]:.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test metrics with dummy data
    np.random.seed(42)
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0])
    y_proba = np.random.rand(10)
    
    # Classification metrics
    class_metrics = evaluate_classification(y_true, y_pred, y_proba)
    print_evaluation_report(class_metrics, "Classification Metrics Test")
    
    # Ranking metrics
    rank_metrics = evaluate_ranking(y_true, y_proba, k_values=[3, 5])
    print_evaluation_report(rank_metrics, "Ranking Metrics Test")
