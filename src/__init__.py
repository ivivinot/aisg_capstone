"""Capstone CLV pipeline.

    src.preprocessing  data contract, audit, split, feature pipeline (eda.ipynb 2-8),
                       and the checks that test and validate what it produced
    src.metrics        the scorecard the EDA evaluates every model with (eda.ipynb 8)
    src.models         baseline models for regression, classification and
                       clustering: the bar a real model has to clear (readme.md 7.3)
    src.advanced_models  two further architectures per task, and the paired
                       comparison that decides whether they earn their cost
    src.model_optimization  cross-validation strategy, hyperparameter search and
                       the over/underfitting diagnosis for every model above
    src.evaluation     held-out test scoring, importances, value tiers (11-13)

Nothing is imported eagerly: each module pulls in a different slice of
scikit-learn, and ``predict.py`` only needs the first one.
"""

__all__ = ["preprocessing", "feature_engineering", "metrics", "models",
           "advanced_models", "model_optimization", "evaluation"]
