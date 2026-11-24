"""Classical Non-ANN model factory and neural exclusion guard for SIDIK-ML."""

from typing import Any, Dict, Tuple, Optional

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SUPPORTED_MODELS: Tuple[str, ...] = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
)

FORBIDDEN_NEURAL_KEYWORDS: Tuple[str, ...] = (
    "ann",
    "mlp",
    "neural",
    "cnn",
    "rnn",
    "lstm",
    "gru",
    "transformer",
    "bert",
    "codebert",
    "deep",
)


def validate_non_ann_model_name(model_name: str) -> str:
    """Validates that the requested model is an approved Non-ANN classical algorithm."""
    norm = model_name.strip().lower()
    for bad in FORBIDDEN_NEURAL_KEYWORDS:
        if bad in norm:
            raise ValueError(
                f"Neural / deep-learning architecture '{model_name}' is excluded by SIDIK-ML Non-ANN policy."
            )
    if norm not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported Non-ANN model '{model_name}'. Choose from: {', '.join(SUPPORTED_MODELS)}"
        )
    return norm


def create_non_ann_model(
    model_name: str,
    random_state: int = 42,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Instantiates an interpretable classical Non-ANN scikit-learn classifier."""
    norm = validate_non_ann_model_name(model_name)
    cfg = dict(params or {})

    if norm == "logistic_regression":
        lr_params: Dict[str, Any] = {
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": random_state,
        }
        lr_params.update(cfg)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**lr_params)),
        ])

    if norm == "decision_tree":
        dt_params: Dict[str, Any] = {
            "max_depth": 6,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": random_state,
        }
        dt_params.update(cfg)
        return DecisionTreeClassifier(**dt_params)

    if norm == "random_forest":
        rf_params: Dict[str, Any] = {
            "n_estimators": 100,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "random_state": random_state,
            "n_jobs": 1,
        }
        rf_params.update(cfg)
        return RandomForestClassifier(**rf_params)

    if norm == "extra_trees":
        et_params: Dict[str, Any] = {
            "n_estimators": 100,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "random_state": random_state,
            "n_jobs": 1,
        }
        et_params.update(cfg)
        return ExtraTreesClassifier(**et_params)

    gb_params: Dict[str, Any] = {
        "n_estimators": 100,
        "learning_rate": 0.08,
        "max_depth": 3,
        "random_state": random_state,
    }
    gb_params.update(cfg)
    return GradientBoostingClassifier(**gb_params)
