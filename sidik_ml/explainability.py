"""Model-level and candidate-level explainability utilities for SIDIK-ML Non-ANN models."""

from typing import Any, Dict, List, Optional, Sequence

from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from sidik_ml.features import FEATURE_NAMES


def _unwrap_estimator(model: Any) -> Any:
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def extract_global_explanations(
    model: Any,
    X_eval: Sequence[Sequence[float]],
    y_eval: Sequence[int],
    feature_names: Optional[List[str]] = None,
    random_state: int = 42,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Extracts native feature weights, permutation importance, and decision rules (if DecisionTree)."""
    names = list(feature_names) if feature_names is not None else list(FEATURE_NAMES)
    est = _unwrap_estimator(model)

    native_weights: Dict[str, float] = {}
    if hasattr(est, "feature_importances_"):
        raw = [float(v) for v in est.feature_importances_]
        paired = sorted(zip(names, raw), key=lambda kv: abs(kv[1]), reverse=True)
        native_weights = {k: round(v, 6) for k, v in paired[:top_k]}
    elif hasattr(est, "coef_"):
        coefs = est.coef_[0] if len(est.coef_.shape) > 1 else est.coef_
        raw = [float(v) for v in coefs]
        paired = sorted(zip(names, raw), key=lambda kv: abs(kv[1]), reverse=True)
        native_weights = {k: round(v, 6) for k, v in paired[:top_k]}

    perm_dict: Dict[str, float] = {}
    if X_eval and y_eval:
        perm = permutation_importance(
            model,
            X_eval,
            y_eval,
            n_repeats=5,
            random_state=random_state,
            n_jobs=1,
        )
        perm_means = [float(v) for v in perm.importances_mean]  # type: ignore[attr-defined]
        paired_perm = sorted(zip(names, perm_means), key=lambda kv: abs(kv[1]), reverse=True)
        perm_dict = {k: round(v, 6) for k, v in paired_perm[:top_k]}

    tree_rules: Optional[str] = None
    if isinstance(est, DecisionTreeClassifier):
        tree_rules = export_text(est, feature_names=names, max_depth=4)

    return {
        "native_importance": native_weights,
        "permutation_importance": perm_dict,
        "decision_tree_rules": tree_rules,
    }


def explain_single_candidate(
    model: Any,
    feature_vector: Sequence[float],
    feature_names: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, float]:
    """Produces top-k feature contribution scores consistent with the trained model."""
    names = list(feature_names) if feature_names is not None else list(FEATURE_NAMES)
    est = _unwrap_estimator(model)

    contribs: List[tuple[str, float]] = []
    if hasattr(est, "coef_"):
        vec = list(feature_vector)
        if isinstance(model, Pipeline) and "scaler" in model.named_steps:
            scaled = model.named_steps["scaler"].transform([vec])[0]
        else:
            scaled = vec
        coefs = est.coef_[0] if len(est.coef_.shape) > 1 else est.coef_
        for name, c_val, x_val in zip(names, coefs, scaled):
            contribs.append((name, float(c_val) * float(x_val)))
    elif hasattr(est, "feature_importances_"):
        for name, imp, x_val in zip(names, est.feature_importances_, feature_vector):
            signal = float(imp) * (1.0 if float(x_val) != 0.0 else 0.0)
            contribs.append((name, signal))
    else:
        for name, x_val in zip(names, feature_vector):
            contribs.append((name, float(x_val)))

    contribs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return {k: round(v, 6) for k, v in contribs[:top_k]}
