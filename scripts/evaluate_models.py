"""
Phase 6: Evaluation
churn/scripts/evaluate_models.py

Loads the two models saved in Phase 5 (models/logistic_regression.joblib,
models/random_forest.joblib), regenerates the exact same test split used
to train them, and produces a proper side-by-side comparison:
    - classification reports (precision/recall/F1 per class)
    - confusion matrices (plotted)
    - ROC curves, both models on one chart
    - feature importance / coefficients, top 15 each

Run:
    python scripts/evaluate_models.py

Output (all in outputs/):
    - confusion_matrices.png
    - roc_curves.png
    - feature_importance_logistic_regression.png
    - feature_importance_random_forest.png
    - evaluation_summary.md (plain-text metrics + top features, for the README)
"""

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend -- no display needed to save PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Reuse Phase 5's exact preprocessing + split logic so the test set here
# is identical to the one the models were evaluated on during training.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_model import load_features, preprocess, split_data

MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")

MODEL_FILES = {
    "Logistic Regression": MODELS_DIR / "logistic_regression.joblib",
    "Random Forest": MODELS_DIR / "random_forest.joblib",
}


def load_models() -> dict:
    """Load both saved Phase 5 models. Raises FileNotFoundError with a clear
    message if either is missing (i.e. Phase 5 wasn't run first).
    """
    models = {}
    for name, path in MODEL_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/train_model.py first (Phase 5)."
            )
        models[name] = joblib.load(path)
    return models


def get_test_split():
    """Regenerate the identical train/test split Phase 5 used, so evaluation
    metrics here match what the models were actually validated on.
    """
    df = load_features()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_test, y_test


def get_confusion_matrix(model, X_test, y_test) -> np.ndarray:
    preds = model.predict(X_test)
    return confusion_matrix(y_test, preds)


def get_classification_report(model, X_test, y_test) -> str:
    preds = model.predict(X_test)
    return classification_report(y_test, preds, target_names=["No Churn", "Churn"])


def get_roc_data(model, X_test, y_test) -> dict:
    """Returns fpr, tpr, and AUC for plotting/comparison."""
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    return {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}


def get_top_features(model, feature_names, top_n: int = 15) -> pd.DataFrame:
    """Top N most influential features, works for either model type.

    Logistic Regression: ranked by |coefficient| (sign kept for direction).
    Random Forest: ranked by feature_importances_ (always non-negative).

    If `model` is a sklearn Pipeline (e.g. StandardScaler + LogisticRegression,
    per the Phase 5 fix), this unwraps to the final estimator step first --
    Pipeline objects don't expose coef_/feature_importances_ directly.
    """
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.steps[-1][1]

    if hasattr(estimator, "coef_"):
        values = estimator.coef_[0]
        df = pd.DataFrame({"feature": feature_names, "importance": values})
        df["abs_importance"] = df["importance"].abs()
        df = df.sort_values("abs_importance", ascending=False).head(top_n)
        return df[["feature", "importance"]].reset_index(drop=True)
    elif hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        df = pd.DataFrame({"feature": feature_names, "importance": values})
        df = df.sort_values("importance", ascending=False).head(top_n)
        return df.reset_index(drop=True)
    else:
        raise ValueError("Model has neither coef_ nor feature_importances_")


def plot_confusion_matrices(cms: dict, save_path: Path) -> None:
    fig, axes = plt.subplots(1, len(cms), figsize=(6 * len(cms), 5))
    if len(cms) == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cms.items()):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Churn", "Churn"])
        ax.set_yticklabels(["No Churn", "Churn"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(roc_data: dict, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, data in roc_data.items():
        ax.plot(data["fpr"], data["tpr"], label=f"{name} (AUC = {data['auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(top_features: pd.DataFrame, model_name: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in top_features["importance"]]
    ax.barh(top_features["feature"][::-1], top_features["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Coefficient / Importance")
    ax.set_title(f"Top Features — {model_name}")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    models = load_models()
    X_test, y_test = get_test_split()
    print(f"Evaluating on {len(X_test)} held-out test rows\n")

    cms = {}
    roc_data = {}
    summary_lines = ["# Phase 6: Evaluation Summary\n"]

    for name, model in models.items():
        print(f"=== {name} ===")
        report = get_classification_report(model, X_test, y_test)
        print(report)

        cms[name] = get_confusion_matrix(model, X_test, y_test)
        roc_data[name] = get_roc_data(model, X_test, y_test)
        print(f"AUC: {roc_data[name]['auc']:.4f}\n")

        top_features = get_top_features(model, X_test.columns, top_n=15)
        plot_feature_importance(
            top_features, name, OUTPUTS_DIR / f"feature_importance_{name.lower().replace(' ', '_')}.png"
        )

        summary_lines.append(f"## {name}\n")
        summary_lines.append(f"AUC: {roc_data[name]['auc']:.4f}\n")
        summary_lines.append("```\n" + report + "```\n")
        summary_lines.append("Top features:\n")
        for _, row in top_features.iterrows():
            summary_lines.append(f"- {row['feature']}: {row['importance']:.4f}")
        summary_lines.append("")

    plot_confusion_matrices(cms, OUTPUTS_DIR / "confusion_matrices.png")
    plot_roc_curves(roc_data, OUTPUTS_DIR / "roc_curves.png")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "evaluation_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved plots and evaluation_summary.md to {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()