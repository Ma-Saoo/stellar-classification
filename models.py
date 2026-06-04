"""
models.py
---------
Stellar classification models with configurable training options.

All options are passed through run_all() from Star_Classification.py:

    train_on_subclass    -- True: train on "G2","K1" etc, coarsen post-prediction
                            False: train directly on "G","K" coarse letters
    use_teff             -- True: add elodieTEff as a feature (rows with missing
                            T_eff are dropped)
    use_smote            -- True: oversample minority classes with SMOTE before
                            fitting (requires: pip install imbalanced-learn)
    use_threshold_tuning -- True: after training, tune per-class decision
                            thresholds on a validation split to reduce F/G/K bias
    min_samples          -- minimum examples per class to keep (default 20)

EVALUATION IS ALWAYS COARSE:
    Regardless of train_on_subclass, every ClassificationResult returned here
    has y_true and y_pred as coarse letters (O,B,A,F,G,K,M).
    evaluate.py never sees subclass labels — it only does class-vs-class.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not installed. run_xgboost() will be unavailable.")
    print("         Install with: pip install xgboost\n")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. USE_SMOTE will be ignored.")
    print("         Install with: pip install imbalanced-learn\n")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPECTRAL_CLASSES = ["O", "B", "A", "F", "G", "K", "M"]
COARSE_CLASSES   = ["F", "G", "K", "M"]

# Base feature set — T_eff added conditionally if use_teff=True
ML_FEATURES_BASE = ["u_g", "g_r", "r_i", "i_z", "u_r", "g_i",
                    "u", "g", "r", "i", "z"]

BASELINE_FEATURES = ["g_r", "u_g"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """
    Holds everything evaluate.py needs.
    y_true and y_pred are ALWAYS coarse letters (O,B,A,F,G,K,M).
    """
    model_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    classes: list[str]
    model: object = None
    feature_names: list[str] = field(default_factory=list)
    X_test: pd.DataFrame = None
    cv_scores: np.ndarray = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_features(use_teff: bool) -> list[str]:
    """Return feature list, optionally appending elodieTEff."""
    if use_teff:
        return ML_FEATURES_BASE + ["elodieTEff"]
    return ML_FEATURES_BASE


def _coarsen(labels) -> np.ndarray:
    """Take first character of each label: 'G2'->'G', 'G'->'G'."""
    return np.array([s[0] for s in labels], dtype=object)


def _prepare_data(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    min_samples: int,
    test_size: float = 0.2,
    val_size: float  = 0.1,
    random_state: int = 42,
) -> tuple:
    """
    Clean, filter rare classes, then split into train / val / test.

    val split is carved out of the training portion and used for threshold
    tuning. Returns (X_train, X_val, X_test, y_train, y_val, y_test).

    y_* are always the raw target strings (subclass or coarse depending on
    the caller). Coarsening to letters happens after prediction in run_all().
    """
    df_clean = df.dropna(subset=features + [target]).copy()

    # For T_eff: drop rows where value is 0 (SDSS sentinel for missing)
    if "elodieTEff" in features:
        before = len(df_clean)
        df_clean = df_clean[df_clean["elodieTEff"] > 0]
        print(f"      Dropped {before - len(df_clean):,} rows with missing T_eff")

    # Drop rare classes
    counts = df_clean[target].value_counts()
    valid  = counts[counts >= min_samples].index
    rare   = counts[counts  < min_samples]
    if len(rare):
        n_dropped = (~df_clean[target].isin(valid)).sum()
        print(f"      Dropping {len(rare)} rare class(es) with <{min_samples} samples "
              f"({n_dropped:,} rows): {sorted(rare.index.tolist())}")
    df_clean = df_clean[df_clean[target].isin(valid)]

    X = df_clean[features].values
    y = df_clean[target].values

    # First split: train+val vs test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Second split: train vs val (val_size is fraction of original data)
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, random_state=random_state, stratify=y_tv
    )

    # Wrap back into DataFrames so downstream code has column names
    X_train_df = pd.DataFrame(X_train, columns=features)
    X_val_df   = pd.DataFrame(X_val,   columns=features)
    X_test_df  = pd.DataFrame(X_test,  columns=features)

    return X_train_df, X_val_df, X_test_df, y_train, y_val, y_test


def _apply_smote(X_train: pd.DataFrame, y_train: np.ndarray,
                 random_state: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Oversample minority classes with SMOTE.
    Returns resampled (X_train, y_train) as DataFrame + array.
    SMOTE operates in feature space so it never leaks test data.
    """
    if not SMOTE_AVAILABLE:
        print("      SMOTE not available — skipping oversampling.")
        return X_train, y_train

    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train.values, y_train)
    print(f"      SMOTE: {len(y_train):,} -> {len(y_res):,} training samples")

    # Log new class distribution
    unique, counts = np.unique(y_res, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"        {cls}: {cnt:,}")

    return pd.DataFrame(X_res, columns=X_train.columns), y_res


def _tune_thresholds(
    clf,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    classes: list[str],
    coarse_true: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """
    Find per-class probability multipliers that maximise weighted F1 on the
    validation set at the COARSE level.

    Strategy: grid search a scalar weight for each class independently.
    Weight > 1 makes the model more likely to predict that class.
    Weight < 1 makes it less likely.

    Returns an array of weights, one per class in `classes` order.
    These are applied at test time by multiplying the predicted probabilities
    before taking argmax.

    Note: coarse_true is always coarse letters even when training on subclasses,
    because threshold tuning targets the coarse evaluation metric.
    """
    print("      Tuning per-class decision thresholds on validation set...")

    # Get probabilities on val set
    proba_val = clf.predict_proba(X_val)   # shape (n_val, n_classes)

    # Map from training class index to coarse letter
    coarse_classes_ordered = [c[0] for c in classes]

    # Current coarse predictions (untuned)
    best_weights = np.ones(len(classes))
    best_f1      = _coarse_f1(proba_val, best_weights, classes,
                               coarse_classes_ordered, coarse_true)

    # Grid search weights in [0.5, 2.0] for each class independently
    grid = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8, 2.0]

    for i, cls in enumerate(classes):
        best_w = 1.0
        for w in grid:
            candidate = best_weights.copy()
            candidate[i] = w
            f1 = _coarse_f1(proba_val, candidate, classes,
                            coarse_classes_ordered, coarse_true)
            if f1 > best_f1:
                best_f1 = f1
                best_w  = w
        best_weights[i] = best_w
        if best_w != 1.0:
            print(f"        {cls}: weight={best_w:.1f}")

    print(f"      Val weighted-F1 before tuning: {_coarse_f1(proba_val, np.ones(len(classes)), classes, coarse_classes_ordered, coarse_true):.3f}")
    print(f"      Val weighted-F1 after tuning : {best_f1:.3f}")
    return best_weights


def _coarse_f1(proba, weights, classes, coarse_classes_ordered, y_true_coarse):
    """Helper: apply weights to proba, predict, coarsen, score."""
    weighted_proba = proba * weights
    pred_idx       = weighted_proba.argmax(axis=1)
    pred_coarse    = np.array([coarse_classes_ordered[i] for i in pred_idx])
    return f1_score(y_true_coarse, pred_coarse,
                    labels=SPECTRAL_CLASSES, average="weighted", zero_division=0)


def _apply_weights(proba, weights, classes):
    """Apply threshold weights to probability matrix and return class predictions."""
    weighted = proba * weights
    pred_idx = weighted.argmax(axis=1)
    return np.array([classes[i] for i in pred_idx])


def print_result(result: ClassificationResult) -> None:
    """Print a coarse-level classification report to console."""
    print(f"\n{'='*55}")
    print(f"Model: {result.model_name}  [coarse evaluation]")
    print(f"{'='*55}")
    print(classification_report(
        result.y_true, result.y_pred,
        labels=result.classes, zero_division=0
    ))
    if result.cv_scores is not None:
        print(f"Cross-val weighted-F1: {result.cv_scores.mean():.3f} "
              f"(± {result.cv_scores.std():.3f})")


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def color_cut_classifier(g_r: float, u_g: float) -> str:
    if g_r < 0.0:
        return "O" if u_g < 0.0 else "B"
    elif g_r < 0.2:
        return "A"
    elif g_r < 0.4:
        return "F"
    elif g_r < 0.6:
        return "G"
    elif g_r < 0.9:
        return "K"
    else:
        return "M"


def run_baseline(df: pd.DataFrame) -> ClassificationResult:
    """
    Classical color-cut rules on full dataset.
    Always predicts and evaluates at coarse level.
    """
    print("\n[Baseline] Applying classical color-cut rules...")

    df_clean = df.dropna(subset=BASELINE_FEATURES + ["spectral_class"])
    df_clean = df_clean[df_clean["spectral_class"].isin(SPECTRAL_CLASSES)]

    y_true = df_clean["spectral_class"].values
    y_pred = np.array([
        color_cut_classifier(row["g_r"], row["u_g"])
        for _, row in df_clean[BASELINE_FEATURES].iterrows()
    ])

    result = ClassificationResult(
        model_name    = "Baseline (color cuts)",
        y_true        = y_true,
        y_pred        = y_pred,
        classes       = SPECTRAL_CLASSES,
        feature_names = BASELINE_FEATURES,
    )
    print_result(result)
    return result


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def run_random_forest(
    df: pd.DataFrame,
    train_on_subclass: bool    = True,
    use_teff: bool             = True,
    use_smote: bool            = True,
    use_threshold_tuning: bool = True,
    min_samples: int           = 20,
    n_estimators: int          = 300,
    cross_validate: bool       = True,
    random_state: int          = 42,
) -> ClassificationResult:
    """
    Random Forest classifier.

    Trains on spectral_subclass or spectral_class depending on train_on_subclass.
    y_true / y_pred in the returned result are ALWAYS coarse letters.
    """
    target   = "spectral_subclass" if train_on_subclass else "spectral_class"
    features = _get_features(use_teff)
    label    = "subtypes" if train_on_subclass else "coarse classes"
    print(f"\n[Random Forest] Training on {label}..."
          + (" + T_eff" if use_teff else "")
          + (" + SMOTE" if use_smote else "")
          + (" + threshold tuning" if use_threshold_tuning else ""))

    X_train, X_val, X_test, y_train, y_val, y_test = _prepare_data(
        df, features, target, min_samples, random_state=random_state
    )

    if use_smote:
        X_train, y_train = _apply_smote(X_train, y_train, random_state)

    # Label universe from actual training data — no gaps
    all_classes = sorted(np.unique(y_train))

    clf = RandomForestClassifier(
        n_estimators = n_estimators,
        class_weight = "balanced",
        random_state = random_state,
        n_jobs       = -1,
    )
    clf.fit(X_train, y_train)

    # Threshold tuning on validation set (coarse level)
    weights = None
    if use_threshold_tuning:
        y_val_coarse = _coarsen(y_val) if train_on_subclass else y_val
        weights = _tune_thresholds(
            clf, X_val, y_val, all_classes, y_val_coarse, random_state
        )

    # Predict on test set
    if weights is not None:
        proba_test = clf.predict_proba(X_test)
        y_pred_raw = _apply_weights(proba_test, weights, all_classes)
    else:
        y_pred_raw = clf.predict(X_test)

    # Coarsen both true and predicted labels for evaluation
    y_true_coarse = _coarsen(y_test) if train_on_subclass else np.array(y_test)
    y_pred_coarse = _coarsen(y_pred_raw) if train_on_subclass else np.array(y_pred_raw)

    # Filter to known spectral classes
    mask = np.isin(y_true_coarse, SPECTRAL_CLASSES)

    cv_scores = None
    if cross_validate:
        print("      Running 5-fold cross-validation...")
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5,
                                    scoring="f1_weighted", n_jobs=-1)

    suffix = " (subclass train)" if train_on_subclass else " (coarse train)"
    result = ClassificationResult(
        model_name    = "Random Forest" + suffix,
        y_true        = y_true_coarse[mask],
        y_pred        = y_pred_coarse[mask],
        classes       = SPECTRAL_CLASSES,
        model         = clf,
        feature_names = features,
        X_test        = X_test,
        cv_scores     = cv_scores,
    )
    print_result(result)
    return result


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def run_xgboost(
    df: pd.DataFrame,
    train_on_subclass: bool    = True,
    use_teff: bool             = True,
    use_smote: bool            = True,
    use_threshold_tuning: bool = True,
    min_samples: int           = 20,
    cross_validate: bool       = True,
    random_state: int          = 42,
) -> ClassificationResult:
    """
    XGBoost classifier.

    LabelEncoder is fit on y_train post-filtering so integers are always
    consecutive [0..N-1] — XGBoost crashes if there are gaps.
    y_true / y_pred in the returned result are ALWAYS coarse letters.
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    target   = "spectral_subclass" if train_on_subclass else "spectral_class"
    features = _get_features(use_teff)
    label    = "subtypes" if train_on_subclass else "coarse classes"
    print(f"\n[XGBoost] Training on {label}..."
          + (" + T_eff" if use_teff else "")
          + (" + SMOTE" if use_smote else "")
          + (" + threshold tuning" if use_threshold_tuning else ""))

    X_train, X_val, X_test, y_train, y_val, y_test = _prepare_data(
        df, features, target, min_samples, random_state=random_state
    )

    if use_smote:
        X_train, y_train = _apply_smote(X_train, y_train, random_state)

    # LabelEncoder on y_train only — guarantees consecutive integers
    all_classes = sorted(np.unique(y_train))
    le = LabelEncoder()
    le.fit(all_classes)

    y_train_enc = le.transform(y_train)
    y_val_enc   = le.transform(y_val)

    clf = XGBClassifier(
        n_estimators  = 300,
        learning_rate = 0.1,
        max_depth     = 6,
        eval_metric   = "mlogloss",
        random_state  = random_state,
        n_jobs        = -1,
    )
    clf.fit(X_train, y_train_enc)

    # Threshold tuning — operate in decoded label space on val set
    weights = None
    if use_threshold_tuning:
        y_val_coarse = _coarsen(y_val) if train_on_subclass else np.array(y_val)
        weights = _tune_thresholds(
            clf, X_val, y_val_enc, all_classes, y_val_coarse, random_state
        )

    # Predict on test set
    if weights is not None:
        proba_test = clf.predict_proba(X_test)
        y_pred_raw = _apply_weights(proba_test, weights, all_classes)
    else:
        y_pred_enc = clf.predict(X_test)
        y_pred_raw = le.inverse_transform(y_pred_enc)

    # Coarsen for evaluation
    y_true_coarse = _coarsen(y_test) if train_on_subclass else np.array(y_test)
    y_pred_coarse = _coarsen(y_pred_raw) if train_on_subclass else np.array(y_pred_raw)

    mask = np.isin(y_true_coarse, SPECTRAL_CLASSES)

    cv_scores = None
    if cross_validate:
        print("      Running 5-fold cross-validation...")
        cv_scores = cross_val_score(clf, X_train, y_train_enc, cv=5,
                                    scoring="f1_weighted", n_jobs=-1)

    suffix = " (subclass train)" if train_on_subclass else " (coarse train)"
    result = ClassificationResult(
        model_name    = "XGBoost" + suffix,
        y_true        = y_true_coarse[mask],
        y_pred        = y_pred_coarse[mask],
        classes       = SPECTRAL_CLASSES,
        model         = clf,
        feature_names = features,
        X_test        = X_test,
        cv_scores     = cv_scores,
    )
    print_result(result)
    return result


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

def run_mlp(
    df: pd.DataFrame,
    train_on_subclass: bool = True,
    use_teff: bool          = True,
    use_smote: bool         = True,
    min_samples: int        = 20,
    hidden_layers: tuple    = (128, 64, 32),
    cross_validate: bool    = False,
    random_state: int       = 42,
) -> ClassificationResult:
    """
    MLP classifier with StandardScaler.

    No threshold tuning for MLP — its softmax output is already well-calibrated
    and threshold tuning on top of it rarely helps. SMOTE is applied if enabled.
    y_true / y_pred in the returned result are ALWAYS coarse letters.
    """
    target   = "spectral_subclass" if train_on_subclass else "spectral_class"
    features = _get_features(use_teff)
    label    = "subtypes" if train_on_subclass else "coarse classes"
    print(f"\n[MLP] Training on {label}..."
          + (" + T_eff" if use_teff else "")
          + (" + SMOTE" if use_smote else ""))

    X_train, X_val, X_test, y_train, y_val, y_test = _prepare_data(
        df, features, target, min_samples, random_state=random_state
    )

    if use_smote:
        X_train, y_train = _apply_smote(X_train, y_train, random_state)

    # LabelEncoder on y_train — consecutive integers, avoids sklearn bug
    all_classes = sorted(np.unique(y_train))
    le = LabelEncoder()
    le.fit(all_classes)

    y_train_enc = le.transform(y_train)
    y_test_enc  = le.transform(y_test)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes = hidden_layers,
            activation         = "relu",
            max_iter           = 500,
            early_stopping     = True,
            validation_fraction= 0.1,
            random_state       = random_state,
        ))
    ])
    pipeline.fit(X_train, y_train_enc)
    y_pred_enc = pipeline.predict(X_test)
    y_pred_raw = le.inverse_transform(y_pred_enc)

    # Coarsen for evaluation
    y_true_coarse = _coarsen(y_test) if train_on_subclass else np.array(y_test)
    y_pred_coarse = _coarsen(y_pred_raw) if train_on_subclass else np.array(y_pred_raw)

    mask = np.isin(y_true_coarse, SPECTRAL_CLASSES)

    cv_scores = None
    if cross_validate:
        print("      Running 5-fold cross-validation (slow for MLP)...")
        cv_scores = cross_val_score(pipeline, X_train, y_train_enc, cv=5,
                                    scoring="f1_weighted", n_jobs=-1)

    suffix = " (subclass train)" if train_on_subclass else " (coarse train)"
    result = ClassificationResult(
        model_name    = f"MLP {hidden_layers}" + suffix,
        y_true        = y_true_coarse[mask],
        y_pred        = y_pred_coarse[mask],
        classes       = SPECTRAL_CLASSES,
        model         = pipeline,
        feature_names = features,
        X_test        = X_test,
        cv_scores     = cv_scores,
    )
    print_result(result)
    return result


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def run_all(
    df: pd.DataFrame,
    train_on_subclass: bool    = True,
    use_teff: bool             = True,
    use_smote: bool            = True,
    use_threshold_tuning: bool = True,
    min_samples: int           = 20,
) -> dict[str, ClassificationResult]:
    """
    Run baseline + all ML models.
    All ClassificationResults have coarse y_true / y_pred regardless of flags.
    """
    kwargs = dict(
        train_on_subclass    = train_on_subclass,
        use_teff             = use_teff,
        use_smote            = use_smote,
        min_samples          = min_samples,
    )

    results = {}
    results["baseline"]      = run_baseline(df)
    results["random_forest"] = run_random_forest(
        df, use_threshold_tuning=use_threshold_tuning, **kwargs
    )
    results["mlp"] = run_mlp(df, **kwargs)

    if XGBOOST_AVAILABLE:
        results["xgboost"] = run_xgboost(
            df, use_threshold_tuning=use_threshold_tuning, **kwargs
        )
    else:
        print("Skipping XGBoost (not installed).")

    return results
