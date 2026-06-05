"""
Star_Classification.py
-----------------------
Main entry point. All experiment options are set in the CONFIG block below.

Usage:
    python Star_Classification.py
    python Star_Classification.py --n 50000
    python Star_Classification.py --n 100000 --out data/
"""

import argparse
import os
from fetch_sdss import fetch_stars, clean_and_engineer, summarize, save
from models import run_all
from evaluate import run_evaluation

# =============================================================================
# CONFIG — edit these flags to switch experiment modes
# =============================================================================

# Train on granular subtypes ("G2", "K1") instead of coarse letters ("G", "K").
# Evaluation is ALWAYS at the coarse class level regardless of this flag.
TRAIN_ON_SUBCLASS = True

# Add elodieTEff (spectroscopic temperature) as a training feature.
# Powerful signal for F/G/K separation (~6000K boundary).
# Rows with missing T_eff are dropped when this is True.
# Criticism: Point of project is to avoid spectroscopy, so using T_eff is a bit of a cheat. But it's interesting to see how much it helps.
USE_TEFF = False

# Oversample minority classes in training using SMOTE.
# Helps with F/G/K imbalance. Requires: pip install imbalanced-learn
# Criticism: SMOTE is a bit of a cheat since it creates synthetic data rather than using real observations.
#            Especially as some star classifications are genuinely rarer
USE_SMOTE = False

# Post-hoc threshold calibration to reduce F overprediction / G underprediction.
# Adjusts per-class decision thresholds on a held-out validation set after training.
# Applied to RF and XGBoost only (MLP uses its own softmax probabilities).
USE_THRESHOLD_TUNING = True

# Minimum number of training examples required per class/subclass.
# Classes below this threshold are dropped before training.
MIN_SAMPLES = 20

# Output directory for CSVs (figures go to a "figures/" subfolder, handled in
# evaluate.py). Defaults to the folder this script lives in, so the pipeline is
# fully portable: clone the repo anywhere and outputs land next to the code.
DEFAULT_OUT = os.path.dirname(os.path.abspath(__file__))

# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stellar classification pipeline.")
    parser.add_argument("--n",   type=int, default=100_000,
                        help="Max rows to fetch from SDSS (default: 100000)")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT,
                        help="Output directory for CSVs and figures")
    args = parser.parse_args()

    # Print active config so runs are reproducible from the console log
    print("\n" + "=" * 55)
    print("CONFIG")
    print("=" * 55)
    print(f"  TRAIN_ON_SUBCLASS    : {TRAIN_ON_SUBCLASS}")
    print(f"  USE_TEFF             : {USE_TEFF}")
    print(f"  USE_SMOTE            : {USE_SMOTE}")
    print(f"  USE_THRESHOLD_TUNING : {USE_THRESHOLD_TUNING}")
    print(f"  MIN_SAMPLES          : {MIN_SAMPLES}")
    print(f"  Evaluation           : always coarse OBAFGKM + FGKM")
    print("=" * 55 + "\n")

    # 1-4: Fetch, clean, summarize, save
    df_raw = fetch_stars(n=args.n)
    df     = clean_and_engineer(df_raw)
    summarize(df)
    save(df, args.out)

    # 5: Train all models
    results = run_all(
        df,
        train_on_subclass    = TRAIN_ON_SUBCLASS,
        use_teff             = USE_TEFF,
        use_smote            = USE_SMOTE,
        use_threshold_tuning = USE_THRESHOLD_TUNING,
        min_samples          = MIN_SAMPLES,
    )

    # 6: Evaluate — always coarse, always class-vs-class
    run_evaluation(df, results, out_dir=args.out)


if __name__ == "__main__":
    main()
