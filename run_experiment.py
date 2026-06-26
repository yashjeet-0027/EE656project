"""
Main Experiment Pipeline
Paper: Verma et al., IEEE Trans. Reliability, 2016
Section VIII: Case Study

Reproduces Tables III–VII (OAO results, 2/3/4/5/10 fold)
for all 6 feature selection methods and various n_features.

Usage:
    python run_experiment.py

Input expected:
    features.npy  — shape (1800, 286) or (1800, 629)
    labels.npy    — shape (1800,) with values 0–7

If you don't have real data yet, a synthetic dataset is generated
so you can test/debug the pipeline immediately.
"""

import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold

from feature_selection.selectors import get_selector
from classification.svm_classifier import SVMFaultClassifier, grid_search_svm


# ─────────────────────────────────────────────
# CONFIG — match paper exactly
# ─────────────────────────────────────────────

FEATURE_SELECTION_METHODS = ['PCA', 'MIFS', 'mRMR', 'NMIFS', 'MIFS-U', 'BD']

# Paper tests these n_features values (subset for speed; add more as needed)
N_FEATURES_LIST = [5, 10, 15, 25, 50, 75, 100, 286]

# k-fold values from paper
K_FOLDS = [2, 3, 4, 5, 10]

# Multiclass decompositions
DECOMPOSITIONS = ['OAO', 'OAA', 'DDAG']

# Internal CV folds for grid search
INTERNAL_CV = 5

# Number of classes
N_CLASSES = 8   # 1 healthy + 7 faulty


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_data():
    """
    Load feature matrix and labels.
    Falls back to synthetic data for testing.
    """
    feat_path  = 'features.npy'
    label_path = 'labels.npy'

    if os.path.exists(feat_path) and os.path.exists(label_path):
        X = np.load(feat_path)
        y = np.load(label_path)
        print(f"Loaded real data: X={X.shape}, y={y.shape}")
    else:
        print("⚠  Real data not found. Generating synthetic data for testing.")
        print("   Replace 'features.npy' and 'labels.npy' with your actual data.\n")
        # 1800 samples, 8 classes (225 per class), 286 features
        np.random.seed(42)
        X = np.random.randn(1800, 286)
        # Make classes slightly separable by adding class-specific offsets
        y = np.repeat(np.arange(N_CLASSES), 1800 // N_CLASSES)
        for c in range(N_CLASSES):
            X[y == c, c * 10:(c + 1) * 10] += 2.0   # signal
        print(f"Synthetic data: X={X.shape}, y={y.shape}, classes={N_CLASSES}\n")

    return X, y


# ─────────────────────────────────────────────
# SINGLE RUN: one (method, n_feat, k, decomp)
# ─────────────────────────────────────────────

def run_single(X, y, method, n_features, k_fold, decomposition):
    """
    Runs k-fold CV for a given configuration.
    Returns: mean accuracy (%) across k folds.

    Steps per fold (exactly as paper):
    1. Split into train/test
    2. Feature selection on TRAIN only
    3. Transform train and test
    4. Grid search C, gamma on train (5-fold internal CV)
    5. Train SVM on full train with best C, gamma
    6. Evaluate on test
    """
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ── Feature Selection ──
        # For PCA: n_features must not exceed min(n_samples, n_original_features)
        actual_n = min(n_features, X_train.shape[1], X_train.shape[0] - 1)

        selector = get_selector(method, actual_n)
        X_train_sel, X_test_sel = selector.fit_transform_data(X_train, X_test, y_train)

        # ── Grid Search (on training data only) ──
        best_C, best_gamma, _ = grid_search_svm(
            X_train_sel, y_train,
            decomposition=decomposition,
            cv=INTERNAL_CV
        )

        # ── Train SVM on full training set ──
        clf = SVMFaultClassifier(
            decomposition=decomposition,
            C=best_C,
            gamma=best_gamma
        )
        clf.fit(X_train_sel, y_train)

        # ── Evaluate ──
        acc = clf.accuracy(X_test_sel, y_test)
        fold_accuracies.append(acc)

    return np.mean(fold_accuracies)


# ─────────────────────────────────────────────
# FULL EXPERIMENT
# ─────────────────────────────────────────────

def run_experiment(X, y,
                   methods=None,
                   n_features_list=None,
                   k_folds=None,
                   decompositions=None,
                   quick_mode=False):
    """
    Runs the full experiment matrix.

    quick_mode=True: runs a fast subset for debugging
                     (1 method, 2 n_feature values, 1 fold, 1 decomp)
    """
    if methods        is None: methods        = FEATURE_SELECTION_METHODS
    if n_features_list is None: n_features_list = N_FEATURES_LIST
    if k_folds        is None: k_folds        = K_FOLDS
    if decompositions is None: decompositions  = DECOMPOSITIONS

    if quick_mode:
        print("🔧 QUICK MODE: running small subset for testing\n")
        methods         = ['mRMR']
        n_features_list = [5, 25]
        k_folds         = [2, 5]
        decompositions  = ['OAO']

    results = {}   # key: (method, n_feat, k_fold, decomp) → accuracy

    total = len(methods) * len(n_features_list) * len(k_folds) * len(decompositions)
    done  = 0

    for method in methods:
        for decomp in decompositions:
            for k in k_folds:
                for n_feat in n_features_list:
                    key = (method, n_feat, k, decomp)
                    t0  = time.time()

                    try:
                        acc = run_single(X, y, method, n_feat, k, decomp)
                        results[key] = acc
                        elapsed = time.time() - t0
                        done += 1
                        print(f"[{done}/{total}] {method:6s} | {decomp:4s} | "
                              f"k={k:2d} | n_feat={n_feat:3d} → acc={acc:.2f}% "
                              f"({elapsed:.1f}s)")
                    except Exception as e:
                        results[key] = None
                        print(f"[{done}/{total}] {method:6s} | {decomp:4s} | "
                              f"k={k:2d} | n_feat={n_feat:3d} → ERROR: {e}")
                        done += 1

    return results


# ─────────────────────────────────────────────
# PRINT RESULTS TABLE (like paper's Tables III-VII)
# ─────────────────────────────────────────────

def print_table(results, decomposition='OAO', k_fold=5,
                n_features_list=None, methods=None):
    """
    Prints accuracy table for a given decomposition and k_fold.
    Matches paper Table format.
    """
    if methods         is None: methods         = FEATURE_SELECTION_METHODS
    if n_features_list is None: n_features_list = N_FEATURES_LIST

    print(f"\n{'='*70}")
    print(f"  OAO={decomposition} | k={k_fold}-Fold Cross Validation")
    print(f"{'='*70}")

    header = f"{'n_feat':>8} | " + " | ".join(f"{m:>8}" for m in methods)
    print(header)
    print('-' * len(header))

    for n_feat in n_features_list:
        row = f"{n_feat:>8} | "
        for method in methods:
            key = (method, n_feat, k_fold, decomposition)
            val = results.get(key, None)
            row += f"{val:>8.2f} | " if val is not None else f"{'N/A':>8} | "
        print(row)

    print(f"{'='*70}\n")


# ─────────────────────────────────────────────
# SAVE / LOAD RESULTS
# ─────────────────────────────────────────────

def save_results(results, path='results.npy'):
    np.save(path, results)
    print(f"Results saved to {path}")

def load_results(path='results.npy'):
    return np.load(path, allow_pickle=True).item()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: small subset for testing')
    parser.add_argument('--decomp', default='OAO',
                        choices=['OAO', 'OAA', 'DDAG'],
                        help='Multiclass decomposition to run')
    parser.add_argument('--method', default=None,
                        help='Single feature selection method to run '
                             '(PCA/MIFS/mRMR/NMIFS/MIFS-U/BD), or None for all')
    args = parser.parse_args()

    # Load data
    X, y = load_data()

    # Run
    methods = [args.method] if args.method else None
    results = run_experiment(
        X, y,
        methods=methods,
        decompositions=[args.decomp],
        quick_mode=args.quick
    )

    # Print table
    for k in K_FOLDS:
        print_table(results, decomposition=args.decomp, k_fold=k)

    # Save
    save_results(results)
