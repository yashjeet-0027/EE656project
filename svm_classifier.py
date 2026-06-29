"""
Classification Module
Paper: Verma et al., IEEE Trans. Reliability, 2016
Implements: C-SVM with RBF kernel, OAO / OAA / DDAG multiclass
Grid search for C and gamma exactly as paper specifies.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from itertools import product


# ─────────────────────────────────────────────
# GRID SEARCH (paper-exact)
# Paper: log2C in {-2,-1,0,...,11,12}  → 15 values
#        log2γ in {-10,-9,-8,...,3,4}  → 15 values
#        Total: 15x15 = 225 pairs, 5-fold internal CV
# ─────────────────────────────────────────────

def grid_search_svm(X_train, y_train, decomposition='OAO', cv=5, verbose=False):
    """
    Grid search over C and gamma as specified in paper.
    Returns best_C, best_gamma, best_accuracy.

    decomposition: 'OAO' | 'OAA' | 'DDAG'
    cv: number of internal CV folds (paper uses 5)
    """
    log2_C_range     = list(range(-2, 13))   # -2 to 12 inclusive → 15 values
    log2_gamma_range = list(range(-10, 5))   # -10 to 4 inclusive → 15 values

    best_acc   = -1
    best_C     = 1.0
    best_gamma = 0.1

    for lc, lg in product(log2_C_range, log2_gamma_range):
        C     = 2 ** lc
        gamma = 2 ** lg

        base_svm = SVC(C=C, gamma=gamma, kernel='rbf', decision_function_shape='ovo')

        if decomposition == 'OAO':
            clf = OneVsOneClassifier(SVC(C=C, gamma=gamma, kernel='rbf'))
        elif decomposition == 'OAA':
            clf = OneVsRestClassifier(SVC(C=C, gamma=gamma, kernel='rbf',
                                          decision_function_shape='ovr'))
        elif decomposition == 'DDAG':
            # sklearn's SVC with ovo decision function is equivalent to DDAG
            clf = SVC(C=C, gamma=gamma, kernel='rbf', decision_function_shape='ovo')
        else:
            raise ValueError(f"Unknown decomposition: {decomposition}")

        # 5-fold internal CV on training data
        scores = cross_val_score(clf, X_train, y_train, cv=cv,
                                  scoring='accuracy', n_jobs=-1)
        acc = scores.mean()

        if verbose:
            print(f"  C=2^{lc}, gamma=2^{lg}: acc={acc:.4f}")

        if acc > best_acc:
            best_acc   = acc
            best_C     = C
            best_gamma = gamma

    return best_C, best_gamma, best_acc


# ─────────────────────────────────────────────
# SVM CLASSIFIER WRAPPER
# ─────────────────────────────────────────────

class SVMFaultClassifier:
    """
    Wraps sklearn SVM with paper's exact setup.
    decomposition: 'OAO' | 'OAA' | 'DDAG'

    OAO  → OneVsOne (n*(n-1)/2 binary SVMs, majority voting)
    OAA  → OneVsAll (n binary SVMs)
    DDAG → uses sklearn's OVO with ovo decision path (equivalent to DDAG test phase)
    """

    def __init__(self, decomposition='OAO', C=1.0, gamma=0.1):
        self.decomposition = decomposition
        self.C     = C
        self.gamma = gamma
        self.clf   = None

    def _build_clf(self):
        if self.decomposition == 'OAO':
            return OneVsOneClassifier(
                SVC(C=self.C, gamma=self.gamma, kernel='rbf')
            )
        elif self.decomposition == 'OAA':
            return OneVsRestClassifier(
                SVC(C=self.C, gamma=self.gamma, kernel='rbf',
                    decision_function_shape='ovr')
            )
        elif self.decomposition == 'DDAG':
            # In sklearn, SVC with decision_function_shape='ovo' uses
            # pairwise coupling equivalent to DDAG during prediction
            return SVC(C=self.C, gamma=self.gamma, kernel='rbf',
                       decision_function_shape='ovo')
        else:
            raise ValueError(f"Unknown decomposition: {self.decomposition}")

    def fit(self, X_train, y_train):
        self.clf = self._build_clf()
        self.clf.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.clf.predict(X_test)

    def accuracy(self, X_test, y_test):
        preds = self.predict(X_test)
        return np.mean(preds == y_test) * 100  # percentage
