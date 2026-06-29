"""
Feature Selection Module
Paper: Verma et al., IEEE Trans. Reliability, 2016
Implements: PCA, MIFS, mRMR, NMIFS, MIFS-U, Bhattacharyya Distance
Optimized: O(N*K) incremental state caching for all MI-based algorithms.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────
# UTILITY: Entropy & Mutual Information
# ─────────────────────────────────────────────

def _histogram_entropy(x, n_bins=15):
    """
    Entropy of a 1D array using histogram-based probability estimation.
    Paper Section VI-B: 15 equal-width bins used.
    """
    x_min, x_max = np.min(x), np.max(x)
    if x_max == x_min:
        return 0.0
    counts, _ = np.histogram(x, bins=n_bins, range=(x_min, x_max))
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # avoid log(0)
    return -np.sum(probs * np.log2(probs))


def _mutual_information(x, y, n_bins=15):
    """
    MI between feature x and class labels y.
    Uses joint histogram to estimate joint probability.
    """
    # Joint histogram
    x_min, x_max = np.min(x), np.max(x)
    y_vals = np.unique(y)
    n_y = len(y_vals)

    # MI = H(X) - H(X|Y)  =  sum_y p(y) * sum_x p(x|y)*log(p(x|y)/p(x))
    h_x = _histogram_entropy(x, n_bins)

    h_x_given_y = 0.0
    for yv in y_vals:
        mask = (y == yv)
        p_y = mask.sum() / len(y)
        if p_y > 0 and mask.sum() > 1:
            h_x_given_y += p_y * _histogram_entropy(x[mask], n_bins)

    return max(0.0, h_x - h_x_given_y)


def _feature_feature_mi(xi, xj, n_bins=15):
    """MI between two continuous features using 2D histogram."""
    counts, _, _ = np.histogram2d(xi, xj, bins=n_bins)
    joint = counts / counts.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    pmi = joint[mask] * np.log2(joint[mask] / (px * py + 1e-12)[mask])
    return max(0.0, pmi.sum())


# ─────────────────────────────────────────────
# 1. PCA
# ─────────────────────────────────────────────

class PCASelector:
    """
    PCA: transforms features to principal component space.
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit(self, X_train, y_train=None):
        self.pca.fit(X_train)
        return self

    def transform(self, X):
        return self.pca.transform(X)

    def fit_transform_data(self, X_train, X_test, y_train=None):
        self.fit(X_train)
        return self.transform(X_train), self.transform(X_test)


# ─────────────────────────────────────────────
# 2. MIFS (Mutual Information Feature Selection)
# ─────────────────────────────────────────────

class MIFSSelector:
    """
    MIFS — Battiti 1994.
    Optimized to O(N*K) using incremental running redundancy.
    """
    def __init__(self, n_features, beta=1.0):
        self.n_features = n_features
        self.beta = beta
        self.selected_indices_ = []

    def fit(self, X_train, y_train):
        n_total = X_train.shape[1]
        mi_class = np.array([_mutual_information(X_train[:, j], y_train) for j in range(n_total)])

        selected = []
        first_feat = int(np.argmax(mi_class))
        selected.append(first_feat)

        running_redundancy = np.zeros(n_total)
        last_selected = first_feat

        while len(selected) < self.n_features:
            
            # Incremental Update: Only calculate MI against the single newest feature
            for j in range(n_total):
                if j not in selected:
                    running_redundancy[j] += _feature_feature_mi(X_train[:, j], X_train[:, last_selected])

            best_score = -np.inf
            best_feat = -1
            
            # Fast O(1) lookup
            for j in range(n_total):
                if j in selected:
                    continue
                
                score = mi_class[j] - self.beta * running_redundancy[j]
                if score > best_score:
                    best_score = score
                    best_feat = j
                    
            selected.append(best_feat)
            last_selected = best_feat

        self.selected_indices_ = selected
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def fit_transform_data(self, X_train, X_test, y_train):
        self.fit(X_train, y_train)
        return self.transform(X_train), self.transform(X_test)


# ─────────────────────────────────────────────
# 3. mRMR (Minimum Redundancy Maximum Relevance)
# ─────────────────────────────────────────────

class mRMRSelector:
    """
    mRMR — Peng et al. 2005.
    Optimized to O(N*K) using incremental running redundancy.
    """
    def __init__(self, n_features):
        self.n_features = n_features
        self.selected_indices_ = []

    def fit(self, X_train, y_train):
        n_total = X_train.shape[1]
        mi_class = np.array([_mutual_information(X_train[:, j], y_train) for j in range(n_total)])

        selected = []
        first_feat = int(np.argmax(mi_class))
        selected.append(first_feat)

        running_redundancy = np.zeros(n_total)
        last_selected = first_feat

        while len(selected) < self.n_features:
            beta = 1.0 / len(selected)
            
            # Incremental Update
            for j in range(n_total):
                if j not in selected:
                    running_redundancy[j] += _feature_feature_mi(X_train[:, j], X_train[:, last_selected])

            best_score = -np.inf
            best_feat = -1
            
            # Fast O(1) lookup
            for j in range(n_total):
                if j in selected:
                    continue
                
                score = mi_class[j] - beta * running_redundancy[j]
                if score > best_score:
                    best_score = score
                    best_feat = j
                    
            selected.append(best_feat)
            last_selected = best_feat

        self.selected_indices_ = selected
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def fit_transform_data(self, X_train, X_test, y_train):
        self.fit(X_train, y_train)
        return self.transform(X_train), self.transform(X_test)


# ─────────────────────────────────────────────
# 4. NMIFS (Normalized Mutual Information FS)
# ─────────────────────────────────────────────

class NMIFSSelector:
    """
    NMIFS — Estevez et al. 2009.
    Optimized to O(N*K) using incremental running redundancy.
    """
    def __init__(self, n_features):
        self.n_features = n_features
        self.selected_indices_ = []

    def _nmi(self, xi, xj):
        mi = _feature_feature_mi(xi, xj)
        hi = _histogram_entropy(xi)
        hj = _histogram_entropy(xj)
        denom = np.sqrt(hi * hj)
        return mi / denom if denom > 1e-12 else 0.0

    def fit(self, X_train, y_train):
        n_total = X_train.shape[1]
        mi_class = np.array([_mutual_information(X_train[:, j], y_train) for j in range(n_total)])

        selected = []
        first_feat = int(np.argmax(mi_class))
        selected.append(first_feat)

        running_redundancy = np.zeros(n_total)
        last_selected = first_feat

        while len(selected) < self.n_features:
            
            # Incremental Update using Normalized MI
            for j in range(n_total):
                if j not in selected:
                    running_redundancy[j] += self._nmi(X_train[:, j], X_train[:, last_selected])

            best_score = -np.inf
            best_feat = -1
            
            for j in range(n_total):
                if j in selected:
                    continue
                
                score = mi_class[j] - running_redundancy[j]
                if score > best_score:
                    best_score = score
                    best_feat = j
                    
            selected.append(best_feat)
            last_selected = best_feat

        self.selected_indices_ = selected
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def fit_transform_data(self, X_train, X_test, y_train):
        self.fit(X_train, y_train)
        return self.transform(X_train), self.transform(X_test)


# ─────────────────────────────────────────────
# 5. MIFS-U (MIFS under Uniform distribution)
# ─────────────────────────────────────────────

class MIFSUSelector:
    """
    MIFS-U — Kwak & Choi 2002.
    Optimized to O(N*K) using incremental running redundancy.
    """
    def __init__(self, n_features):
        self.n_features = n_features
        self.selected_indices_ = []

    def fit(self, X_train, y_train):
        n_total = X_train.shape[1]
        mi_class = np.array([_mutual_information(X_train[:, j], y_train) for j in range(n_total)])

        selected = []
        first_feat = int(np.argmax(mi_class))
        selected.append(first_feat)

        running_redundancy = np.zeros(n_total)
        last_selected = first_feat
        
        # Track the sum of entropies to calculate mean H(S) without looping
        running_h_s_sum = _histogram_entropy(X_train[:, last_selected])

        while len(selected) < self.n_features:
            
            # Incremental Update
            for j in range(n_total):
                if j not in selected:
                    running_redundancy[j] += _feature_feature_mi(X_train[:, j], X_train[:, last_selected])

            # O(1) H(S) calculation
            h_s = max(running_h_s_sum / len(selected), 1e-12)

            best_score = -np.inf
            best_feat = -1
            
            for j in range(n_total):
                if j in selected:
                    continue
                
                score = mi_class[j] - (1.0 / h_s) * running_redundancy[j]
                if score > best_score:
                    best_score = score
                    best_feat = j
                    
            selected.append(best_feat)
            last_selected = best_feat
            running_h_s_sum += _histogram_entropy(X_train[:, last_selected])

        self.selected_indices_ = selected
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def fit_transform_data(self, X_train, X_test, y_train):
        self.fit(X_train, y_train)
        return self.transform(X_train), self.transform(X_test)


# ─────────────────────────────────────────────
# 6. Bhattacharyya Distance (BD)
# ─────────────────────────────────────────────

class BDSelector:
    """
    Bhattacharyya Distance feature selection.
    """
    def __init__(self, n_features):
        self.n_features = n_features
        self.selected_indices_ = []

    def _bd_feature(self, xi, y):
        """Mean Bhattacharyya distance of feature xi across all class pairs."""
        classes = np.unique(y)
        n_cls = len(classes)
        if n_cls < 2:
            return 0.0

        total_bd = 0.0
        count = 0
        for i in range(n_cls):
            for j in range(i + 1, n_cls):
                xi_i = xi[y == classes[i]]
                xi_j = xi[y == classes[j]]
                mu_i, var_i = np.mean(xi_i), np.var(xi_i) + 1e-12
                mu_j, var_j = np.mean(xi_j), np.var(xi_j) + 1e-12
                # Univariate Bhattacharyya distance (Gaussian assumption)
                term1 = 0.25 * ((mu_i - mu_j) ** 2) / (var_i + var_j)
                term2 = 0.5 * np.log(0.5 * (var_i / var_j + var_j / var_i + 2))
                total_bd += term1 + term2
                count += 1

        return total_bd / count if count > 0 else 0.0

    def fit(self, X_train, y_train):
        n_total = X_train.shape[1]
        bd_scores = np.array([self._bd_feature(X_train[:, j], y_train)
                               for j in range(n_total)])

        selected = []
        # First: feature with max mean BD
        selected.append(int(np.argmax(bd_scores)))

        while len(selected) < self.n_features:
            best_score = -np.inf
            best_feat = -1
            for j in range(n_total):
                if j in selected:
                    continue
                # Mean BD of feature j from all selected features
                mean_bd_from_selected = np.mean([
                    self._bd_feature(
                        np.column_stack([X_train[:, j], X_train[:, s]]).mean(axis=1),
                        y_train
                    )
                    for s in selected
                ])
                if mean_bd_from_selected > best_score:
                    best_score = mean_bd_from_selected
                    best_feat = j
            selected.append(best_feat)

        self.selected_indices_ = selected
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def fit_transform_data(self, X_train, X_test, y_train):
        self.fit(X_train, y_train)
        return self.transform(X_train), self.transform(X_test)


# ─────────────────────────────────────────────
# FACTORY: get selector by name
# ─────────────────────────────────────────────

def get_selector(method, n_features):
    """
    Returns selector instance by name.
    method: 'PCA', 'MIFS', 'mRMR', 'NMIFS', 'MIFS-U', 'BD'
    """
    methods = {
        'PCA':    lambda n: PCASelector(n),
        'MIFS':   lambda n: MIFSSelector(n),
        'mRMR':   lambda n: mRMRSelector(n),
        'NMIFS':  lambda n: NMIFSSelector(n),
        'MIFS-U': lambda n: MIFSUSelector(n),
        'BD':     lambda n: BDSelector(n),
    }
    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(methods.keys())}")
    return methods[method](n_features)