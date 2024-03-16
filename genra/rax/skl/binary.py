"""
Module to define BinaryMixin that supports binary calculations - Similarity Weighted Activity (SWA), AUC, pval, and optimum threshold.
"""

import warnings

import numpy as np
import pandas as pd
import sklearn.metrics as metrics

from genra.rax.skl.cls import GenRAPredClass


def calc_uncertainty_from_neigh_swa(y, neigh_swa, N=100):
    """Uncertainty calculations, taken and slightly modified (removed pos_label) from previous genrapred module."""

    # get AUC
    with warnings.catch_warnings():
        fpr, tpr, t0 = metrics.roc_curve(
            y,
            neigh_swa,
            drop_intermediate=False,
        )
    auc = metrics.auc(fpr, tpr)

    # get optimized threshold
    tnr = 1 - fpr
    Roc = pd.DataFrame(zip(fpr, tnr, tpr, t0), columns=["fpr", "sp", "sn", "t0"])
    Roc["BA"] = 0.5 * (Roc.sp + Roc.sn)
    Roc0 = Roc.query("t0<=1")
    idx_max = Roc0.BA.idxmax()
    t0 = Roc0.t0.loc[idx_max]

    # get p_val
    AUC = []
    for i in range(N):
        y = np.array(y.copy())
        np.random.shuffle(y)
        fpr, tpr, _ = metrics.roc_curve(y, neigh_swa)
        AUC.append(metrics.auc(fpr, tpr))
    p_val = 1.0 * np.sum(np.array(AUC) > auc) / N

    return auc, p_val, t0


class BinaryMixin:
    """Mixin to support binary methods (similarity weighted activity/uncertainty)."""

    def _get_y_binary(self, pos_label):
        """Get self._y vector represented in binary form, {0,1}. Raise warning in the case of sparse data (only one class represented) or
        and exception if pos_label not represented in this bindary data.

        Parameters
        ----------
        pos_label : obj (default=1)
            The encoding of positive observations in the training data

        """
        y = self.classes_[self._y]
        y[y == pos_label] = 1
        y[y != 1] = 0
        return y.astype(int)

    def fit(self, X, y):
        """Call the parent class's fit method, then validate that the data is truly binary."""
        # call GenRAPredHybrid.fit(X,y)
        res = super().fit(X, y)
        # validation that it's truly binary data
        classes = self.classes_
        n_classes = len(classes)
        if n_classes > 2:
            raise Exception(
                "More than 2 classes are present in the data, unable to fit in a binary model"
            )
        elif n_classes == 1:
            warnings.warn(
                f"The data is homogenous - only one class is represented: {classes[0]}"
            )
        return res

    def get_max_distance(self):
        """Helper to convert from distances to similarity.
        If paring this Mixin with GenRAPredValue, should be considered together with GenRAPredValue.kneighbors_sim(...)
        method.

        For jaccard and cosine,
        """
        if self.metric in ["jaccard", "cosine"]:
            # because lowest similarity is 0 (corresponding to distance=1)
            max_distance = 1
        else:
            # e.g., maybe fill_value = pi for haversine distance?
            raise NotImplementedError(
                "Upper bound on distance must be set for this metric"
            )

        return max_distance

    def calc_neigh_swa(self, pos_label=1, include_self=False):
        """Returns size 1d numpy array of each neighbor's Similarity Weighted Activity (SWA).
        By default, itself is not included in its calculated SWA.
        NOTE: this method relies on implementation of self.kneighbors_graph(...)

        Parameters
        ----------
        pos_label : obj (default=1)
            The encoding of positive observations in the training data

        include_self : bool (default=False)
            Whether to include self in calculation of Similarity Weighted Activity

        """
        y = self._get_y_binary(pos_label=pos_label)
        # -1 because it gets incremented when X=None, this is default scikit behavior
        n = len(self._y) - 1
        # symmatric matrix of similarities
        S = (
            self.get_max_distance()
            - self.kneighbors_graph(
                X=None,
                n_neighbors=n,
                mode="distance",
            ).toarray()
        )
        if not include_self:
            # remove self from neighborhood
            np.fill_diagonal(S, 0)
        swa = np.dot(S, y)
        normalizer = np.sum(S, axis=1)
        # TODO: This needs to be addressed by PO, what to do when similarities are all 0 within neighborhood?
        #       Currently setting to 0.
        mask = normalizer != 0
        swa[mask] = swa[mask] / normalizer[mask]
        swa[normalizer == 0] = 0

        return swa

    def calc_uncertainty(self, include_self=False, N=100, pos_label=1):
        """Returns a tuple of Area Under Curve of the ROC curve, p-value, and optimized
        neighborhood threshold (t0). The optimized threshold is based on the maximum balanced accuracy.

        NOTE: this relies entirely on a class method and attribute: self.calc_swa() and self._y

        Parameters
        ----------
        include_self : bool (default=False)
            Whether to include self in calculation of Similarity Weighted Activity

        N : int (default=100)
            Number of permutations for testing significance of AUC

        pos_label : obj (default=1)
            The encoding of positive observations in the training data
        """
        if self.classes_.shape[0] < 2:
            # if both binary classes are not represented in data, then uncertainty cannot be calculated
            return np.nan, np.nan, np.nan

        neigh_swa = self.calc_neigh_swa(
            pos_label=pos_label,
            include_self=include_self,
        )
        y = self._get_y_binary(pos_label=pos_label)
        return calc_uncertainty_from_neigh_swa(y, neigh_swa, N=N)

    def predict_with_uncertainty(
        self,
        X,
        include_self=False,
        N=100,
        pos_label=1,
    ):
        """Runs predictions

        Parameters
        ----------
        X : list (array-like of shape (n_samples, n_features))
            Test samples

        include_self : bool (default=False)
            Whether to include self in calculation of Similarity Weighted Activity

        N : int (default=100)
            Number of permutations for testing significance of AUC

        pos_label : obj (default=1)
            The encoding of positive observations in the training data

        """
        proba = self.predict_proba(X)
        # self.classes_ has shape (k,) so np.where returns tuple of (array([...]),), so take first element
        pos_idx = np.where(self.classes_ == pos_label)[0]
        if len(pos_idx):
            # TODO: what if multidimensional?
            # take first elem
            pos_idx = pos_idx[0]
            # SWA is equivalent to normalized probability of positive observation
            swa = proba[0, pos_idx]
        else:
            # case: no positives in data, negative
            swa = 0

        auc, p_val, t0 = self.calc_uncertainty(
            include_self=include_self,
            N=N,
            pos_label=pos_label,
        )

        if np.isnan(auc) or p_val > 0.2:
            # if auc is np.NaN, means only one class represented in data
            t0 = 0.5

        if swa >= t0:
            # case: positive prediction
            pred = pos_label
        else:
            # case: negative prediction
            negative_labels = list(set(self.classes_) - set([pos_label]))
            # negative label must exist since there must be at least one negative data point;
            # if no negatives then swa=1>=t0 for all t0, and therefore wouldn't reach this condition
            assert (
                len(negative_labels) == 1
            ), "There must be at least one negative data point, or SWA/t0 may be incorrectly calculated"
            # negative label
            pred = negative_labels[0]

        return pred, swa, auc, p_val, t0


class GenRAPredBinary(GenRAPredClass, BinaryMixin):
    """Binary specific class for GenRAPredClass. Supports the calculation of similarity weighted activity (SWA) and the uncertainty of
    neighborhood prediction.
    """
