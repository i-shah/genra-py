import warnings
from collections.abc import Iterable
from itertools import zip_longest

import numpy as np
import pandas as pd
import sklearn.metrics as metrics

from genra.rax.skl.cls import GenRAPredClass, GenRAPredClassHybrid

def calc_uncertainty_from_swa(y, swa, N=100):
    """Uncertainty calculations, taken and slightly modified (removed pos_label) from previous genrapred module."""

    # get AUC
    fpr, tpr, t0 = metrics.roc_curve(
        y,
        swa,
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
        fpr, tpr, _ = metrics.roc_curve(y, swa)
        AUC.append(metrics.auc(fpr, tpr))
    p_val = 1.0 * np.sum(np.array(AUC) > auc) / N

    return auc, p_val, t0

class GenRAPredBinary(GenRAPredClass):
    """Binary specific class for GenRAPredClass. Supports the calculation of similarity weighted activity (SWA) and the 'uncertainty' of
    neighborhood.
    """

    def _get_y_binary(self, pos_label):
        """Get self._y vector represented in binary form, {0,1}. Raise warning in the case of sparse data (only one class represented) or
        and exception if pos_label not represented in this bindary data.

        Parameters
        ----------
        pos_label : obj (default=1)
            The encoding of positive observations in the training data

        """
        classes = self.classes_
        n_classes = len(classes)
        y = classes[self._y]
        if n_classes == 1:
            if pos_label in classes:
                warnings.warn(
                    f"The data is homogenous - only the positive class is represented: {pos_label}"
                )
            else:
                warnings.warn(
                    f"The data is homogenous - only the negative class is represented: {classes[0]}"
                )
        else:
            # case: n_classes == 2
            if pos_label not in classes:
                raise Exception(
                    f"Given pos_label ({pos_label} not represented by classes in data: {set(classes)}"
                )
        y = classes[self._y]
        y[y == pos_label] = 1
        y[y != 1] = 0
        return y.astype(int)

    def fit(self, X, Y):
        """Call the parent class's fit method, then validate that the data is truly binary.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data

        Y : {array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_outputs)
            Target values

        """
        res = super().fit(X, Y)
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

    def calc_neigh_swa(self, include_self, pos_label):
        """Returns size n 1d numpy array of each neighbor's Similarity Weighted Activity (SWA).
        By default, a chemical's data is not included in its calculated SWA.

        Parameters
        ----------
        include_self : bool (default=False)
            Whether to include self in calculation of Similarity Weighted Activity

        pos_label : obj (default=1)
            The encoding of positive observations in the training data
        """
        # get symmetric n-by-n matrix of pairwise similarities
        # TODO: what about if distance not in [0,1]?
        pairwise_sim = 1 - self.kneighbors_graph(self._fit_X, mode="distance").toarray()
        if not include_self:
            # set diagonal entries to 0 so self removed from calculation
            np.fill_diagonal(pairwise_sim, 0)
        sim_sum = pairwise_sim.sum()
        if sim_sum == 0:
            raise Exception("All pairwise similarities are 0; cannot calculate SWA")

        y = self._get_y_binary(pos_label)

        # matrix-vector multiplication then divide by sum of similarities for normalization
        return np.dot(pairwise_sim, y) / sim_sum

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
            # if both binary classes are not present in data, then uncertainty cannot be calculated
            return np.nan, np.nan, np.nan

        swa = self.calc_neigh_swa(include_self=include_self, pos_label=pos_label)
        y = self._get_y_binary(pos_label)
        return calc_uncertainty_from_swa(y, swa, N=N)


class GenRAPredBinaryHybrid(GenRAPredClassHybrid):
    """GenRA-py binary prediction class that supports uncertainty calculations (AUC, p_val, optimized threshold.)

    Hybrid SWA calculation relies on this following characteristic:
    weighted linear sum across fp_i of ( SWA of ( similarity_matrix_fp_i ) )
    is equivalent to
    SWA of ( weighted linear sum across fp_i of ( similarity_matrix_fp_i ) )

    This is because SWA is a linear operation, just like taking the weighted linear sum for hybridization, so
    they can be interchanged. I.e., an example of f(g(x)) = g(f(x)) because f and g are linear operators.
    """

    component_class = GenRAPredBinary

    def _get_y_binary(self, pos_label):
        """Get self._y vector represented in binary form, {0,1}. Raise warning in the case of sparse data (only one class represented) or
        and exception if pos_label not represented in this bindary data.

        Parameters
        ----------
        pos_label :
            Value that represents the positive observation in this binary data

        """
        classes = self.classes_
        n_classes = len(classes)
        y = classes[self._y]
        if n_classes == 1:
            if pos_label in classes:
                warnings.warn(
                    f"The data is homogenous - only the positive class is represented: {pos_label}"
                )
            else:
                warnings.warn(
                    f"The data is homogenous - only the negative class is represented: {classes[0]}"
                )
        else:
            # case: n_classes == 2
            if pos_label not in classes:
                raise Exception(
                    f"Given pos_label ({pos_label} not represented by classes in data: {set(classes)}"
                )
        y = classes[self._y]
        y[y == pos_label] = 1
        y[y != 1] = 0
        return y.astype(int)

    def calc_neigh_swa(self, include_self=False, pos_label=1, hybrid_weights="even"):
        """Returns size n 1d numpy array of each neighbor's Similarity Weighted Activity (SWA).
        By default, a chemical's data is not included in its calculated SWA.

        Parameters
        ----------
        include_self : bool (default=False)
            Whether to include self in calculation of Similarity Weighted Activity

        pos_label : obj (default=1)
            The encoding of positive observations in the training data

        hybrid_weights : list(float) | "even" (default="even")
            List of float representing the hybrid fingerprint weights

        """
        sum_weights = 0
        # initialize empty dataframe; this will eventually be weighted sum of SWA across hybrid components
        swa = pd.DataFrame()
        for idx, (model, weight) in enumerate(
            zip_longest(
                self._models, self.get_hybrid_weights(hybrid_weights), fillvalue=None
            )
        ):
            if model is None or weight is None:
                raise Exception(
                    "Models and weights must have the same number of components"
                )

            assert weight >= 0, "Weights must be non-negative"
            sum_weights += weight
            swa = swa.add(
                pd.DataFrame(
                    model.calc_neigh_swa(include_self=include_self, pos_label=pos_label)
                    * weight,
                    index=model._index,
                ),
                fill_value=0,
            )
        assert sum_weights > 0, "At least one non-zero weight must be provided"
        swa = swa.reindex(self._index)
        # last step, normalize
        return np.array(swa) / sum_weights

    def calc_uncertainty(
        self, include_self=False, N=100, pos_label=1, hybrid_weights="even"
    ):
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

        hybrid_weights : list(float) | "even" (default="even")
            List of float representing the hybrid fingerprint weights

        """
        if self.classes_.shape[0] < 2:
            # if both binary classes are not present in data, then uncertainty cannot be calculated
            return np.nan, np.nan, np.nan

        swa = self.calc_neigh_swa(
            include_self=include_self,
            pos_label=pos_label,
            hybrid_weights=hybrid_weights,
        )
        y = self._get_y_binary(pos_label=pos_label)
        return calc_uncertainty_from_swa(y, swa, N=N)
