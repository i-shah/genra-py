"""
GenRA Classifier

Adapted from sklearn.neighbors.KNeighborsClassifier

"""

# Authors: Imran Shah (shah.imran@epa.gov)

import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors._base import _check_weights, _get_weights
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted,_is_arraylike, _num_samples
from sklearn.utils.multiclass import unique_labels
from scipy import stats
from sklearn.utils.extmath import weighted_mode

from genra.rax.skl.hybrid_base import GenRAPredHybrid
from collections.abc import Iterable
from itertools import zip_longest
import pandas as pd

import warnings

class GenRAPredClass(KNeighborsClassifier):
    """GenRA Classifier implementing the k-nearest neighbors vote.

    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : string or callable, default '1-jaccard'
        the distance metric to use for the tree.  The default metric is
        1-jaccard, and with p=2 is equivalent to the standard Euclidean
        metric. See the documentation of the DistanceMetric class for a
        list of available metrics.

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    n_jobs : int
    or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        Doesn't affect :meth:`fit` method.

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels known to the classifier

    effective_metric_ : string or callble
        The distance metric used. It will be same as the `metric` parameter
        or a synonym of it, e.g. 'euclidean' if the `metric` parameter set to
        'minkowski' and `p` parameter set to 2.

    effective_metric_params_ : dict
        Additional keyword arguments for the metric function. For most metrics
        will be same with `metric_params` parameter, but may also contain the
        `p` parameter value if the `effective_metric_` attribute is set to
        'minkowski'.

    outputs_2d_ : bool
        False when `y`'s shape is (n_samples, ) or (n_samples, 1) during fit
        otherwise True.

    """

    def __init__(self, n_neighbors=5,
                 weights='uniform', algorithm='auto', leaf_size=30,
                 p=2, metric='jaccard', metric_params=None, n_jobs=None,
                 **kwargs):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size, metric=metric, p=p,
            metric_params=metric_params,
            n_jobs=n_jobs, **kwargs)
        self.weights = _check_weights(weights)

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like, shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : array of shape [n_queries] or [n_queries, n_outputs]
            Class labels for each data sample.
        """
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_outputs = len(classes_)
        n_queries = _num_samples(X)
        weights = _get_weights(neigh_dist, self.weights)

        y_pred = np.empty((n_queries, n_outputs), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            if weights is None:
                mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
            else:
                mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)

            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()

        return y_pred
        
    
    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like, shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        p : array of shape = [n_queries, n_classes], or a list of n_outputs
            of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_queries = _num_samples(X)

        weights = _get_weights(neigh_dist, self.weights)
        if weights is None:
            weights = np.ones_like(neigh_ind)

        all_rows = np.arange(X.shape[0])
        probabilities = []
        for k, classes_k in enumerate(classes_):
            pred_labels = _y[:, k][neigh_ind]
            proba_k = np.zeros((n_queries, classes_k.size))

            # a simple ':' index doesn't work right
            for i, idx in enumerate(pred_labels.T):  # loop is O(n_neighbors)
                proba_k[all_rows, idx] += weights[:, i]

            # normalize 'votes' into real [0,1] probabilities
            normalizer = proba_k.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba_k /= normalizer

            probabilities.append(proba_k)

        if not self.outputs_2d_:
            probabilities = probabilities[0]

        return probabilities


class GenRAPredClassHybrid(GenRAPredHybrid):
    """GenRA-py classification prediction class that supports hybridized calculations. Designed to support multi-class
    values.
    """

    # The component class to use for each hybrid component;
    component_class = GenRAPredClass

    def fit(self, X, Y):
        """Same as sklearn.KneighborsClassifier.fit() method, but modified for the hybrid version.

        Note:
        - neighbors without data in some components are ignored in the final calculations

        Parameters
        ----------
        X : Iterable(DataFrame)
            Iterable of Pandas DataFrame representing each component training X data.
            Must have matching (row) index with complementary Y parameter.

        Y : Iterable(DataFrame)
            Iterable of Pandas DataFrame representing each component training Y (target) data.
            Must have matching (row) index with complementary X parameter.

        """

        # X and Y must be iterables
        assert isinstance(X, Iterable), "X must be an iterable"
        assert isinstance(Y, Iterable), "Y must be an iterable"
        models = []
        # initialize empty dataframe; this will eventually be union of y data across hybrid components
        y = pd.DataFrame()
        for idx, (X_component, Y_component) in enumerate(
            zip_longest(X, Y, fillvalue=None)
        ):
            if X_component is None or Y_component is None:
                raise Exception("X and Y must have the same number of components")
            n = X_component.shape[0]
            list(X_component.index) == list(
                Y_component.index
            ), f"X_component and Y_component must have the same indices at component {idx}"
            model = self.component_class(**self.component_params)
            # override n_neighbors in case Y_component smaller than the parameter set
            model.set_params(n_neighbors=min(n, model.n_neighbors))
            model.fit(X_component, Y_component)
            # set component model index
            model._index = Y_component.index
            models.append(model)
            # takes the union
            y = y.combine_first(Y_component)

        # attributes used in other class methods
        self._models = tuple(models)
        self._n_models = len(models)
        self._index = y.index
        # raw values of the data
        # e.g., array(['A', 'B', 'A', 'C', 'A'], dtype=object)
        y_vals = y.to_numpy().flatten()
        # classes in the data; part of the upstream sklearn classifier's object attribute
        # e.g., array(['A', 'B', 'C'], dtype=object)
        self.classes_ = np.unique(y_vals)
        # dictionary that maps class values to index; needed to do lookups
        # e.g., {'A': 0, 'B': 1, 'C': 2}
        self.classes_to_y_dict = {k: v for v, k in enumerate(self.classes_)}
        # Indices of classes of values in the data; part of the upstream sklearn classifier's object attribute
        # e.g., array([0, 1, 0, 2, 0])
        self._y = np.vectorize(self.classes_to_y_dict.get)(y_vals)

        # see https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.fit
        return self

    def predict_proba(self, X, hybrid_weights="even"):
        """Same as GenRAPredClass.pred_proba() method, but modified for the hybrid version.

        Note:
        neighbors without data in some components are ignored in the final calculations

        Parameters
        ----------
        X : Iterable(DataFrame)
            Iterable of Pandas DataFrame representing each component testing X data

        hybrid_weights : list(float) | "even" (default="even")
            List of float representing the hybrid fingerprint weights

        """
        # initialize empty dataframe; this will eventually be sum of probabilities across hybrid components
        proba = pd.DataFrame()
        sum_weights = 0
        for model, X_component, weight in zip_longest(
            self._models, X, self.get_hybrid_weights(hybrid_weights), fillvalue=None
        ):
            if model is None or X_component is None or weight is None:
                raise Exception(
                    "The number of components in fitted models, X, and hybrid_weights must be the same"
                )
            assert weight >= 0, "Weights must be non-negative"
            sum_weights += weight
            curr_proba = pd.DataFrame(
                model.predict_proba(X_component) * weight, columns=model.classes_
            )
            if not proba.empty:
                assert (
                    curr_proba.shape[0] == proba.shape[0]
                ), "The number of test samples (n_queries) must match across components"
            proba = proba.add(curr_proba, fill_value=0)

        assert sum_weights > 0, "At least one non-zero weight must be provided"
        return np.array(proba) / sum_weights

    def predict(self, X, hybrid_weights="even"):
        """Same as GenRAPredClass.predict() method, but modified for the hybrid version.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples

        hybrid_weights : list(float) | "even" (default="even")
            List of float representing the hybrid fingerprint weights

        """

        weighted_probas = self.predict_proba(X, hybrid_weights=hybrid_weights)
        class_idx = np.argmax(weighted_probas, axis=1)
        return self.classes_[class_idx]
