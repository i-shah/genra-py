"""
GenRA Classifier

Adapted from sklearn.neighbors.KNeighborsClassifier

"""

# Authors: Imran Shah (shah.imran@epa.gov)

import numpy as np
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.neighbors._base import BaseEstimator, NeighborsBase,\
        KNeighborsMixin, SupervisedIntegerMixin,\
        _check_weights, _get_weights
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted,_is_arraylike, _num_samples
from sklearn.utils.multiclass import unique_labels
from scipy import stats
from sklearn.utils.extmath import weighted_mode

import warnings

class GenRAPredClass(NeighborsBase, KNeighborsMixin,
                     SupervisedIntegerMixin, ClassifierMixin):
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

