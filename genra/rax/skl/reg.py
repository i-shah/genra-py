"""
GenRAPredValue 

Adapted from sklearn.neighbors.KNeighborsRegressor

"""

# Authors: Imran Shah (shah.imran@epa.gov)

import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsRegressor
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

class GenRAPredValue(KNeighborsRegressor):
    """GenRA Value Prediction based on k-nearest neighbors.

    The target is predicted by local interpolation of the targets
    associated of the nearest neighbors in the training set.

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

    p : integer, optional (default = 2)
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric : string or callable, default 'minkowski'
        the distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. See the documentation of the DistanceMetric class for a
        list of available metrics.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`Glossary <sparse graph>`,
        in which case only "nonzero" elements may be considered neighbors.

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        Doesn't affect :meth:`fit` method.


    """

    def __init__(self, n_neighbors=5, 
                 algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', 
                 metric_params=None, n_jobs=None,
                 **kwargs):
        super().__init__(
              n_neighbors=n_neighbors,
              algorithm=algorithm,
              leaf_size=leaf_size, metric=metric, p=p,
              metric_params=metric_params, n_jobs=n_jobs, 
              **kwargs)
        self.weights = _check_weights('uniform')

    @property
    def _pairwise(self):
        # For cross-validation routines to split data correctly
        return self.metric == 'precomputed'

    def kneighbors_sim(self,X):
        """
        Find the k-nearest neighbours for each instance and similarity scores. 
        All distances (D) are converted to similarity (S) by:
        
                 D - D.min()
        Sim =   --------------
                D.max()-D.min()
        We assume D.min()==0

        """
        neigh_dist, neigh_ind = self.kneighbors(X)
        
        # Convert distances to similarities:
        if self.metric == 'jaccard':
            neigh_sim = 1-neigh_dist
        else:
            neigh_dist_n = neigh_dist / neigh_dist.max()
            neigh_sim = 1 - neigh_dist_n                       
        
        return neigh_sim, neigh_ind
    
    def predict(self, X):
        """Predict the target for the provided data

        Parameters
        ----------
        X : array-like, shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : array of int, shape = [n_queries] or [n_queries, n_outputs]
            Target values
        """
        X = check_array(X, accept_sparse='csr')

        neigh_sim, neigh_ind = self.kneighbors_sim(X)
        
        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        y_pred = np.empty((X.shape[0], _y.shape[1]), dtype=np.float64)

        for j in range(_y.shape[1]):
            denom=np.sum(neigh_sim[j])
            num = np.sum(_y[neigh_ind, j] * neigh_sim[j], axis=1)
            y_pred[:, j] = num / denom

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred

class GenRAPredValueHybrid(GenRAPredHybrid):
    """GenRA-py continuous prediction class that supports hybridized calculations.

    This class inherits from GenRAPredValue so that it is able to use self.get_params() method to
    instantiate the component model objects for each fingerprint. Therefore, caution must be used, as
    some of the inherited methods will not work.
    """

    # The component class to use for each fingerprint component
    component_class = GenRAPredValue

    def fit(self, X, Y):
        """Same as sklearn.KneighborsRegressors.fit() method, but modified for the hybrid version.
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
        # X and Y should be iterables
        assert isinstance(X, Iterable), "X must be an iterable"
        assert isinstance(Y, Iterable), "Y must be an iterable"
        models = []
        for idx, (X_component, Y_component) in enumerate(
            zip_longest(X, Y, fillvalue=None)
        ):
            if X_component is None or Y_component is None:
                raise Exception("X and Y must have the same number of components")
            n = X_component.shape[0]
            assert (
                n == Y_component.shape[0]
            ), f"X_component and Y_component must have the same number of rows at index={idx}"
            model = self.component_class(**self.component_params)
            # override n_neighbors in case Y_component smaller than the parameter set
            model.set_params(n_neighbors=min(n, model.n_neighbors))
            model.fit(X_component, Y_component)
            models.append(model)

        # should be immutable
        self._models = tuple(models)
        self._n_models = len(models)

        # see https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.fit
        return self

    def predict(self, X, hybrid_weights="even"):
        """
        Same as GenRAPredValue.predict() method, but modified for the hybrid version.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples

        hybrid_weights : list(float) | "even" (default="even")
            List of float representing the hybrid fingerprint weights
        """

        # initialize empty dataframe; this will eventually be sum of weighted values across hybrid components
        pred = pd.DataFrame()
        sum_weights = 0
        for model, X_component, weight in zip_longest(
            self._models, X, self.get_hybrid_weights(hybrid_weights), fillvalue=None
        ):
            if model is None or X_component is None or weight is None:
                raise Exception(
                    "The number of components in fitted models, X, and hybrid_weights must be the same"
                )
            assert weight >= 0, "Weights must be non-negative"
            pred = pred.add(
                pd.DataFrame(
                    model.predict(X_component) * weight,
                ),
                fill_value=0,
            )
            sum_weights += weight
        assert sum_weights > 0, "At least one non-zero weight must be provided"
        return np.array(pred) / sum_weights
