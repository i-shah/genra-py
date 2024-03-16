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

    def __init__(self, n_neighbors=5, sim_params=dict(),
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

        denom=np.sum(neigh_sim, axis=1)
        for j in range(_y.shape[1]):
            num = np.sum(_y[neigh_ind, j] * neigh_sim, axis=1)
            y_pred[:, j] = num / denom

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred

