"""
GenRAPredValue 

Adapted from sklearn.neighbors.KNeighborsRegressor

"""

# Authors: Imran Shah (shah.imran@epa.gov)

import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors._base import _get_weights
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted,_is_arraylike, _num_samples
from sklearn.utils.multiclass import unique_labels
from scipy import stats
from sklearn.utils.extmath import weighted_mode
from sklearn.metrics import pairwise_distances

import warnings

class GenRAPredValue(KNeighborsRegressor):
    """GenRA Value Prediction based on k-nearest neighbors.

    The target is predicted by local interpolation of the targets
    associated of the nearest neighbors in the training set.

    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    weights: {'uniform', 'distance'} or callable, default = 'uniform'

    ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.

    ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.

    [callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.


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

    p : float, optional (default = 2)
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

    universal_distance: boolean, default 'True'
        determines whether a maximum distance should be used with the given metric.
        For example, the Jaccard metric takes values between 0 and 1, so keeping this
        feature on with the value True will scale all distances by a factor of 1 then subtracted
        from 1 to obtain similarity scores. The Canberra metric takes values between 0 and 
        the number of features used, so if you have 2048 fingerprint bits all distances will 
        be scaled by 2048 before subtraction from 1. Setting this value to 'False' instead scales
        distances by the distance of each chemical in the test set from its nth neighbor. 

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
                 metric_params=None, n_jobs=None, universal_distance = True,
                 **kwargs):
        super().__init__(
              n_neighbors=n_neighbors,
              algorithm=algorithm,
              leaf_size=leaf_size, metric=metric, p=p,
              metric_params=metric_params, n_jobs=n_jobs,
              **kwargs)
        self.universal_distance = universal_distance
    

    @property
    def _pairwise(self):
        # For cross-validation routines to split data correctly
        return self.metric == 'precomputed'
    
    def maxDistance(self, X = None):
        """
        Compute the maximum distance between two chemicals (with binary print values, also works for
        Canberra metric with continuous prints)
        
        Helps to identify test chemicals lacking source analogues.
        """
        query_is_train  = X is None
        if query_is_train:
            X = self._fit_X
        dims = X.shape[1]
        empty = np.zeros((1, dims), dtype=np.float64)
        full = np.empty((1,dims), dtype=np.float64)
        full.fill(1)
        max_distance = pairwise_distances(empty, full, metric = self.metric)
        
        return max_distance

    def kneighbors_sim(self,X = None):
        """
        Find the k-nearest neighbours for each instance and similarity scores. 
        All distances (D) are converted to similarity (S) by:
        
                 D - D.min()
        Sim =   --------------
                D.max()-D.min()
        We assume D.min()==0

        """
        neigh_dist, neigh_ind = self.kneighbors(X)

        # Check for chems with no similarities (only works with universal_distance on)
        if self.universal_distance:
            lost_chems = np.all(neigh_dist == self.maxDistance(X), axis = 1)
            lost_chem_indices = []
            counter = 0
            for boolean in lost_chems:
                if boolean:
                    lost_chem_indices.append(counter)
                counter += 1
            if len(lost_chem_indices) > 0:
                print(f"According to this metric, the training data may not contain source analogues for the chemical(s) \n with the following row indices within the testing set: {lost_chem_indices} ")
        
        
        # Convert distances to similarities:
        #This will use the univeral_distance feature, as old Jaccard uses did
        if self.universal_distance:
            division_scalar = self.maxDistance(X)
            neigh_dist_n = neigh_dist/division_scalar
            neigh_sim = 1-neigh_dist_n
        #This will use the version previously matched to non-Jaccard metrics, somputing row-by-row
        #maximums
        else:
            division_array = neigh_dist.max(1).copy()
            division_array[division_array == 0] = 1
            neigh_dist_n = neigh_dist / division_array[:,None]
            neigh_sim = 1 - neigh_dist_n                       
        
        return neigh_sim, neigh_ind
    
    def predict(self, X = None):
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
        query_is_train = X is None
        if query_is_train:
            X = self._fit_X
        
        X = check_array(X, accept_sparse='csr')

        if query_is_train:
            neigh_sim, neigh_ind = self.kneighbors_sim()
        else:
            neigh_sim, neigh_ind = self.kneighbors_sim(X)
        
        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        y_pred = np.empty((X.shape[0], _y.shape[1]), dtype=np.float64)

        denom=np.sum(neigh_sim, axis=1)
        for j in range(_y.shape[1]):
            num = np.sum(_y[neigh_ind, j] * neigh_sim, axis=1)
            if 0 in denom:
                bool_list = list(denom == 0)
                denom[bool_list] = neigh_sim.shape[1]
                num[bool_list] = np.sum(_y[neigh_ind,j][bool_list], axis = 1)
            y_pred[:, j] = num / denom

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred

