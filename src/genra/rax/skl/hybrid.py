"""
Module to support hybridized genrapy estimators.
GenRAPredHybrid is the abstract class that implements the core hybridized approach.

Approach
========
In the hybridized case, the features for each component are concatenated to create one concatenated training data (X). 
In the constructor, a list of slice objects - the `slices` parameter - is passed along so the model can identify the respective component's column slice for the training data.  

Example:
Training data with 3 components, n_samples=6

X:
   |   component0    | | component1  | |component2| 
    1	0	0	0	1	1	0	1	1	0	0	1
    1	1	0	1	0	0	0	1	0	1	1	0
    0	0	0	1	1	1	0	0	0	0	1	0
    0	0	1	1	1	0	1	1	1	0	1	1
    1	0	1	0	1	1	1	0	0	1	1	1
    1	1	0	0	0	0	1	0	0	1	0	1

slices = [
    slice(0,5),     # component0: first 5 columns
    slice(5,8),     # component1: 6th through 8th columns
    slice(8,None),  # component2: 8th through rest (10th)
]

See tests/hybrid_test.py as well as genrapy_hybrid_test_calculation_steps.xlsx to see further details.

"""

import abc
from itertools import zip_longest

import numpy as np
import scipy
from sklearn.neighbors._base import KNeighborsMixin, NeighborsBase

from genra.rax.skl.binary import BinaryMixin, GenRAPredBinary
from genra.rax.skl.cls import GenRAPredClass
from genra.rax.skl.reg import GenRAPredValue


class GenRAPredHybrid(abc.ABC, KNeighborsMixin, NeighborsBase):
    """Abstract class for hybridized estimators of GenRAPred."""

    def __init__(
        self,
        n_neighbors=5,
        *,
        slices=[slice(0, None)],
        hybrid_weights="uniform",
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        """Parameters mirror those from __init__ methods in Scikit-learn's Kneighborsclassifier and Kneighborsregressor,
        which are also mirroed by GenRAPredClass and GenRAPredValue. Two additional parameters are added

        Parameters
        ----------
        slices : list(slice), default=slice(0, None, None)
            Slices that represent the columns of each component. If not provided one component with all columns.

        hybrid_weights : list(float), default="even"
            Floats that represent the hybrid component weights. If not provided all hybrid components have uniform weights.
        """
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
            weights=weights,
        )
        self.slices = slices
        self.hybrid_weights = hybrid_weights

    @property
    @abc.abstractmethod
    def component_class(self):
        """The model class that should be used for each hybrid component. This is a class attribute. Designed to be one of
        GenRAPredClass, GenRAPredBinary, or GenRAPredValue, etc."""
        pass

    def _check_and_get_hybrid_weights(self):
        """Helper to get a numpy array of weights and perform validation.

        Returns
        -------
        hybrid_weights : ndarray of shape (n_components,)
            Weights for components
        """
        # get weights
        if isinstance(self.hybrid_weights, str) and self.hybrid_weights == "uniform":
            hybrid_weights = np.ones(len(self.slices))
        else:
            hybrid_weights = np.array(self.hybrid_weights)

        # valiedate weights
        assert hybrid_weights.shape == (
            len(self.slices),
        ), f"Number of components don't match up across self.slices ({len(self.slices)}) and hybrid_weights ({len(hybrid_weights)})"
        assert (
            np.sum(hybrid_weights) > 0
        ), "At least one non-zero hybrid weight must be provided"

        return hybrid_weights

    def fit_extras(self):
        """Helper to fit(...) method. Used for setting model specific attributes - e.g., self.ckasses_ for GenRAPredClassHybrid"""
        pass

    def fit(self, X, y):
        """Same as scikitlearn estimator's fit(...) method, but modified for the hybrid version.

        It creates a model for each component
        """
        # TODO: is it worth calling super.fit(...)? Considerations: would take care of setting some attributes.
        # super.fit(X, y)

        # TODO: check array
        X = np.array(X)

        # this will be passed on to each component model's constructor
        params = self.get_params()
        # remove hybrid params
        params.pop("slices")
        params.pop("hybrid_weights")

        models = []
        for _slice in self.slices:
            # validate slice
            error_str = f"{_slice} is not valid for X with shape {X.shape}"
            assert _slice.start < X.shape[1], error_str
            if _slice.stop is not None:
                assert _slice.stop < X.shape[1], error_str
            # create component model, and fit
            model = self.component_class(**params)
            model.fit(X[:, _slice], y)
            models.append(model)
            # not necessary but could be useful
            model._slice = _slice

        # Set object attributes. Some of these attributes are needed by other methods already defined
        #   in this abstract class - e.g., see self._get_distances(...) that uses self.n_samples_fit_.
        #   Some may be useful - e.g., self._y would likely be used by self.predict(...) if implemented.
        #   Some are just to stay consistent with certain scikit estimators (primarily KNeighborsClassifier
        #   and KNeighborsRegressor) - e.g., self.n_features_in_. Note that not all of those attributes are
        #   implemented, read more above.

        # scikit attributes
        self._y = models[0]._y.copy()
        self._fit_X = np.array(X)
        self.n_features_in_ = X.shape[1]
        self.n_samples_fit_ = X.shape[0]
        # NOTE: feature_names_in_ is not supported right now

        # hybrid attributes
        self._models = tuple(models)

        # model specific adjustments
        self.fit_extras()

        # see https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.fit
        return self

    def _get_distances(
        self,
        X=None,
    ):
        """Helper that returns hybrid distances. This is the weighted sum of distances (normalized) across each component,
        computed using the component model's kneighbors_graph(...) method. This is the core method that other methods in this mixin rely on.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_queries, n_features), or (n_queries, n_indexed) if metric == ‘precomputed’, default=None
            The query point or points. If not provided, neighbors of each indexed point are returned.

        Returns
        -------
        A : ndarray of shape (n_queries, n_neighbors)
            `A[i, j]` gives the hybrid distance from `i` to `j`.
        """
        # this will eventually be weighted (and normalized) sum of distances across hybrid components
        # labeled A to keep consistent with scikit
        A = None
        sum_weights = 0
        # currently only implemented for jaccard and cosine
        for idx, model, _slice, hybrid_weight in zip_longest(
            range(len(self._models)),
            self._models,
            self.slices,
            self.hybrid_weights,
            fillvalue=None,
        ):
            if model is None or slice is None or hybrid_weight is None:
                raise Exception(
                    f"The number of components do not match up, at index={idx}"
                )
            assert hybrid_weight >= 0, "Weights must be non-negative"
            sum_weights += hybrid_weight

            if X is None:
                X_component = model._fit_X
            else:
                X_component = np.array(X)[:, _slice]

            A_component = (
                model.kneighbors_graph(
                    X=X_component, n_neighbors=self.n_samples_fit_, mode="distance"
                ).toarray()
                * hybrid_weight
            )

            if A is None:
                A = A_component
            else:
                assert A.shape == A_component.shape
                A = np.add(A, A_component)

        assert not np.isnan(np.sum(A)), "Unexpected NaN(s) in A"
        # normalize distances
        A = A / sum_weights

        return A

    def _get_and_check_n_neighbors(self, query_is_train, n_neighbors=None):
        """Helper that validates and returns n_neighbors to keep consistent with behaviors of scikit
        estimators that utilizes NeighborsBase class. Used to for DRY-ness."""
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        n = n_neighbors
        if query_is_train:
            # see https://github.com/scikit-learn/scikit-learn/blob/093e0cf14/sklearn/neighbors/_base.py#L795
            n += 1
        n_samples_fit = len(self._y)
        if n > n_samples_fit:
            # this validation taken from scikit
            raise ValueError(
                "Expected n_neighbors <= n_samples, "
                " but n_samples = %d, n_neighbors = %d (Note that query_is_train=%s)"
                % (n_samples_fit, n, query_is_train)
            )
        return n_neighbors

    def kneighbors_graph(
        self,
        X=None,
        n_neighbors=None,
        mode="connectivity",
    ):
        """Hybrid implementation of scikit's NeighborsBase.kneighbors_graph(...) method."""
        # from https://github.com/scikit-learn/scikit-learn/blob/3f89022fa04d293152f1d32fbc2a5bdaaf2df364/sklearn/neighbors/_base.py#L794
        query_is_train = X is None

        A = self._get_distances(X=X)

        if query_is_train:
            assert (
                A.shape[0] == A.shape[1]
            ), "expecting a symmetric matrix since query_is_train=True"
            # If query_is_train=True, query point itself should not be included in neighborhood. Set to np.inf to ensure
            # this is true for sorting below.
            np.fill_diagonal(A, np.inf)

        n_neighbors = self._get_and_check_n_neighbors(
            query_is_train=query_is_train, n_neighbors=n_neighbors
        )

        # Array of indices, sorted for only the top n_neighbors with the rest unsorted to avoid computation time
        part_ind = np.argpartition(A, range(n_neighbors), axis=1)
        # indices of points that didn't make the neighborhood
        out_ind = part_ind[:, n_neighbors:]
        # the 0 here represents "not in neighborhood", as opposed to "distance=0".
        np.put_along_axis(arr=A, indices=out_ind, values=0, axis=1)

        if mode == "connectivity":
            # sorted indices of neighborhood
            in_ind = part_ind[:, :n_neighbors]
            np.put_along_axis(arr=A, indices=in_ind, values=1, axis=1)

        if query_is_train:
            # Undoing what's done above. Set to 0 because query point itself is not included in neighborhood when X=None,
            # to keep consistent with scikit behavior. Possibly misleading because 0 here represents "not in neighborhood",
            # as opposed to "distance=0", which in fact it also is.
            np.fill_diagonal(A, 0)

        # to keep consistent with scikit
        return scipy.sparse.csr_matrix(A)

    def kneighbors(
        self,
        X=None,
        n_neighbors=None,
        return_distance=True,
    ):
        """Hybrid implementation of scikit's NeighborsBase.kneighbors(...) method."""

        query_is_train = X is None

        A = self._get_distances(X=X)

        if query_is_train:
            assert (
                A.shape[0] == A.shape[1]
            ), "expecting a symmetric matrix since query_is_train=True"
            # If query_is_train=True, query point itself should not be included in neighborhood. Set to np.inf to ensure
            # this is true for sorting below. Since _get_and_check_n_neighbors(...) below checks that
            # n_neighbors < n_samples, these np.inf values won't be returned even if return_distance=True below.
            np.fill_diagonal(A, np.inf)

        n_neighbors = self._get_and_check_n_neighbors(
            query_is_train=query_is_train, n_neighbors=n_neighbors
        )

        # sorted indices of neighborhood, use argpartition to avoid sorting everything
        neigh_ind = np.argpartition(A, range(n_neighbors), axis=1)[:, :n_neighbors]

        if return_distance:
            neigh_dist = np.take_along_axis(A, neigh_ind, axis=1)
            return neigh_dist, neigh_ind
        else:
            return neigh_ind


class GenRAPredClassHybrid(GenRAPredHybrid, GenRAPredClass):
    """GenRA-py classification prediction class that supports hybridized calculations. Designed to support multi-class
    values.
    """

    component_class = GenRAPredClass

    def fit_extras(self):
        """Set additional attributes related to multiclass data"""
        # more attributes from scikit KNeighborsClassifier
        self.classes_ = self._models[0].classes_
        # this is specifically needed in GenRAPredClass.predict(...)
        self.outputs_2d_ = self._models[0].outputs_2d_


class GenRAPredBinaryHybrid(GenRAPredClassHybrid, BinaryMixin):
    """Binary specific class for GenRAPredClassHybrid. Supports the calculation of hybrid similarity weighted activity (SWA) and the uncertainty of
    neighborhood prediction.
    """

    component_class = GenRAPredBinary


class GenRAPredValueHybrid(GenRAPredHybrid, GenRAPredValue):
    """GenRA-py continuous prediction class that supports hybridized calculations."""

    component_class = GenRAPredValue
