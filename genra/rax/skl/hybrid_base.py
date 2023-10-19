"""Abstract class to implement hybridized GenRAPred classes; see GenRAPRedClassHybrid, GenRAPredValueHybrid."""

import abc
import numpy as np

class GenRAPredHybrid(abc.ABC):
    """Abstract class for hybridized GenRA prediction modeling."""

    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        """Parameters mirror those from __init__ methods in Scikit-learn's Kneighborsclassifier and Kneighborsregressor,
        which are also mirroed by GenRAPredClass and GenRAPredValue. All parameters are stored in an object attribute
        `component_params` so that they can be passed on to the constructor for each component models.
        """
        self.component_params = dict(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )

    def get_hybrid_weights(self, hybrid_weights):
        """Helper to get numpy array of 1's sized to number of hybrid components."""
        if hybrid_weights == "even":
            hybrid_weights = np.ones(self._n_models)
        return hybrid_weights

    @property
    @abc.abstractmethod
    def component_class(self):
        """The model class that should be used for each hybrid component. This is a class attribute. Designed to be one of
        GenRAPredClass, GenRAPredBinary, or GenRAPredValue, etc."""
        pass

    @abc.abstractmethod
    def fit(self, X, Y):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass