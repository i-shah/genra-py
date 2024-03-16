"""Tests for hybridized-genrapy."""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (accuracy_score, explained_variance_score,
                             f1_score, make_scorer, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from genra.rax.skl.binary import GenRAPredBinary
from genra.rax.skl.cls import GenRAPredClass
from genra.rax.skl.hybrid import (GenRAPredBinaryHybrid, GenRAPredClassHybrid,
                                  GenRAPredValueHybrid)
from genra.rax.skl.reg import GenRAPredValue

# Set up training data & models
# pasted from spreadsheet
data_str = """
1	0	0	0	1	1	0	1	1	0	0	1
1	1	0	1	0	0	0	1	0	1	1	0
0	0	0	1	1	1	0	0	0	0	1	0
0	0	1	1	1	0	1	1	1	0	1	1
1	0	1	0	1	1	1	0	0	1	1	1
1	1	0	0	0	0	1	0	0	1	0	1
"""
X = []
for line in data_str.split("\n"):
    line = line.strip()
    if len(line) > 1:
        X.append([int(num) for num in line.split()])
X_sample = np.array(X[1:])
# Y data
Y_binary = np.array([1, 0, 1, 0, 1])
Y_multiclass = np.array(["A", "B", "A", "C", "A"])
Y_continuous = np.array([70, -10, 45, 90, 33])

X_query = np.array(X[0:1])

hybrid_weights = [2, 1, 3]

slices = [
    slice(0, 5),
    slice(5, 8),
    slice(8, None),
]

hybrid_binary = GenRAPredBinaryHybrid(
    algorithm="brute",
    metric="jaccard",
    weights=lambda distances: 1 - distances,
    slices=slices,
    hybrid_weights=hybrid_weights,
)
scikit_binary = KNeighborsClassifier(
    algorithm="brute",
    metric="jaccard",
    weights=lambda distances: 1 - distances,
)
hybrid_multiclass = GenRAPredClassHybrid(
    algorithm="brute",
    metric="jaccard",
    weights=lambda distances: 1 - distances,
    slices=slices,
    hybrid_weights=hybrid_weights,
)
scikit_multiclass = KNeighborsClassifier(
    algorithm="brute",
    metric="jaccard",
    weights=lambda distances: 1 - distances,
)
hybrid_continuous = GenRAPredValueHybrid(
    algorithm="brute",
    metric="jaccard",
    weights=lambda distances: 1 - distances,
    slices=slices,
    hybrid_weights=hybrid_weights,
)
scikit_continuous = KNeighborsRegressor(
    algorithm="brute",
    metric="jaccard",
    weights=lambda distances: 1 - distances,
)
hybrid_binary.fit(X_sample, Y_binary)
scikit_binary.fit(X_sample, Y_binary)
hybrid_multiclass.fit(X_sample, Y_multiclass)
scikit_multiclass.fit(X_sample, Y_multiclass)
hybrid_continuous.fit(X_sample, Y_continuous)
scikit_continuous.fit(X_sample, Y_continuous)

# Set up testing data
hybrid_dist_query = 1 - np.array(
    [[0.16666667, 0.19444444, 0.47222222, 0.40277778, 0.27777778]]
)
hybrid_sorted_ind_query = np.argsort(hybrid_dist_query, axis=1)
hybrid_sorted_dist_query = np.take_along_axis(
    hybrid_dist_query, hybrid_sorted_ind_query, axis=1
)

hybrid_dist_sample = 1 - np.array(
    [
        [1, 0.333333333, 0.275, 0.4, 0.388888889],
        [0.333333333, 1, 0.388888889, 0.333333333, 0],
        [0.275, 0.388888889, 1, 0.472222222, 0.208333333],
        [0.4, 0.333333333, 0.472222222, 1, 0.5],
        [0.388888889, 0, 0.208333333, 0.5, 1],
    ]
)
hybrid_sorted_ind_sample = np.argsort(hybrid_dist_sample, axis=1)
hybrid_sorted_dist_sample = np.take_along_axis(
    hybrid_dist_sample, hybrid_sorted_ind_sample, axis=1
)

# Set up helpers
def is_equal(a, b):
    """If a, b are equal/close enough"""
    if "scipy.sparse" in str(type(a)):
        return is_equal(a.toarray(), b)
    elif "scipy.sparse" in str(type(b)):
        return is_equal(a, b.toarray())
    elif type(a) != type(b):
        return False
    elif isinstance(a, tuple) and isinstance(b, tuple):
        for _a, _b in zip(a, b, strict=True):
            if not is_equal(_a, _b):
                return False
        return True
    elif isinstance(a, np.ndarray):
        return np.array_equal(a, b) or np.allclose(a, b)
    elif isinstance(a, pd.DataFrame) or isinstance(a, pd.Series):
        return a.equals(b)
    else:
        return a == b


def is_similar(a, b):
    """If a, b have similar shape, to test output types"""
    if "scipy.sparse" in str(type(a)):
        return is_similar(a.toarray(), b)
    elif "scipy.sparse" in str(type(b)):
        return is_similar(a, b.toarray())
    if type(a) != type(b):
        return False
    elif isinstance(a, tuple) and isinstance(b, tuple):
        for _a, _b in zip(a, b, strict=True):
            if not is_similar(_a, _b):
                return False
        return True
    elif any([isinstance(a, _type) for _type in [np.ndarray, pd.DataFrame, pd.Series]]):
        return a.shape == b.shape
    else:
        return False


# Set up test sets & expected results
# See spreadsheet for calculation of expected results
test_sets = [
    {
        "name": "hybrid_binary",
        "model": hybrid_binary,
        "reference_model": scikit_binary,
        "tests": {
            "predict_proba": {
                "query": np.array([0.394495413, 0.605504587]),
                "sample": np.array(
                    [
                        [0.30590962, 0.69409038],
                        [0.64864865, 0.35135135],
                        [0.36729858, 0.63270142],
                        [0.49281314, 0.50718686],
                        [0.2384106, 0.7615894],
                    ]
                ),
            },
            "predict": {
                "query": np.array([1]),
                "sample": np.array([1, 0, 1, 1, 1]),
            },
        },
    },
    {
        "name": "hybrid_multiclass",
        "model": hybrid_multiclass,
        "reference_model": scikit_multiclass,
        "tests": {
            "predict_proba": {
                "query": np.array([0.605504587, 0.128440367, 0.266055046]),
                "sample": np.array(
                    [
                        [0.694090382, 0.139049826, 0.166859791],
                        [0.351351351, 0.486486486, 0.162162162],
                        [0.632701422, 0.165876777, 0.201421801],
                        [0.507186858, 0.123203285, 0.369609856],
                        [0.761589404, 0, 0.238410596],
                    ]
                ),
            },
            "predict": {
                "query": np.array(["A"]),
                "sample": np.array(["A", "B", "A", "A", "A"]),
            },
        },
    },
    {
        "name": "hybrid_continuous",
        "model": hybrid_continuous,
        "reference_model": scikit_continuous,
        "tests": {
            "predict": {
                "query": np.array([50.4587155963303]),
                "sample": np.array(
                    [53.34298957, 29.59459459, 46.80687204, 56.33470226, 54.64238411]
                ),
            },
        },
    },
]
neigh_swa = np.array(
    [0.475149105, 0.684210526, 0.359504132, 0.804560261, 0.544303797]
),

# additional tests to run for each test set
kng_tests = {
    "kneighbors": {
        "query": (hybrid_sorted_dist_query, hybrid_sorted_ind_query),
        "sample": (hybrid_sorted_dist_sample, hybrid_sorted_ind_sample),
        "kwargs": {"return_distance": True},
    },
    "kneighbors_graph": {
        "query": hybrid_dist_query,
        "sample": hybrid_dist_sample,
        "kwargs": {"mode": "distance"},
    },
}

@pytest.mark.parametrize("test_set", test_sets)
def test_hybrids(test_set):
    model, ref_model = test_set["model"], test_set["reference_model"]

    # specified tests + kneighbors tests
    for test, expected in list(test_set["tests"].items()) + list(kng_tests.items()):
        func_to_test, ref_func = getattr(model, test), getattr(ref_model, test)
        kwargs_query = expected.get("kwargs", {}).copy()
        kwargs_query["X"] = X_query
        kwargs_sample = expected.get("kwargs", {}).copy()
        kwargs_sample["X"] = X_sample
        res_query, ref_res_query = func_to_test(**kwargs_query), ref_func(
            **kwargs_query
        )
        res_sample, ref_res_sample = func_to_test(**kwargs_sample), ref_func(
            **kwargs_sample
        )

        fail_string = f"{test_set['name']} failed {test}"

        assert is_equal(
            res_query,
            expected["query"],
        ), f"{fail_string}, expected different values got={res_query} vs. expected={expected['query']}"
        assert is_equal(
            res_sample,
            expected["sample"],
        ), f"{fail_string}, expected different values got={res_sample} vs. expected={expected['sample']}"

        assert is_similar(
            res_query,
            ref_res_query,
        ), f"{fail_string}, expected different structures got={res_query} vs. expected={ref_res_query}"
        assert is_similar(
            res_sample,
            ref_res_sample,
        ), f"{fail_string}, expected different structures got={res_sample} vs. expected={ref_res_sample}"

    # model specific tests
    # check neigh SWA if binary
    if "hybrid_binary" == test_set["name"]:
        assert is_equal(
            model.calc_neigh_swa(include_self=False),
            neigh_swa
        )

    # test gridsearchcv
    if (test_name := test_set["name"]) in ["hybrid_binary", "hybrid_continuous"]:
        _y = Y_binary if test_name == "hybrid_binary" else Y_continuous
        score = model.score(X_sample, _y)
        print(score)
        os.environ["PYTHONWARNINGS"] = "ignore"
        grid_model = model.__class__(
            algorithm="brute",
        )
        params = [
            {
                "n_neighbors": range(1, 2),
                "hybrid_weights": [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)],
                "metric": ["jaccard", "cosine"],
                "slices": [
                    slices,
                ],
            },
            {
                "n_neighbors": [2],
                "hybrid_weights": [(1,)],
                "metric": ["jaccard"],
                "slices": [slice(0, None)],
            },
        ]
        scoring = f1_score if test_name == "hybrid_binary" else r2_score
        # check everything runs without error
        Grid = GridSearchCV(
            estimator=grid_model,
            param_grid=params,
            n_jobs=15,
            cv=2,
            verbose=2,
            scoring=make_scorer(scoring),
        )
        Best = Grid.fit(model._fit_X, model._y)
