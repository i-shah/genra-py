# Release Notes: Version 1.0.1

This update includes the introduction of several new features, as well as some minor bug fixes. A tutorial for the use of the new GenRAPredHybrid prediction engine is available in [notebooks](notebooks/misc/Tutorial_genra-py.ipynb).

## New Features

### GenRAPredHybrid
This newly introduced prediction class allows users to make predictions using hybrid fingerprints with custom weights. Compatible with value, binary, and class predictions as GenRAPredValueHybrid, GenRAPredBinaryHybrid, and GenRAPredClassHybrid respectively, this estimator class requires new hyperparameters **slices** and **hybrid_weights**:

- **slices**: list(slice), default=slice(0, None, None) \
            Slices that represent the columns of each component. If not provided, GenRA will use one component with all columns. Slices are matched by column index to the X input.
- **hybrid_weights**: hybrid_weights : list(float), default="even" \
            Floats that represent the hybrid component weights. If not provided, all hybrid components have uniform weights.

### Universal Distance
The original version of genra-py focused heavily on the use of the Jaccard metric for identifying nearest neighbors. As more fingeprints and metrics have become relevant to read-across studies with GenRA, the developers have worked to find the most appropriate way to convert calculated distances into similarity scores for each fingerprint and metric combination. The result is the introduction of the **universal_distance** hyperparameter to GenRAPRedValue:

- **universal_distance**: boolean, default = True \
            The boolean value of this hyperparameter indicates whether distances should be scaled by the maximum possible distance determined by the metric and fingerprint combination used or on a chemical-by-chemical basis. 

Use of the universal_distance setting should be based on user knowledge of the metric and fingerprints used. For example, the Jaccard metric used with binary fingerprints has a maximum distance of 1, making it compatible with universal_distance = True. However, the Euclidean metric used with continuously valued fingerprints does not have a maximum distance, and is more appropriately used with universal_distance = False. 

The underlying calculation for universal_distance computes the distance between two fingerprints matching the shape of the fingerprints used in the X input, where one fingerprint is composed entirely of 0s and the other is composed entirely of 1s. This is then used as the factor for scaling raw distances to relative distances for conversion to similarity scores. 

**Note**: The universal_distance function has been implemented with the intention of maintaining consistency with past calculations for as many users as possible. Predictions made previously using the Jaccard metric will remain the same without any manual parameter changes. Predictions made using other metrics will be altered if rerun after this version release, but the original calculation can be obtained again by setting universal_distance = False in the estimator declaration. 

### Blank Predictions for Performance Testing
As research explores the predictive power of GenRA, it becomes increasingly useful to test GenRA predictions on known endpoints. Consistent with scikit-learn's self.kneighors() functionality from the KNeighborsRegressor, it is now possible to run self.predict() without a new X input. After defining and fitting an estimator (self) to a data set, running **self.predict()** will make predictions for every chemical used in the fit set without including any of the chemicals in their own neighborhood. The post-fit attribute functions for the estimator including self.kneighbors, self.kneighbors_sim, and self.maxDistance are also compatible with this functionality. Empty parentheses on the self.predict function can be used with any of GenRAPredValue, GenRAPredBinary, and GenRAPredClass. 

## Bug Fixes
- **GenRAPredValue.kneighbors_sim** Corrected scaling methods applied to turn raw distances into relative distances for conversion to similarity scores, which previously could result in a division by zero. Implementation of bug fix includes introduction of the universal_distance feature to achieve consistency between treatment of Jaccard and other metrics. 

- **GenRAPredValue.predict** Corrected scaling methods applied to turn similarity scores into contribution weights for final calculation, which previously could result in a division by zero. 
