# Utility Functions

Based on: [Utilities for developer](https://scikit-learn.org/stable/developers/utilities.html#developers-utils)

### Assert

- `validation.assert_all_finite`: Throw an error if array contains NaNs or Infs.
- `_testing.assert_allclose`: quasi equality of arrays, using a `tol` parameter

### Formater / converter

- `validation.as_float_array`: convert input to array of float
- `validation.check_array`:
    - check that input is a 2D array. Sparse matrix and other dimensions can be optionally allowed
    - call `assert_all_finite`
- `validation.check_X_y`:
    - check that X and y have consistent lengths
    - call `check_array` on X, `column_or_1d` on y (use `multi_output=True` for multilabel y)
- `validation.indexable`: check that input arrays have consistent length and can be sliced or indexed (useful for cross-validation)
- **do not** use `np.asanyarray` or `np.atleast_2d` since it lets `np.matrix` through (different API from `np.array`)

### Random state

- `validation.check_random_state`: randomness must be handle with `np.random.RandomState` only

### Estimators

- `validation.check_is_fitted`: check that estimator has been fitted before calling predict or transform method
- `validation.has_fit_parameter`: check that the fit method has a given parameter
- `all_estimators`: return a list of all scikit-learn estimators
- `multiclass.type_of_target`: return the type of the target y among continuous, continuous-multioutput, binary, multiclass, multiclass-multioutput, multilabel-indicator or unknown
- `multiclass.unique_labels`: Extract an ordered array of unique labels, "multiclass-multioutput" input type not allowed.

### Warnings & Exceptions

- `utils.deprecated`: decorator to mark a function or class as deprecated
- `exceptions.ConvergenceWarning`: Custom warning to catch convergence problems (used in Lasso)