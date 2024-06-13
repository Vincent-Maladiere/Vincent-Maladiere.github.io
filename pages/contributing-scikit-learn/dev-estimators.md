# Developing estimators

Based on: [Developing scikit-learn estimators](https://scikit-learn.org/stable/developers/develop.html)

Or “How to safely interact with Pipelines and model selection”

## Estimator API

### Instantiation

- `__init__` accepts model constraints, but must not accept training data (reserved for fitting)
    
    ✅ Do:
    
    ```python
    estimator = SVC(C=1)
    ```
    
    ❌ Don’t:
    
    ```python
    estimator = SVC([[0, 1], [1, 2]])
    ```
    
- Model hyper parameters should have default values, so that the user can instantiate a model without passing any arguments
- Every parameter should directly match an attribute, without additional logic to enable `model_selection`.
    
    ✅ Do:
    
    ```python
    def __init__(self, p_1=1, p_2=2):
    	self.p_1 = p_1
    	self.p_2 = p_2
    ```
    
    ❌ Don’t
    
    ```python
    def __init__(self, p_1=1, p_2=2):
    	if p_1 > 1:
    		p_1 += 1
       self.p_1 = p_1
    		
    	self.p_3 = p_2
    ```
    
- No parameter validation in `__init__`, only in `fit`

### Fitting

- For supervised
    
    ```python
    estimator = estimator.fit(X, y)
    ```
    
    or for unsupervised
    
    ```python
    estimator = estimator.fit(X)
    ```
    
- `kwargs` can be added, **restricted to data dependent variable**
    
    exemple: a precomputed matrix is data dependent, a tolerance criterion is not
    
- The estimator holds no reference to X or y (exceptions for precomputed kernel where this data must be stored for use by the predict method)
- Use utils `check_X_y` to ensure that X and y length are consistent
- Even if the fit method doesn’t imply a target `y`, `y=None` must be set to enable pipelines
- The method must return `self` for better usability with chained operations
    
    ```python
    y_pred = SVC(C=1).fit(X_train, y_train).predict(X_test)
    ```
    
- Fit must be idempotent, and any new call to fit overwrites the result of the previous call (exceptions when using `warm_start=True` strategy to speed-up next fit operations)
- Names of attributes created during fit must end with a trailing underscore: `param_`
- `n_features_in_` keyword can be added to make input expectations explicit

### Predictor

- For supervised or unsupervised:
    
    ```python
    prediction = predictor.predict(X)
    ```
    
- Classification can also offer to quantify a prediction (without applying thresholding)
    
    ```python
    prediction = predictor.predict_proba(X)
    ```
    

### Transformer

- For transforming or filtering data, in a supervised or unsupervised way
    
    ```python
    X_transformed = transformer.transform(X)
    ```
    
- When fitting and transforming can be implementing together more efficiently
    
    ```python
    X_transformed = transformer.fit_transform(X)
    ```
    

## Rolling your own estimator

### Back bone

- Test your estimator using `check_estimator`
    
    ```python
    from sklearn.utils.estimator_checks import check_estimator
    from sklearn.svm import LinearSVC
    check_estimator(LinearSVC())  # passes
    ```
    
- You can leverage the [project template](https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/_template.py) to get started with all the estimator required methods.
- You can also use inheritance from `BaseEstimator`, `ClassifierMixin` or `RegressorMixin` ot significantly reduce the amount of boilerplate code, including:
    - `get_params`: take no arguments, return fit parameters.
        
        use `deep=True` to return submodel parameters
        
    - `set_params`: overwrite fit parameters
    - `base.clone` compatibility is enable with `get_params`
    - `_estimator_type` must be `"classifier"` , `"regressor"` or `"clusterer"`
        
        This is automatic with `ClassifierMixin`, `RegressorMixin` or `ClusterMixin` inheritance
        

### Specific estimators

- A classifier fit can accept y with integers or string values, with the following conversion:
    
    ```python
    self.classes_, y = np.unique(y, return_inverse=True)
    ```
    
- A classifier predict method must return arrays containing class labels from `classes_`
    
    ```python
    def predict(self, X):
        D = self.decision_function(X)
        return self.classes_[np.argmax(D, axis=1)]
    ```
    
- In linear models, coefficient are stored in `coef_` and intercept in `intercept_`
- The `sklearn.utils.multiclass` module contains useful functions for working with multiclass and multilabel problems.
- Also, check [tags](https://scikit-learn.org/stable/developers/develop.html#estimator-tags) to define your estimator capabilities

## Coding guidelines

How new code should be written for inclusion in scikit-learn and make review easier.

### Style

- Format and indentation follows [PEP8](https://peps.python.org/pep-0008/)
- Use underscores to separate words in non class names: `n_samples` rather than `nsamples`.
- Avoid multiple statements on one line. Prefer a line return after `if/for`
- Use relative imports for references inside scikit-learn.
- **Please don’t use** `import *` **in any case**
    - code becomes hard to read
    - no reference for static analysis tool to run
- Use the [numpy docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html#numpydoc-docstring-guide)

Check the utils module for better integration and reusability