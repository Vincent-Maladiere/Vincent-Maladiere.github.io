# 23. ML meet SQL, Dan Sullivan, 4 Mile Analytics

[https://www.youtube.com/watch?v=T1StmzI0RbQ&ab_channel=Tecton](https://www.youtube.com/watch?v=T1StmzI0RbQ&ab_channel=Tecton)

- Big query is helpful for serverless warehouse
    
    A lot of features for a data analytics platform: big query ML, BI engine, GIS, Omni (outside of Google Cloud, part of the Kubernetes ecosystem) 
    
- Big query ML allow to create ML model in SQL
    
    Linear Regression, Matrix factorization, Boosting tree, Tensorflow, AutoML (params search and algo search for you with good performances)
    
    Hyperparams tuning is much easier
    
    No need to export data, no need to be proficient with Python or Java
    
- Create model
    
    ```sql
    CREATE MODEL `our_model`
    OPTIONS (
    	(model_type='linear_reg',
    	input_label_cols=['weight_pounds']
    ) AS
    SELECT
    	weight_pounds,
      feature_1,
    	feature_2,
      ...
    FROM big_query.dataset
    WHERE filter ...
    ```
    
- Predictions
    
    ```sql
    SELECT
    	predicted_weight_pounds
    FROM
    	ML.PREDICT(
    		MODEL `our_model`,
    		(
    			SELECT
    				is_male,
    				gestation_weeks,
    				mother_age,
    				...
    			FROM big_query.dataset
    			WHERE filter ...
    		)
    ```