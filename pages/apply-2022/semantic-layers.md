# 16. Semantic Layers & Feature stores, Drew Banin, dbt Labs

- Semantic layers are *in*
    - Main idea: define your dataset and metrics, map out their relationships, translate semantic query into SQL
    - Example metrics: revenue per country, churn rate …
- One example
    
    ![Screen Shot 2022-05-24 at 15.18.56.png](./Screen_Shot_2022-05-24_at_15.18.56.png)
    
    - in dbt:
    
    ![Screen Shot 2022-05-24 at 15.19.27.png](./Screen_Shot_2022-05-24_at_15.19.27.png)
    
- Precision & consistency
    - Many people, many teams but only one way to define revenue
    - Avoid repeating work or copy-paste, and inconsistency can arise

- Bridging the gap
    - Standardisation: feature store for ML training and serving
    - Semantic layers: feature store in the BI world, output for analytics
    - Doing it once and correctly
    
    ![Screen Shot 2022-05-24 at 15.22.13.png](./Screen_Shot_2022-05-24_at_15.22.13.png)
    
    - get reuse and consistency