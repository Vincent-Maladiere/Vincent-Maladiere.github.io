# 19. Faire’s journey toward modern data and ML stack, Daniele Perito, Faire

[https://www.youtube.com/watch?v=hTR-YSrN60Q&ab_channel=Tecton](https://www.youtube.com/watch?v=hTR-YSrN60Q&ab_channel=Tecton)

- Faire is a B2B marketplace helping independent retailers
    - ML to connect brands to retailers
    - Challenges include: search discovery & ranking, risk & fraud, seller success (lead scoring), incentive optimization

- Year 1 (data team size: 1 → 1)
    - Main decisions
        - Online DB to use
        - BI tool
        - Data warehouse
        - Event recording
        - First real-time feature store for fraud
    - Data structure decisions can be very sticky, not easy to evaluate in the short term, a lot of critical and far-reaching decisions
        
        Make sure to have cofounders or advisers that have worked with data systems
        
    - 2 choices of DB: MongoDB vs MySQL
        
        MongoDB pro: fast to set up and run
        
        MySQL pro: data replication, consistency
        

- Year 2 (data team size: 1 → 3)
    - Main decisions
        - Improved search system
        - Kubeflow vs Airflow for simple orchestration
        - First standardized core tables for analytics
    - So far, usage of GCP and Kubeflow seemed the most natural move
        
        When scaling, decision making needs a more framework: create a rubric
        

- Year 3 (data team size: 3 → 6)
    - Main decisions
        - Unify feature store
        - Migration to Airflow
        - Data quality monitoring
        - Experimentation platform
        - Events platform
    - Build / Buy experiment
        
        We had a basic experiment system and we didn’t know what we were missing, so we tried to use an external solution
        
        Took longer than expected and ultimately failed because of sync issues between our warehouse and our suppliers. Difficulty to map both data models.
        
    - Learnings: there is a cost in maintaining data consistency with external providers. Also, make sure that data infra decisions are tailored to your need.

- Year 4 (data team size: 6 → 12)
    - Main decisions
        - Real-time ranking and low latency inference
        - Migration to Snowflake
        - Terraform everywhere
    - Snowflake migration from Redshifts
        
        Thousands of tables to migrate, very painful over 6 months
        
        Big data infra changes are hard, and necessitate a lot of coordination and project management to bring the whole company with you
        
        No migration will be perfect, so set up an accuracy threshold to consider the job finished
        

- Year 5 (2021, data team size: 12 → 30)
    - Main decisions
        - ML platform unification
        - Analytics unification
    - Make sure you have seasoned technical leaders to understand when it’s time to make big investments
        
        Without feedback from DS and MLE and their workflow, you might not understand that there is a need to invest in analytics or ML platform
        
        These DS projects are critical for the future
        
- Summary
    - Make sure to have experimented folks in the early days
    - Close collaboration between eng and data team during the early days is crucial
    - Establish rubrics for decision making
    
- Q&A
    - A concrete example of a data decision rubric?
        
        Recently: does the data catalogue integrate with the BI tool? Can people follow tables to foster collaboration (”hey I notice you are an expert with this table…”)?
        
    - What data are you using?
        
        Essentially customers’ behaviour and interaction with the marketplace
        
    - Analytics platform vs ML platform?
        
        ML platform = unified feature store, unified ML training pipeline, model registry for later used, model serving (batch & real-time)
        
        Analytics platform = how are we adding decisions, and who are adding them? All of these things need to be defined at once, in a precise place. Every single fact of the business is defined once, and everyone is using it.
        
    - Real-time feature store?
        
        How grown, ranking and fraud system
        
        Very interested in a unified feature store and understanding when to make the switch
        
        Feature computation in real-time is a really hard topic, the cost of sync with an external provider is very real
        
    - Biggest growth challenges?
        
        Hundred of small changes in the data analytics platform during internationalization, different currencies, when to make the conversion before computing the revenue?