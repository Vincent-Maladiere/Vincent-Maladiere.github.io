# 15. Accelerating model deployment velocity, Emmanuel Ameisen, Stripe

[https://www.youtube.com/watch?v=tClDQk7DqlY&ab_channel=Tecton](https://www.youtube.com/watch?v=tClDQk7DqlY&ab_channel=Tecton)

- Speeding up the art of ML model deployment
1. The value of redeployed models
    - Modelling and eng part to ship it ⇒ what happens when we have new features, or drift, etc
    - Any model that you train today will be obsolete tomorrow. By how much will it be obsolete?
    
    ![Screen Shot 2022-05-24 at 15.01.26.png](./Screen_Shot_2022-05-24_at_15.01.26.png)
    
    - Domain shift
    
    ![Screen Shot 2022-05-24 at 15.02.46.png](./Screen_Shot_2022-05-24_at_15.02.46.png)
    
    - Bottleneck when it comes to production
    
    ![Screen Shot 2022-05-24 at 15.03.15.png](./Screen_Shot_2022-05-24_at_15.03.15.png)
    
2. Skill set for regular ML deployments
    - Need to be 10x DS?
        
        ![Screen Shot 2022-05-24 at 15.04.35.png](./Screen_Shot_2022-05-24_at_15.04.35.png)
        
        - very operational work to bring value
    
3. Improving model release processes
    - automate the majority of the pipeline
        
        ![Screen Shot 2022-05-24 at 15.06.57.png](./Screen_Shot_2022-05-24_at_15.06.57.png)
        
    - another trick to de-risk the pipeline is to leverage shadow mode
        - deploy our prod model and also shadow, to observe how it behaves in production
    - Schedule it
        
        ![Screen Shot 2022-05-24 at 15.08.41.png](./Screen_Shot_2022-05-24_at_15.08.41.png)
        
    
    ⇒ [Stripe radar technical guide](https://stripe.com/en-fr/guides/primer-on-machine-learning-for-fraud-protection#next-steps)