# 14. Declarative ML Systems and Ludwig, Pierre Molino & Travis Addair, Predibase

[https://www.youtube.com/watch?v=74hqlj5k4Zg&ab_channel=Tecton](https://www.youtube.com/watch?v=74hqlj5k4Zg&ab_channel=Tecton)

- Organizations take inefficient ML approach
    
    ![Screen Shot 2022-05-24 at 14.46.05.png](./Screen_Shot_2022-05-24_at_14.46.05.png)
    
    - Each project takes too long to bring value
    - Bespoke solution are hard to maintain and bring tech debt
    - Organization can’t hire enough ML engineers

- Introducing declarative ML system
    
    ![Screen Shot 2022-05-24 at 14.47.46.png](./Screen_Shot_2022-05-24_at_14.47.46.png)
    
    - higher abstraction, ease of use
    - open the door to non experts for ML
    - Pioneer project with Ludwig (Uber) and Overton (Apple)

- How does Ludwig works? a configuration system with yaml
    
    ![Screen Shot 2022-05-24 at 14.49.40.png](./Screen_Shot_2022-05-24_at_14.49.40.png)
    
- End to end deep learning architecture

![Screen Shot 2022-05-24 at 14.51.16.png](./Screen_Shot_2022-05-24_at_14.51.16.png)

- Task flexibility

![Screen Shot 2022-05-24 at 14.51.53.png](./Screen_Shot_2022-05-24_at_14.51.53.png)

- How to scale this concept and work with bigger amount of data?
    - Scalable backend over Ray
    - Doesn’t require you to provision heavy weighted infra, like a spark cluster, everything on the same layer

![Screen Shot 2022-05-24 at 14.52.40.png](./Screen_Shot_2022-05-24_at_14.52.40.png)

- Predibase on top of Ludwig:
    - Take a look of the end-to-end problem of data flow in ML model to put it in production
    - Both batch and real-time production
    - Low code
    
    ![Screen Shot 2022-05-24 at 14.55.02.png](./Screen_Shot_2022-05-24_at_14.55.02.png)
    

- Workflow
    
    ![Screen Shot 2022-05-24 at 14.56.22.png](./Screen_Shot_2022-05-24_at_14.56.22.png)
    
    ⇒ Check their paper about declarative ML