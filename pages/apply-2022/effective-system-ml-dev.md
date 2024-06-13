# 10. Effective system ML development, Leonard Aukea, Volvo

[https://www.youtube.com/watch?v=Tb_IKFvlFo8&ab_channel=Tecton](https://www.youtube.com/watch?v=Tb_IKFvlFo8&ab_channel=Tecton)

👉 [Slides](https://www.dropbox.com/s/66ett3t7h2ra2hl/apply(conf)-Leonard)

- ML is (data-intensive) software: let’s not forget it
- Uncertainty is a feature of ML, also sometimes a bug
- Harder to test: test model + test data
- Some low hanging fruits
    - care about system design
    Not being done properly in practice
    - adopt a branching strategy
    - review process: code and analysis, ensure quality & distribute knowledge across the team
    - write tests, statistical tests as well. adopt this mentality
    - documentation is paramount, your approach, your analysis, and your codebase in general
    - monitoring and alerting, beware of silent errors
    - automation: learn how to use git properly to use CI/CD
    - plan for disaster: prepare a disaster recovery plan. start simple and iterate on it
    
- Q&A
    - Elaborate on branching and review process?
        
        ⇒ git flow is quite simple for branching strategy. build a solid foundation to collaborate. running integration needs to have a branching strategy.
        ⇒ the review process needs different perspectives: some software dev and also senior MLE or DS
        make it more interactive to look at plot / dist, and ensure quality. not merge unless it’s been reviewed
        
        ⇒ we haven’t cared about these things at all in the ML sphere, we need to shape up
        
    - what disaster ML technique have practitioners used?
        
        ⇒ should be discussed during the design process, setting up requirements, and which scope it should function. running some type of stress testing to estimate the worst-case scenario in production, things that might be exposed to the user.