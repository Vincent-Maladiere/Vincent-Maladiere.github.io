# 12. Panel: What engineers should know when building for data scientists?

[https://www.youtube.com/watch?v=0lzTx5yEGRE&ab_channel=Tecton](https://www.youtube.com/watch?v=0lzTx5yEGRE&ab_channel=Tecton)

“⇒” are different spoke person intervention

- How are DS and Eng (SWE) different in problem-solving?
    
    ⇒ DS is such an umbrella, hard to find 2 people to approach their work, stack and problem solving the same way
    Analytics and ML engineer started popping out, depending on the scale and industry domain of the company
    Pretty clear boundaries where my work and your work start/end. However sometimes you just explore a topic, seems more like a hacking thing together. In a perfect scenario, engineers are supporting a bunch of MLE or DS
    
    ⇒ We’re in a data lifecycle, SWEs are data creators. They show DS insights to the end-users. there’s a key partnership and team effort.
    
    ⇒ Coming from the UX side, the scale and resources of the organisation impact a lot what is available (more resources = more specialized). At Spotify, DS are insights generators.
    
    ⇒ Cross func teams, PM + Backend + Frontend + Front. Hard for SWE to understand the experimental approach of DS. It’s a real adaptation of expectation, to face research uncertainty.
    

- Main challenges of working with SWE?
    
    ⇒ SWE and DS think differently. Experimentation phase. SWE give end goal and give requirements, but on a DS side not as straightforward.
    

- Who’s responsible for models in prod?
    
    ⇒ Every project has its own ownership and SREs are the main guard of everything.
    
    ⇒ Realistically, clear chart of ownership. It’s nearly always a collaboration between DS and MLE to handle crisis
    
    ⇒ As your increase complexity, pager duty becomes necessary. People at larger companies are responsible, accountable, consulted and then informed (RACI matrix). Create this matrix to draw boundaries of responsibilities, and see where gaps and bottlenecks are. Solve things early before conflicts.
    

- How do you feel about productizing ML vs productizing software? Where do we go beyond DevOps?
    
    ⇒ The major diff is data, larger chunks that make it different and act unpredictably
    
    ⇒ 2 main diff: regular deployment process (one-off), and expectations of testing and coverage
    
- Is eng the path for most DS? Or should DS should focus where they should add the most value
    
    ⇒ It comes down to data maturity. In startups, extra data eng skills are a superpower. Go back on the infra to support the DS model. In Spotify, this wouldn’t be the case, focus on the main thing adding value.