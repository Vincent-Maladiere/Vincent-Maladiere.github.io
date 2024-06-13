# 27. Fire chat: Is ML a subset or superset of programming? Mike & Martin

[https://www.youtube.com/watch?v=urx00Wfm4dw&ab_channel=Tecton](https://www.youtube.com/watch?v=urx00Wfm4dw&ab_channel=Tecton)

- Where is ML delivering and undelivering?
    
    Continuum of views:
    
    - The most bearish: “AI/ML is nothing new, just expensive regression”
    - The most bullish: “So powerful we won’t need theories anymore”
    
    Where in this continuum does reality lies?
    
    There’s absolutely something innovative and new. Implications on consumption layer and world understanding.
    
    What we don’t know is how far that goes. This doesn’t cut humans out of the loop, not the end of the theory.
    

- Intermediate steps?
    
    Some companies come in and tell the “end of theory” story. Try to use AI to find PMF. It’s too bullish.
    
    More pragmatic approach: documents are tremendously unstructured, some rule-based approaches to solve this problem. And we’re going to apply this domain where they’re buyers.
    

- Is ML fundamentally different from software eng?
    
    How do you break down the AI market? Analytics (computer-aided decision making) and operational ML
    
    Analytics is a natural extension of traditional analytics. The assumption is that the output of dashboards and BI is to be consumed by human
    
    On the operational one, AI can be anywhere in the program being used.
    
    Otherwise, ML is fundamentally different. When it comes to traditional system building, we’re dealing with finite state machines, and control it, control the complexity. For ML it is different, complexity of the natural universe. The state spaces are huge. No tool can manage that, new skillsets and new tools are required.
    

- Does ML system handle continuous state?
    
    Computers are the right architecture for everything. The nature of problems we’re solving requires more than software. AI is part of the system, so ML is a superset.
    
    Some set of approaches to handle data that is very messy, you’re supposed to put structure on that.
    

- Are there different layers on the ML stack?
    
    You need people to understand this stuff. The role for someone that can extract features to build software. Skillset much matches a set of tooling to help them. Insights from these teams translate to traditional system building
    
- For the average MLE or DS, what does it mean for them, does that boil down to adding structure to datasets where there is no structure? What skills to be great at?
    
    You never really have a raw data source. Adding structure to unstructured data is not refining raw data. Being really good at figuring out how to extract info from a dataset, also higher-level reasoning “what dataset could be out there, and what could I extract”, and this skill won’t come from an AutoML tool. Having a solid mental of causality.
    
    Ultimately you’re trying to build predictions, there are some domains where it doesn't apply like sub-similarity. Some companies claim to build chatbots on some domains, and some realizes that they can answer less than 50% of queries.
    
    Data centers don’t handle traffic smartly but send it to random destinations. No matter the zoom level, the stochastic probabilities are the same. Before starting an ML project, ask whether ML makes sense in the first place.
    
    There are some problems where linear regression is fine. Some can’t be solved with ML. Other problems actually have the data and seem a good fit, but you don’t know until you know the distribution. Think more like scientists than engineers.
    

- Are there other signs to look for when we don’t know the distribution of the data, that can save us the 6 months of pain?
    
    In a practical perspective, when ML is not off the right start? When is it doom to fail?
    
    - When you don’t have the right data of right skillset on your team
    - Do you have level of organizational/executive support. Putting ML in prod today takes a lot of work, tool and team. Doing that is an investment. Make sure it’s both doable and worth it. The business value might not be high enough for ML.
    - For seed-stage and A-stage, building the ML infrastructure is a real challenge.
    - Difficulty to put operational ML in production on the long term if often underestimate
    - Data products often allow companies to stand out from competition though