# 13. Is production RL at a tipping point? Waleed Kadous, Anyscale

[https://www.youtube.com/watch?v=ufdhVCj-tpg&ab_channel=Tecton](https://www.youtube.com/watch?v=ufdhVCj-tpg&ab_channel=Tecton)

- There’s a very standard way of doing supervised ML, but reinforcement learning (RL) challenges that
    - Research has shown that RL does super well on real-world tasks
    - Yet we rarely see RL in prod. Why? Sometimes we get freaked out by RL because it's new, but we’ll show that it’s a natural extension.
    - I’ll give some tips and traps to watch out for
    - Using [RLlib](https://www.ray.io/rllib), popular open-source distributed library
- Understanding RL structure
    - How to escalate from Bandit
    - Deployment will be discussed
    
- Bandit
    - You have slots machines giving a payout based on an unknown proba. A good example if UI treatment: 5 different ways of showing UI. You don’t want to test them all uniformly
    - The challenge with Bandit in production is the explore/exploit tradeoff. How to balance it? Epsilon greedy algorithm is a tradeoff. 50% of the time you act random and 50% you maximize your gain based on your knowledge
    - Contextual bandit leverage metadata: is it sunny? A very natural extension of bandit. based on this user profile, and watch this episode last, I’ll suggest this user watch this
    
- RL is bandit with states (or sequential)
    - Order of actions in chess
    - A sequence of steps to a payout, how to distribute the reward to the last move?
    - You need a temporal credit assignment.
    - There is also a delay problem. What happens if the reward is delayed? Even more complex, like a sacrifice move in chess, negative reward in the short term, but a reward 50 moves later.

- If we expand the state and action spaces
    - Instead of 4 bandit machines, I have 32
    - State-space grows exponentially

- For so many real-world applications, I got a log (here’s where people clicked yesterday), learning RL policy, how to run it without experiment?
    - When you move to offline RL, you are stuck with the historic.
    
- Multi-agent RL (MRL)
    - How do you share probability between different users?
    - Is it a cooperative or competitive scenario?
    - The stock market is a sum of very small players
    - MRL get way more complex
    - How do you model reward between all actors
    
- Is RL at a tipping point?
    - All companies use RL for recommenders, so it’s started to cross the boundary, but why not more popular? 4 factors before prod ready. Recent progress in each
        1. Huge amount of training: alpha go played 5 million games. Only huge tech can afford that. We started to see transfer learning, you don’t always have to start to scratch. Imitation learning mimics human behaviour.
        2. The default implementation is online: it’s designed to run live. Changing a dynamic model in production is pretty scary. Hard to get data to train them. Offline learning can help.
        3. Temporal credit assignment: which action to reward? Contextual bandit is RL without the temporal credit assignment, limited but simple to deploy and start getting adopted
        4. Large action and state space. Recently, high fidelity simulator, deep learning approach to learning the state space, embedding approaches to learn action space (candidate selection then ranking), offline learning doesn’t require relearning
    
- 3 Common patterns in successful RL applications
    1. a good simulator is 50% of the work
        
        running a lot of simulations at once using distributed RL (RLlib)
        
        batching by merging results for many experiments
        
        getting close with a simulator, then fine-tuning in real-world
        
        ex1: games are good simulators!
        
        ex2: markets. simulations don’t need to be perfect.
        
    2. low temporality: do you really need temporal assignment?
        
        ex1: last played game + user profile (the contextual part)
        
        what if millions of users and hundred games? use embedding to reduce the dimensionality of users and embedding to find the game
        
    3. optimization: the next generation. Linear programming can be used with RL.
        
        RL is optimization in a data-driven way, does not require modelling, but many experiments. Obviously, it takes a lot more computation but often plug and play with optimization
        
- 2 tips for production RL
    1. Keep it simple.
    start with stateless, then add context and up forward.
    online vs offline
    small discrete state and action spaces vs large and continuous
    single-agent vs multi-agent shared policy vs true multi-agent
    2. RLOps?
    Workflow is different
    Validate? Update? Monitoring? Retrain when it's offline data? Real problems with RL in production
    
- Conclusion: a tipping point in some areas, some early adopters.
    
    
- Q&A
    - How much training data does a simple recommender system need?
        
        ⇒ Think about embeddings: you don’t need RL to build them. 10k examples can be enough (ideally 100k or 1m).
        Contextual bandit is quite off the shelf now.