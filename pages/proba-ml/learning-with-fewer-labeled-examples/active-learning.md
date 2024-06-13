# 19.4 Active learning

In active learning, the goal is to identify the true predictive mapping $y=f(\bold{x})$ by querying as few $(\bold{x},y)$ points as possible. There are three variants.

In **query synthesis**, the algorithm gets to choose any input $\bold{x}$ and can ask for its corresponding output $y=f(\bold{x})$.

In **pool-based** **active learning**, there is large, but fixed, set of unlabeled points and the algorithm gets to ask a label for one or more of these points

Finally, in **stream-based active learning** the incoming data is arriving continuously, and the algorithm must choose whether it wants to request a label for the current input or not.

They are various closely related problems. In **Bayesian optimization** the goal is to estimate the location of the global optimum $\bold{x}^*=\argmin_{\bold{x}}f(\bold{x})$ in as few queries as possible. Typically, we fit a surrogate (response surface) model to the intermediate $(\bold{x},y)$ queries, to decide which question to ask next.

In **experiment design**, the goal is to estimate $p(\theta|\mathcal{D})$, using as little data as possible (this can thought of as an unsupervised form of active learning).

In this section, we discuss the pool based approach.

### 19.4.1 Decision-theoretic approach

We define the utility of querying $\bold{x}$ in terms of the **value of information**. We define the utility of issuing query $\bold{x}$ as:

$$
U(\bold{x})\triangleq \mathbb{E}_{p(y|\bold{x},\mathcal{D})}[\min_a (R(a|\mathcal{D})-R(a|\mathcal{D},(\bold{x},y)))]
$$

where $R(a|\mathcal{D})=\mathbb{E}_{p(\theta|\mathcal{D})}[\ell(\theta,a)]$ is the posterior expected loss of taking some future action $a$ given the data $\mathcal{D}$ observed so far.

Unfortunately, evaluating $U(\bold{x})$ for each $\bold{x}$ is quite expensive, since for each possible response $y$ we might observe, we have to update our beliefs given $(\bold{x},y)$ to see what effect it might have on our future decisions (similar to look ahead search technique applied to belief states).

### 19.4.2 Information-theoretic approach

In the information-theoretic approach, we avoid using a task specific loss function and focus on learning our model as well as we can.

It has been proposed to define the utility of querying $\bold{x}$ in terms of information gain about the parameter $\theta$, ie the reduction in entropy:

$$
\begin{align}
U(\bold{x})&=\mathbb{H}(p(\theta|\mathcal{D}))-\mathbb{E}_{p(y|\bold{x},\mathcal{D})}[\mathbb{H}(p(\theta|\mathcal{D}, (\bold{x},y)))]\\
&= I(\theta, y|\mathcal{D}, \bold{x}) \\
&= \mathbb{H}(p(y|\bold{x},\mathcal{D}))-\mathbb{E}_{p(\theta|\mathcal{D})}[\mathbb{H}(p(y|\bold{x},\theta))]

\end{align}
$$

We used the symmetry of the mutual information to get (3), and the advantage of this approach is that we now only have to reason about the uncertainty of the predictive distribution over $y$ and not over $\theta$.

Note that this objective is identical to the expected change in the posterior over the parameter:

$$
U'(\bold{x})\triangleq\mathbb{E}_{p(y|\bold{x},\mathcal{D})}[D_{\mathbb{KL}}\big(p(\theta|\mathcal{D},(\bold{x},y))||p(\theta|\mathcal{D})\big)]
$$

Eq (3) has an interesting interpretation: the first term prefers example $\bold{x}$ for which there is uncertainty in the label. Just using it as a criterion selection is called **maximum entropy sampling**.

However, this can have problems with ambiguous or mislabeled examples. The second term in the equation will discourage this behavior, since it prefers examples $\bold{x}$ that are fairly certain once we know $\theta.$

In other words, this equation will select examples $\bold{x}$ for which the model makes confident predictions which are highly diverse. This approach is called **Bayesian active learning by disagreement (BALD)**

This methods can be used to train classifiers where expert labels are hard to acquire, such as medical images.

### 19.4.3 Batch active learning

So far, we have assumed a greedy strategy, in which we select a single example $\bold{x}$. Sometimes, we have a budget to select a set of $B$ samples.

In this case, the information criterion becomes:

$$
U(X)=\mathbb{H}(p(\theta|\mathcal{D}))-\mathbb{E}_{p(Y|X,\mathcal{D})}\mathbb{H}(p())
$$

Unfortunately, optimizing this is NP-hard in the horizon length $B$.

Fortunately, the greedy strategy is near-optimal in certain conditions.

First note that, for any given $X$, the information gain function is:

$$
f(Y)\triangleq \mathbb{H}(p(\theta|\mathcal{D}))-\mathbb{H}(p(\theta|Y,X, \mathcal{D}))
$$

$f$ maps a set of labels to a scalar.

It is clear that $f(\empty)=0$, and that $f$ is non-decreasing, i.e. $f(Y^{\mathrm{large}})\geq f(Y^{\mathrm{small}})$, due to the “more information never hurts” principle.

It has also been shown that $f$ is **submodular**.

As a consequence, a sequential greedy approach is within a constant factor of optimal. If we combine this greedy technique with BALD objective, we get a method called **BatchBALD**.