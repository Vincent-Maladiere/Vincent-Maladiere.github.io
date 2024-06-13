# 19.7 Weakly supervised learning

In **weakly supervised learning** we don’t have an exact label for each example in the training set.

*i)* One scenario is having a distribution over labels for each example, rather than a single one. Fortunately, we can still do ML training, by using the cross entropy:

$$
\mathcal{L}(\theta)=-\sum_{n}\sum_{y} p(y|\bold{x}_n)\log q(y|\bold{x}_n;\theta)
$$

where $p(y|\bold{x}_n)$ is the label distribution for example $n$ and $q(y|\bold{x}_n;\theta)$ is the predicted distribution.

It is often use to replace labels with a “soft” version, putting e.g. 90% of mass on a label and spreads the remaining 10% uniformly across the other choices, instead of using a delta function. This is called **label smoothing** and is a useful form of regularization.

*ii)* Another scenario is having a set of example $\bold{x}_n=\{\bold{x}_{n,1},\dots,\bold{x}_{n,B}\}$ and having a single label for the entire set $y_n$, not for individual members $y_{n,b}$.

We often assume that if one example in the set is positive, then all the set is labeled positive $y_n=\lor_{b=1}^B y_{n,b}$, without knowing which example “caused” the positive outcome.

However, if all members are negative, the set is negative. This is known as **multi-instance learning (MIL)** (this technique has recently been used for Covid19 risk-score learning).

Various algorithms can be used to solve the MIL problem, depending on the correlation of labels across the sets or the fraction of positive.

*iii)* Yet another scenario is known as **distant supervision**, in which we use a ground truth label like *Married(A, B)* to label every sentence in an unlabeled training corpus in which the entity A and B are mentioned as being a positive example of the “Married” relation.

For example “A and B invited 100 people to their wedding” will be labeled positive. But this heuristic might include false positives for example “A and B went out to dinner”, thus the labels will be noisy.

We discuss some way of handling it in section 10.4 with robust logistic regression.