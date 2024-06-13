# 16. Exemplar-based Models

So far, we have dealt with **parametric models**, either unconditional $p(\bold{y}|\theta)$ or conditional $p(\bold{y|x},\theta)$.

$\theta$ is a vector of parameters estimated from a training dataset $\mathcal{D}=\{(\bold{x}_n,\bold{y}_n),n=1:N\}$, which is thrown away after training.

In this section, we consider various kinds of **nonparametric model** that keep the training data at test time —we call them **examplar-based models**.

Therefore, the number of parameters can grow with $|\mathcal{D}|$, and we focus on the similarity (or distance) between a test input $\bold{x}$ and training inputs $\bold{x}_n$.