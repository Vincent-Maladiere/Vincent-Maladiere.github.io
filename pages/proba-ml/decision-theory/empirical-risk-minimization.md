# 5.4 Empirical risk minimization

How to apply frequentist decision theory to supervised learning?

### 5.4.1 Empirical risk

In supervised learning, the true state of nature is the distribution $p^*(x,y)$ and the estimator $\pi$ is a prediction function $f(x)=\hat{y}$.

We define the population risk as:

$$
R(f,p^*)=R(f)\triangleq \mathbb{E}_{p^*(y|x)p^*(x)}[\ell(f(x),y)]
$$

We can approximate $p^*$ using its empirical distribution:

$$
p_{\mathcal{D}}(x,y|\mathcal{D})=p_{training}(x,y|\mathcal{D})=\frac{1}{|\mathcal{D}|}\sum_{x_n,y_n \in \mathcal{D}} \delta(x-x_n)\delta(y-y_n)
$$

Plugging this gives the empirical risk:

$$
R(f,\mathcal{D})=\mathbb{E}_{p_\mathcal{D}(x,y)}[\ell(f(x_n),y_n)]=\frac{1}{N}\sum_{n=1}^N\ell(f(x_n),y_n)
$$

$R(f,\mathcal{D})$ is a random variable since it depends on the training set. We chose an estimator by minimizing the empirical risk over a specific hypothesis space of functions $\mathcal{H}$:

$$
\hat{f}_{ERM}=\argmin_{f\in\mathcal{H}} R(f,D)
$$

**Approximation error vs generalization error**

Let’s define:

- $f^{**}=\argmin_f R(f)$, the function that achieve the minimal possible population risk
- $f^* =\argmin_{f\in\mathcal{H}}R(f)$ the best function of our hypothesis space $\mathcal{H}$
- $f_{N}^*=\argmin_{f\in\mathcal{H}}R(f,\mathcal{D})=\mathbb{E}_{p_{tr}}[\ell(y,f(x))]$ the function that minimizes the empirical risk, since we can’t compute the population risk.

One can show that the risk of our chosen predictor can be compared to the best possible estimator with a two terms decomposition: the approximation error and the generalization error.

We can approximate this by the difference between the training and testing set errors:

$$
\begin{align}
\mathbb{E}_{p^*}[R(f^*_N)-R(f^{**})]&=\overbrace{R(f^*)-R(f^{**})}^{\mathcal{E}_{app}(\mathcal{H})} + \overbrace{\mathbb{E}_{p^*}[R(f^*_N)-R(f^*)]}^{\mathcal{E}_{est}(\mathcal{H,N})}
\\ & \approx \mathbb{E_{p_{tr}}}[\ell(f^*_N(x), y)]-\mathbb{E_{p_{te}}}[\ell(f^*_N(x), y)]
\end{align}
$$

We can reduce the approximation error with a more complex model, but it may result in overfitting and increasing the generalization error.

**Regularized risk**

We add a complexity penalty $C(f)$ to the objective function. Since we usually work with parametric functions, we apply the regularizer to the parameters $\theta$ themselves:

$$
R_\lambda(\theta, \mathcal{D})=R(\theta, \mathcal{D})+\lambda C(\theta)
$$

with $\lambda >0$ a hyperparameter.

If the risk is the log-loss and the penalty is the negative log prior, minimizing the empirical regularized risk is equivalent to estimating the MAP:

$$
R_\lambda(\theta,\mathcal{D})=-\frac{1}{N}\sum_{n=1}^N \log p(y_n|x_n,\theta)-\lambda \log p(\theta)
$$

### 5.4.2 Structural risk

We estimate the hyperparameters with bilevel optimization:

$$
\hat{\lambda}=\argmin_\lambda \min_\theta R_\lambda(\theta,\mathcal{D})
$$

However, this won’t work since this technique will always pick the least amount of regularization $\hat{\lambda}=0$.

If we knew the population risk, we could minimize the **structural risk** and find the right complexity (the value of $\lambda$).

We can estimate the population risk via cross-validation or statistical learning.

### 5.4.3 Cross validation

Let $\mathcal{D}_{train}$ and $\mathcal{D}_{val}$ be two partitions of $\mathcal{D}$, and:

$$
\hat{\theta}_\lambda(\mathcal{D})=\argmin_\theta R_\lambda(\mathcal{D}, \theta)
$$

For each model $\lambda$, we fit it to the training set to get $\hat{\theta}_\lambda(\mathcal{D}_{train})$.

We then use the unregularized risk on the validation set as an estimate of the population risk:

$$
R_\lambda^{val}\triangleq R_0(\hat{\theta}_\lambda(\mathcal{D}_{train}),\mathcal{D}_{val})
$$

### 5.4.4 **Statistical learning theory (SLT)**

Cross validation is slow, so in SLT we derive analytically upper bound of the population risk (or more precisely the generalization error). If the bound is satisfied, we can be confident that minimizing the empirical risk will have low population risk.

For binary classifiers, we say the hypothesis is probably approximately correct (PAC), and the hypothesis class is PAC learnable.

**Bounding the generalization error**

Let use a finite of hypothesis space and $\dim(\mathcal{H)=|H|}$. For any dataset $\mathcal{D}$ of size $N$ drawn from $p^*$:

$$
P(\max_{f\in\mathcal{H}}|R(f)-R(f,\mathcal{D})|>\epsilon)\leq 2\dim(\mathcal{H)}e^{-2N\epsilon^2}
$$

where:

- $R(f,\mathcal{D})=\frac{1}{N}\sum_{i=1}^N \mathbb{I}(f(x_i)\neq y_i)$ is the population risk
- $R(f)=\mathbb{E}[\mathbb{I}(f(x)\neq y^*)]$ is the empirical risk

**VC dimension**

If the hypothesis space is infinite, we need to use an estimate of the degree of freedom of the hypothesis class. This is called VC dimension. Unfortunately this is hard to compute and the bounds are very loose.

However various estimates of the generalization error like the PAC-Bayesian bounds have recently been designed, especially for DNNs.