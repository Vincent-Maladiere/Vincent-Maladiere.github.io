# 12.1 Introduction

We previously discussed:

- the logistic regression model $p(y|\bold{x},\bold{w})=\mathrm{Ber}(\sigma(\bold{x}^\top \bold{w}))$
- the linear regression model $p(y|\bold{x},\bold{w})=\mathcal{N}(y|\bold{x}^\top \bold{w},\sigma^2)$.

For both of the models, the mean of the output $\mathbb{E}[y|\bold{x},\bold{w}]$ is a linear function of the input $\bold{x}$.

Both models belong to the broader family of **generalized linear models (GLM)**.

A GLM is a conditional version of an exponential family distribution, in which the natural parameters are a linear function of the input:

$$
p(y_n|\bold{x}_n,\bold{w},\sigma^2)=\exp \Big[\frac{y_n\eta_n-A(\eta_n)}{\sigma^2} +\log h(y_n,\sigma^2)\Big]
$$

where:

- $\eta_n\triangleq \bold{w}^\top \bold{x}_n$ is the (input dependent) natural parameter
- $A(\eta_n)$ is the log normalizer
- $y=\mathcal{T}(y)$ is the sufficient statistic
- $\sigma^2$ is the dispersion term

We denote the mapping from the linear inputs to the mean of the output using $\mu_n=\ell^{-1}(\eta_n)$, known as the **mean function,** where $\ell$ is the **link function.**

$$
\begin{align}
\mathbb{E}[y_n|\bold{x}_n,\bold{w},\sigma^2] &=A'(\eta_n) \triangleq \ell^{-1}(\eta_n) \\
\mathbb{V}[y_n|\bold{x_n},\bold{w},\sigma^2] &= A''(\eta_n)\sigma^2
\end{align}
$$