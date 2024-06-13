# 12.4 Maximum likelihood estimation

GLMs can be fit similarly to logistic regression. In particular, the NLL is:

$$
\mathrm{NLL}(\bold{w})=-\log p(\mathcal{D}|\bold{w})=-\frac{1}{\sigma^2}\sum_{n=1}^N \ell_n
$$

where:

$$
\ell_n\triangleq y_n\eta_n-A(\eta_n)
$$

and $\eta_n=\bold{w}^\top \bold{x}_n$. We assume $\sigma^2=1$.

We can compute the gradient as follow:

$$
\bold{g}_n=\frac{\partial \ell_n}{\partial \bold{w}}=\frac{\partial \ell_n}{\partial \eta_n}\frac{\partial \eta_n}{\partial \bold{w}}=(y_n-A'(\eta_n)) \bold{x}_n=(y_n-\mu_n)\bold{x_n}
$$

where $\mu_n=f(\eta_n)$ and $f$ is the inverse link function mapping the canonical parameters to the mean parameters.

In the case of logistic regression, we have $\mu_n=\sigma(\eta_n)$.

This gradient expression can be used in SGD or other gradient methods.

The Hessian is given by:

$$
H=\frac{\partial^2}{\partial \bold{w}\partial \bold{w}^\top }\mathrm{NLL}(\bold{w})=-\sum_{n=1}^N  \frac{\partial \bold{g_n}}{\partial \bold{w}^\top }
$$

where:

$$
\frac{\partial \bold{g}_n}{\partial \bold{w}^\top}=\frac{\partial \bold{g}_n}{\partial \eta_n}\frac{\partial \eta_n}{\partial \bold{w}^\top } =-\bold{x}_nf'(\eta_n)\bold{x}_n^\top 
$$

hence:

$$
H=\sum_{n=1}^N f'(\eta_n)\bold{x}_n\bold{x}_n^\top
$$

For example, in the case of logistic regression, $f'(\eta_n)=\sigma'(\eta_n)=\sigma(\eta_n)(1-\sigma(\eta_n))$

In general, we see that the Hessian is positive definite since $f'(\eta_n)>0$, hence the NLL is convex, so the MLE for the GLM is unique (assuming $f(\eta_n)>0$ for all $n$).