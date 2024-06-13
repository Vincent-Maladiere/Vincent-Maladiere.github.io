# 11.3 Ridge regression

Maximum likelihood estimation can result in overfitting, a simple solution is to use MAP estimation with zero-mean Gaussian prior:

$$
p(\bold{w})=\mathcal{N}(\bold{w}|0, \lambda^{-1} \mathrm{I})
$$

We compute the MAP as:

$$
\bold{w}_{map}=\argmin_{\bold{w}}\mathrm{RSS}(\bold{w}) + \lambda ||\bold{w}||_2^2
$$

Therefore we are penalizing the weights that become too large in magnitude. This is called $\ell_2$ regularization or weight decay.

We don’t penalize the term $w_0$ since it doesn’t contribute to overfitting.

### 11.3.1 Computing the MAP estimate

The MAP estimation corresponds to computing the minimizing the objective:

$$
J(\bold{w})=||\bold{y}-X\bold{w}||^2_2+ \lambda ||\bold{w}||^2_2
$$

we have:

$$
\nabla_\bold{w}J(\bold{w})=2(X^\top X\bold{w}-X ^\top \bold{y}+\lambda \bold{w})
$$

Hence:

$$
\begin{align}
\hat{\bold{w}}_{map}&=(X^\top X+\lambda \mathrm{I}_D)^{-1}X^\top  \bold{y} \\
&= (\sum_n \bold{x}_n\bold{x}_n^\top+\lambda \mathrm{I}_D)^{-1}(\sum_{n}\bold{x}_ny_n)
\end{align}
$$

**11.3.1.1 Solving using QR**

Naively performing the matrix inversion can be slow and numerically unstable. We propose a way to convert the problem to a standard least square, where we can apply QR decomposition as previously seen.

We assume the prior has the form $\mathcal{N}(0,\Lambda^{-1})$ where $\Lambda =(\lambda/\sigma^2)\mathrm{I}$ is the precision matrix.

We can emulate this prior by augmenting our training data as:

$$
\tilde{X}=\begin{bmatrix}X/\sigma\\ \sqrt{\Lambda}
\end{bmatrix},\quad
\tilde{\bold{y}}=\begin{bmatrix}\bold{y/\sigma}\\\bold{0_{D}}\end{bmatrix}
$$

where $\Lambda=\sqrt{\Lambda}\sqrt{\Lambda}^\top$  is the Cholesky decomposition.

We now show that the RSS of this expended data is equivalent to the penalized RSS on the original data:

$$
\begin{align}
f(\bold{w})&=(\bold{\tilde{y}}-\tilde{X}\bold{w})^\top(\tilde{\bold{y}}-\tilde{X}\bold{w}) \\
&= \Bigg(\begin{bmatrix}\bold{y}/\sigma\\\bold{0}\end{bmatrix} - \begin{bmatrix}X/\sigma\\\sqrt{\Lambda}\end{bmatrix}\bold{w}\Bigg)^\top \Bigg(\begin{bmatrix}\bold{y/\sigma}\\\bold{0}\end{bmatrix} - \begin{bmatrix}X/\sigma\\\sqrt{\Lambda}\end{bmatrix}\bold{w}\Bigg) \\ 
&=\begin{bmatrix}1/\sigma(\bold{y}-X\bold{w})\\-\sqrt{\Lambda}\bold{w}\end{bmatrix}^\top
\begin{bmatrix}1/\sigma(\bold{y}-X\bold{w})\\-\sqrt{\Lambda }\bold{w}\end{bmatrix}\\
&= \frac{1}{\sigma^2}(\bold{y}-X\bold{w})^\top(\bold{y}-X\bold{w})+\bold{w}^\top\Lambda\bold{w}
\end{align}
$$

Hence, the MAP estimate is given by:

$$
\hat{\bold{w}}_{map}=(\tilde{X}^\top \tilde{X})^{-1}\tilde{X}^\top \bold{\tilde{y}}
$$

And then solving using standard OLS method, by computing the QR decomposition of $\tilde{X}$. This takes $O((N+D)D^2)$

**11.3.1.2 Solving using SVD**

In this section, we assume $D>N$, which is a framework that suits ridge regression well. In this case, it is faster to compute SVD than QR.

Let $X=USV=RV$, with

- $U\in\mathbb{R}^{N\times N}$, so that $UU^\top =U^\top U=\mathrm{I}_N$
- $V\in\mathbb{R}^{N\times D}$, so that $V^\top V =I_N$
- $R\in\mathbb{R}^{N\times N}$

One can show that:

$$
\hat{\bold{w}}_{map}= V(R^\top R+\lambda \mathrm{I}_N)^{-1}R^\top \bold{y}
$$

In other words, we can replace the $D$-dimensional vector $\bold{x}_i$ with $N$-dimensional vector $\bold{r}_i$ and perform our penalized fit as before.

The resulting complexity is $O(DN^2)$, which is less than $O(D^3)$ if $N<D$

### 11.3.2 Connection between ridge regression and PCA

The ridge predictions on the training set are given by:

$$
\begin{align}
\hat{\bold{y}}=X\bold{\hat{w}}&=USV^\top V(S^2+\lambda I_N)^{-1}SU^\top \bold{y}\\
&= U\tilde{S}U^\top \bold{y}=\sum_{j=1}^D \bold{u}_j \tilde{S}_{jj}\bold{u}_j^\top \bold{y}
\end{align}
$$

with:

$$
\tilde{S}_{jj}\triangleq[S(S^2+\lambda I_N)^{-1}S]_{jj}=\frac{\sigma_j^2}{\lambda +\sigma_j^2}
$$

Hence:

$$
\hat{\bold{y}}=X\hat{\bold{w}}_{map}=\sum_{j=1}^D \bold{u}_j\frac{\sigma^2_j}{\sigma^2_j+\lambda}\bold{u}_j^\top \bold{y}
$$

In contrast, the least square prediction are:

$$
\hat{\bold{y}}=X\hat{\bold{w}}_{mle}=\sum_{j=1}^D \bold{u}_j\bold{u}_j^\top \bold{y}
$$

If $\sigma^2_j\ll \lambda$, the direction $\bold{u}_j$ will have a small impact on the prediction. This is what we want, since small singular values corresponds to direction with high posterior variance. These are the directions ridge shrinks the most.

There is a related technique called **principal components regression**, a supervised PCA reducing dimensionality to $K$ followed by a regression. However, this is usually less accurate than ridge, since it only uses $K$ features, when ridge uses a soft-weighting of all the dimensions.

### 11.3.3 Choosing the strength of the regularizer

To find the optimal $\lambda$, we can run cross-validation on a finite set of values and get the expected loss.

This approach can be expensive for large set of hyper-parameters, but fortunately we can often warm-start the optimization procedure, using the value of $\hat{\bold{w}}(\lambda_k)$ as a initializer for $\hat{\bold{w}}(\lambda_{k+1})$.

If we set $\lambda_{k+1}>\lambda_k$, we start from a high amount of regularization and gradually diminish it.

We can also use empirical Bayes to choose $\lambda$, by computing:

$$
\hat{\lambda}=\argmax_\lambda \log p(\mathcal{D}|\lambda)
$$

where $p(\mathcal{D|\lambda})$ is the marginal likelihood.

This gives the same result as CV estimate, however the Bayesian approach only fit a single model, and $p(\mathcal{D}|\lambda)$ is a smooth function of $\lambda$, so we can use gradient-based optimization instead of a discrete search.