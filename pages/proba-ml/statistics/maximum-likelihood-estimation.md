# 4.2 Maximum likelihood estimation (MLE)

### 4.2.1 Definition

Pick the parameter estimation assigning the highest probability to the training data, defined as:

$$
\begin{equation}
\hat{\theta}_{mle} \triangleq \argmax_{\theta}p(\mathcal{D}|\theta)
\end{equation}
$$

Wit the i.i.d. assumptions this becomes:

$$
\begin{equation}
p(\mathcal{D}|\theta)=\prod^N_{n=1}p(\bold{y}_n|\bold{x}_n,\theta)
\end{equation}
$$

And the Negative Log Likelihood (since most optimization algorithms are designed to minimize cost functions)

$$
\begin{equation}
NLL(\theta)\triangleq -\sum^N_{n=1}\log p(\bold{y}_n|\bold{x}_n,\theta)
\end{equation}
$$

With

$$
\hat{\theta}_{mle}= \argmin_{\theta} NLL(\theta)
$$

### 4.2.2 Justification for MLE

MLE can be viewed as a point estimation to the Bayesian posterior $p(\theta|\mathcal{D})$

$$
\hat{\theta}_{map}= \argmax_{\theta} \log p(\theta|\mathcal{D})=\argmax \log p(\mathcal{D}|\theta)+\log p(\theta)
$$

So $\hat{\theta}_{map}=\hat{\theta}_{mle}$ if the prior is uniform.

Another way to see the MLE is that the resulting predictive distribution $p(y|\hat{\theta}_{mle})$ is as close as possible to the empirical distribution of the data.

If we defined the empirical distribution by

$$
\begin{equation}
p_{\mathcal{D}}(\bold{y}) \triangleq \frac{1}{N} \sum_{n=1}^N\delta(y-y_n)
\end{equation}
$$

Minimizing the KL divergence between the empirical $p=p_{\mathcal{D}}$ and an estimated distribution $q=p(y_n|\theta)$ is equivalent to minimizing the NLL and therefore computing the MLE.

$$
\begin{align} D_{\mathbb{KL}}(p||q) &\triangleq \sum_y p(y) \log \frac{p(y)}{q(y)}
\\ &= -\mathbb{H}(p_{\mathcal{D}}) -\frac{1}{N} \sum_{n=1}^N \log p(y_n|\theta) \\
&= \mathrm{const} +\mathrm{NLL}(\theta)
\end{align} 
$$

The same logic applies for supervised settings, with:

$$
p_{D}(x,y)=p_{\mathcal{D}}(y|x) p_{\mathcal{D}}(x)=\frac{1}{N} \sum_{n=1}^N \delta(x-x_n)\delta(y-y_n)
$$

### 4.2.3 MLE for the Bernoulli distribution

Let $\theta =p(Y=1)$ be the probability of heads in a coin toss.

$$
\begin{align}
\mathrm{NLL}(\theta)&=-\log \prod^N_{n=1}p(y_n|\theta) \\
&=-\log \prod^N_{n=1}\theta^{\mathbb{I}(y_n=1)}(1-\theta)^{\mathbb{I}(y_n=0)} \\ &= -[N_1 \log\theta+N_0 \log(1-\theta)]
\end{align}
$$

The MLE can be found by:

$$
\frac{d}{d\theta} \mathrm{NLL}(\theta)=- \frac{N_1}{\theta}+\frac{N_0}{1-\theta} \Rightarrow \hat{\theta}_{mle}=\frac{N_1}{N}
$$

### 4.2.4 MLE for the categorical distribution

Let $Y_n \sim \mathrm{Cat}(\theta)$.

$$
\mathrm{NLL}(\theta)=-\sum_k N_k \log \theta_k
$$

To compute the MLE, we have to minimize the NLL subject the constraint $\sum^K_{k=1}\theta_k=1$ using the following Lagrangian:

$$
\mathcal{L}(\theta,\lambda) \triangleq -\sum_kN_k \log \theta_k-\lambda \Big(1-\sum_k \theta_k\Big)
$$

We get the MLE by taking the derivative of $\mathcal{L}$ for $\lambda$ and $\theta_k$

$$
\hat{\theta}_k=\frac{N_k}{\lambda}=\frac{N_k}{N}
$$

### 4.2.5 MLE for the univariate Gaussian

Suppose $Y \sim N(\mu,\sigma^2)$. We estimate here again the parameters using the MLE:

$$
\begin{align}
\mathrm{NLL}(\mu,\sigma^2) &=-\sum_{n=1}^N \log \Big[\big(\frac{1}{2\pi\sigma^2}\big)^{\frac{1}{2}} \exp\big(-\frac{(y_n-\mu)^2}{2\sigma^2}\big)\Big] \\ 
&= \frac{1}{2\sigma^2}\sum_{n=1}^N(y_n-\mu)^2 + \frac{N}{2}\log (2\pi\sigma^2)
\end{align}
$$

We find the stationary point by $\frac{\partial}{\partial \mu}\mathrm{NLL}(\mu,\sigma^2)=0$ and $\frac{\partial}{\partial \sigma^2}\mathrm{NLL}(\mu,\sigma^2)=0$

$$
\begin{align}
\hat{\mu}_{mle} &=\frac{1}{N}\sum_{n=1}^N y_n=\bar{y} \\
\hat{\sigma}^2_{mle} &= \frac{1}{N}\sum^N_{n=1}(y_n-\hat{\mu}_{mle})^2=s^2- \bar{y}^2 \\
s^2& \triangleq \frac{1}{N}\sum^N_{n=1}y_n^2
\end{align}

$$

$\bar{y}$ and $s^2$ are the sufficient statistics of the data, since they are sufficient to compute the MLE.

The unbiased estimator for variance (not the MLE) is:

$$
\hat{\sigma}^2_{unb}=\frac{1}{N-1} \sum^N_{n=1}(y_n-\hat{\mu}_{mle})^2
$$

### 4.2.6 MLE for MVN

$$
\mathrm{LL}(\mu,\Sigma)=\log p(\mathcal{D}|\mu,\Sigma)=-\frac{1}{2}\sum_{n=1}^N(y_n-\mu)^\top\Lambda(y_n-\mu)+\frac{N}{2} \log|\Lambda|
$$

with $\Lambda=\Sigma^{-1}$ the precision matrix.

Using $z_n=y_n-\mu$:

$$
\frac{\partial}{\partial \mu}(y_n-\mu)^\top \Sigma^{-1}(y_n-\mu)=\frac{\partial}{\partial z_n}z_n^\top\Sigma^{-1}z_n\frac{\partial z_n}{\partial \mu}=-(\Sigma^{-1}+\Sigma^{-T})z_n
$$

Hence

$$
\frac{\partial}{\partial \mu}\mathrm{LL}(\mu,\Sigma)=-\frac{1}{2}\sum -2\Sigma^{-1}(y_n-\mu)=0 \\\Rightarrow \hat{\mu}=\bar{y}
$$

So the MLE of $\mu$ is just the empirical mean.

Using the trace trick:

$$
\begin{align}
\mathrm{LL}(\hat{\mu},\Lambda)&=-\frac{1}{2} \sum_ntr[(y_n-\mu)(y_n-\mu)^\top \Lambda]+\frac{N}{2}\log |\Lambda| \\
&= -\frac{1}{2} tr[S_{\bar{y}} \Lambda]+\frac{N}{2}\log |\Lambda|
\end{align}
$$

With $S_{\bar{y}}=\sum_n(y_n-\bar{y})(y_n-\bar{y})^\top$ the scatter matrix centered on $\bar{y}$

Resolving the derivative for $\Lambda$ gives $\hat{\Lambda}^{-1}=\Sigma=\frac{1}{N} S_{\bar{y}}$

### 4.2.7 MLE for linear regression

Let suppose the model corresponds to $p(y|x,\theta)=\mathcal{N}(y|w^\top x,\sigma^2)$. If we fix $\sigma^2$ to focus on $w$:

$$
\mathrm{LL}(w)=-\sum_{n=1}^N \log \Bigg[ \Big(\frac{1}{2\pi\sigma^2}\Big)^{1/2} \exp\Big(-\frac{1}{2\sigma^2}(y_n-w^\top x_n)^2\Big) \Bigg]
$$

Dropping the irrelevant additive constants:

$$
\mathrm{RSS}(w) \triangleq \sum_{n=1}^N (y_n-w^\top x_n)^2=\sum^N_{n=1}r_n^2
$$

Note that $\mathrm{MSE}=\frac{1}{N}\mathrm{RSS}$ and $\mathrm{RMSE}=\sqrt{MSE}$

Writing the $\mathrm{RSS}$ in matrix notation:

$$
\mathrm{RSS}(w)=||\bold{X}w^\top-\bold{y}||^2_2
$$

And the equation of OLS is:

$$
\hat{w}_{mle} \triangleq \argmin_w \mathrm{RSS}(w) = \bold{(X^\top X)^{-1}X^\top y}
$$