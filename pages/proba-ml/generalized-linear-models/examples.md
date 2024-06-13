# 12.2 Examples

### 12.2.1 Linear regression

The linear regression has the form:

$$
\begin{align}
\log p(y_n|\bold{x}_n,\bold{w},\sigma^2)&=-\frac{(y_n-\eta_n)^2}{2\sigma^2}-\frac{1}{2}\log (2\pi \sigma^2) \\
&= \frac{y_n\eta_n-\frac{\eta_n^2}{2}}{\sigma^2}-\frac{1}{2}\Big(\frac{y_n^2}{\sigma^2}-\log(2\pi\sigma^2)\Big)
\end{align}
$$

where $\eta_n=\bold{w}^\top \bold{x}_n$, and we see that $A(\eta_n)=\eta^2_n/2$

Hence:

$$
\begin{align}
\mathbb{E}[y_n]&=\eta_n=\bold{w}^\top \bold{x}_n \\
\mathbb{V}[y_n]&=\sigma^2
\end{align}
$$

### 12.2.2 Binomial regression

If the response variable is the number of success in $N_n$ trials, $y_n\in\{0,\dots,N_n\}$ we can use the **binomial regression**:

$$
p(y_n|\bold{x}_n,\bold{w},N_n)=\mathrm{Bin}(y_n|\sigma(\bold{w}^\top \bold{x}_n),N_n)
$$

We see that binary logistic regression is the special case when $N_n=1$.

The log pdf is:

$$
\begin{align}
\log p(y|\bold{x},\bold{w},N_n)&=y_n\log \mu_n + (N_n-y_n)\log (1-\mu_n)+\log \binom{N_n}{y_n} \\
&= y_n \log\frac{\mu_n}{1-\mu_n} + N_n\log (1-\mu_n)+\log \binom{N_n}{y_n} 
\end{align}
$$

where $\mu_n=\sigma(\eta_n)$, and $\eta_n=\frac{\mu_n}{1-\mu_n}$

We rewrite this in GLM form:

$$
\log p(y|\bold{x}_n,\bold{w},N_n)=y_n \eta_n-A(\eta_n)+h(y_n)
$$

with $A(\eta_n)=-N_n\log(1-\mu_n)= N_n \log (1+e^{\eta_n})$

Hence:

$$
\begin{align}
\mathbb{E}[y_n]&=\frac{d A}{d\eta_n}=\frac{N_n e^{\eta_n}}{1+e^{\eta_n}}=\frac{N_n}{1+e^{-\eta_n}}=N_n \mu_n \\
\mathbb{V}[y_n]&= \frac{d^2A}{d\eta_n^2}=N_n\mu_n(1-\mu_n)
\end{align}

$$

### 12.2.3 Poisson regression

If the response variable is an integer count, $y_n\in\{0,1,\dots\}$, we can use the **Poisson regression**:

$$
p(y_n|\bold{x}_n,\bold{w})=\mathrm{Poi}(y_n|\exp(\bold{w}^\top \bold{x}_n))
$$

where:

$$
\mathrm{Poi}(y|\mu)=e^{-\mu }\frac{\mu^y}{y!}
$$

The Poisson distribution is highly used in bio-stats application, where $y_n$ might represent the number of diseases at a given place.

The log pdf is:

$$
\begin{align}
\log p(y_n|\bold{x}_n,\bold{w})&=y_n \log \mu_n-\mu_n+\log y_n \\
&= y_n \eta_n -A(\eta_n)+h(y_n)
\end{align}
$$

where $\mu_n=\exp(\eta_n)=\exp(\bold{w}^\top \bold{x})$, and $A(\eta_n)= \mu_n$

Hence:

$$
\begin{align}
\mathbb{E}[y_n]&=\frac{d A} {d\eta_n}=\exp(\eta_n)=\mu_n\\
\mathbb{V}[y_n]&= \frac{d^2A}{d\eta_n^2}=\exp(\eta_n)=\mu_n
\end{align}
$$