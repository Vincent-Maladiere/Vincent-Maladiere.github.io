# 18.2 Ensemble learning

We saw that decision tree can be quite unstable, in the sense that their predictions might vary a lot with a small perturbation in the input data. They are high variance estimators.

A simple way to reduce the variance is to average over multiple models. This is called **ensemble learning**. The result model has the form:

$$
f(y|\bold{x})=\frac{1}{|\mathcal{M}|}\sum_{m\in\mathcal{M}}f_m(y|\bold{x})
$$

where $f_m$ is the $m$th base model.

The ensemble will have similar bias to the base models, but lower variance, generally resulting in better overall performances.

Averaging is a sensible way to combine predictions from regression models. For classifiers, we take a majority vote of the outputs (called **committee method**)

To see why this can help, suppose each base model is a binary classifier with accuracy $\theta$, and suppose 1 is the correct class. Let $Y_m\in\{0,1\}$ be the prediction for the $m$th model and $S=\sum_{m=1}^M Y_m$ the number of class of vote for class 1.

We define the final predictor to be the majority vote, i.e. class 1 if $S>M/2$ and class 0 otherwise. The probability that the ensemble will pick class 1 is:

$$
p(S>M/2)=1-B(M/2,M,\theta)
$$

where $B(x,M,\theta)$ is the cdf of the Binomial distribution evaluated in $x$.

For $\theta=0.51$ and $M=1000$, we get $p=0.73$. With $M=10,000$ we get $p=0.97$.

The performance of the voting approach is dramatically improved because we assume each predictor made independent errors. In practice, their mistakes might be correlated, but as long as we ensemble sufficiently diverse models, we still can come ahead.

### 18.2.1 Stacking

An alternative to using unweighted average or majority vote is to learn how to combine the base models, using **stacking** or **“stacked generalization”**:

$$
f(y|\bold{x})=\sum_{m\in\mathcal{M}} w_m f_m(y|\bold{x})
$$

We need to learn the combination weight on a separated dataset, otherwise all their mass will be put on the best performing base model.

### 18.2.2 Ensembling is not Bayes model averaging

Note that an ensemble of models is not the same as using BMA. An ensemble considers a larger hypothesis class of the form:

$$
p(y|\bold{x},\bold{w},\theta)=\sum_{m\in\mathcal{M}}w_mf_m(y|\bold{x},\theta_m)
$$

whereas BMA uses:

$$
p(y|\theta,\mathcal{D})=\sum_{m\in\mathcal{M}}p(m|\mathcal{D})p(y|\bold{x},m,\mathcal{D})
$$

The key difference is that the BMA weights $p(m|\mathcal{D})$ sum to one, and in the limit of infinite data, only a single model will be chosen (the MAP model). In contrary, the ensemble weights $w_m$ are arbitrary and don’t collapse in this way in a single model.