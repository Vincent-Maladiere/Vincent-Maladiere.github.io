# 9. Linear Discriminant Analysis

We consider classification models of the form:

$$
p(y=c|\theta, x)=\frac{p(x|y=c,\theta)p(y=c,\theta)}{\sum_c' p(x|y=c',\theta)p(y=c',\theta)}
$$

where $p(y=c,x)$ is the prior over the class labels, and $p(x|y=c,\theta)$ is the class conditional density for $c$.

- The overall model is a **generative model** since it specifies the distribution over the feature $x$, $p(x|y=c,\theta)$
- By contrast, a **discriminative model** directly estimates the class posterior $p(y=c|\theta,x)$