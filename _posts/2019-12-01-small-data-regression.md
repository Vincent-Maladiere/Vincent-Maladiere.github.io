---
title: 'Small Data [1/4] - Regression'
date: 2019-12-01
permalink: /posts/2019/12/small-data-regression/
tags:
  - machine learning
  - linear regression 
  - ridge 
  - lasso
  - OLS
  - ALS
  - residuals
---

# Small Intro & Aknowledgment
---

Hello reader, this article is a short version of the MS&E 226 [Stanford course](http://web.stanford.edu/~rjohari/teaching/notes/)
of Ramesh Johari, who kindly let me use his materials, so all credits go to him.

This is the first part of the small data series. I wrote it after an extensive reading of this course so that you don't have to.
This series aims at mastering the skills that will help you for "small" data analysis and thus for any data analysis.

Note that all the code was written in R langage in the original papers and that I converted it into Python.


# 1 Linear Regression
---

The amount of data we collect is gargantuous (Facebook posts, tweets, medical tests...) and it often arriving faster than we can store and analyze it. 
> "Big" data can't be analyzed on a single machine.
On the contrary, small data can be analyzed and collected on a single machine (even though we now have 64GB of RAM at home).

We will use the [child.iq](http://www.stat.columbia.edu/~gelman/arm/examples/child.iq/) data throughout this article.
We will see which are the most important variables to explain child's IQ thanks to inputs like mother's IQ, 
mother's workload during child's first years of life and so on.

[1-1 summerizing a sample](#1-1-summerizing-a-sample)<br>
[1-2 modeling relationships](#1-2-modeling-relationships)<br>
[1-3 linear regression model](#1-3-linear-regression-model)<br>
[1-4 ordinary least square](#1-4-ordinary-least-square)<br>
[1-5 residuals and R2](#1-5-residuals-and-R2)<br>
[1-6 Interpreting regression coefficients](#1-6-interpreting-regression-coefficients)<br>
[1-7 Beyond linearity](#1-7-beyond-linearity)<br>
[1-8 Beyond minimizing MSE: Ridge and Lasso Penalization](#1-8-beyond-minimizing-sse)<br>
[1-9 Data Transformations](#1-9-data-transformations)<br>
[1-10 Alternating Least Square (ALS)](#1-10-alternating-least-square)<br>

## 1-1 Summerizing a sample
How can we summerize a sample? We first use simple statistics:
+ Sample mean
+ Sample median (more robust to outliers)
+ Sample standard deviation

All of them can be computed in one shot using pandas:
<script src="https://gist.github.com/Vincent-Maladiere/92c2cf07035962f50e23a72b57299208.js"></script>

## 1-2 Modeling relationships
We focus on modelizing relationships between observations. 

+ Let $Y=(Y_1, ..., Y_n)$ be the outcome, or target.
+ Let $X$ be the features where rows are $X_i=(X_{i1}, ..., X_{ip})$.

Our goal is to find a functional relationship such that

$$Y_i \approx f(X_i)$$

To answer the question *how is* `kid_score` *related to the other variables?*

<script src="https://gist.github.com/Vincent-Maladiere/a5bfd0162e30fca9596d89b1a4b3e2d8.js"></script>

We have:
+ continuous variables: `kid_score` and `mom_iq` (they can be constrained, as `mom_iq` can't be negative)
+ categorical variables: `mom_hs` is $0$ if the mother did attend high school, o.w. 1. `mom_work` range from $1$ to $4$
  + $1=$ did not work in the first 3 years of child's life
  + $2=$ worked in 2nd or 3d year
  + $3=$ worked part-time in first year
  + $4=$ worked full-time in first year

We also use model for:
+ Associations and correlations
+ Predictions
+ Causal relationships

## 1-3 Linear regression model

We first focus on modeling linear relationship between outcomes and covariates.
We look for coefficients 

$$\hat{\beta} = [\hat{\beta}_0, ..., \hat{\beta}_p]^T$$ 

such that

$$Y_i \approx \hat{\beta}_0 + \hat{\beta}_1 X_{i1} + ... + \hat{\beta}_p X_{ip} = X_i \hat{\beta}$$

A picture of our $Y$, $X$ and $\hat{\beta}$:
<script src="https://gist.github.com/Vincent-Maladiere/20d422af2c70e8e8598e31986f249711.js"></script>

Let's build a simple regression model of `kid_score` against `mom_iq`.
<script src="https://gist.github.com/Vincent-Maladiere/62e8ed53b3407f53f5ac28d2ac36f1d4.js"></script>

That is to say, `kid_score = mom_iq x 0.61 + 25.8`.
Let's plot our model

<script src="https://gist.github.com/Vincent-Maladiere/afb6f817f3d671558621d9c0f94af28f.js"></script>

<br>
So, how to choose $\hat{\beta}$?
We focus on *ordinary least square* (OLS).
We choose $\hat{\beta}$ so that

$$SSE=sum\;of\;squared\;errors=\sum^n_{i=1}(Y_i-\hat{Y}_i)^2=\|Y-X\hat{\beta}\|^2$$

is minimized, where

$$\hat{Y_i}=X_i\hat{\beta}=\hat{\beta}_0+\sum^p_{j=1}\hat{\beta}_j X_{ij}$$

is the *fitted* value of the $i$'th observation.

+ Is the resulting model a good fit?
+ Does it make sense to use a linear model?
+ Is minimizing SSE the right objective?

We start down this road by working through the algebra of *linear regression*.

## 1-4 Ordinary least squares

The vector $\hat{\beta}$ that minimizes SSE is given by

$$\hat{\beta}=(X^TX)^{-1}X^TY$$

<details>
<summary>The key here is that $X^TX$ is invertible, whereas $X$ may not be.</summary>
$$X^TXq=0 \implies q^TX^TXq=0 \implies (Xq)^TXq=0 \implies \|Xq\|^2=0 \implies Xq=0$$
So $q = 0$ and thus the null space of $X^TX = \{0\}$.
Since $X^TX$ is a square matrix, this means that $X^TX$ is invertible.

<br>
_______________________________________________________________________________________
<br><br>
</details>

<details>
<summary>We also need to proove that $\hat{\beta}$ is the only solution.</summary>
<br>
We proove that 
$$X^T\hat{r}=0 \implies \hat{\beta} = argmin_{\beta}\|Y-\beta X|$$ where $\hat{r}=Y-X\hat{\beta}$ is the vector of residual.
That is to say: the residual vector from $\hat{\beta}$ is orthogonal to every column of X.

Let's consider any vector $\gamma$, we have
$$Y-X\gamma = \hat{r}+X(\hat{\beta}-\gamma)$$
Since $\hat{r}$ is orthogonal to $X$, we get
$$\|Y-X\gamma\|^2=\|\hat{r}\|^2+\|X(\hat{\beta}-\gamma)\|^2$$
this is minimized when $X(\hat{\beta}-\gamma)=0$ and since $X$ has rank $p+1$, 
the unique solution is $\gamma=\hat{\beta}$

<br>
_______________________________________________________________________________________
<br>
</details>


## 1-5 Residuals and R2

<details>
<summary>We show why $R^2=\frac{\sum^n_{i=1}(\hat{Y}_i-\hat{\bar{Y}})^2}{\sum^n_{i=1}(Y_i-\bar{Y})^2}$</summary>
<br>
Let $\hat{r}=Y-\hat{Y}=Y-X\hat{\beta}$ be the vector of residuals.
We know that $\hat{r}$ is orthogonal to every column of $X$ (in particular to 1st column of $X$).
This means that the sum of residuals is $0$, thus
$$\sum^n_{i=1}\hat{r}_i=\sum^n_{i=1}(Y_i-\hat{Y}_i)=0 \implies \bar{Y}=\hat{\bar{Y}}$$

Since $\hat{r}$ is orthogonal to every column of $X$, we use the Pythagorean theorem to get:
$$\|Y\|^2=\|\hat{r}\|^2+\|\hat{Y}\|^2$$
using equality of sample means we get:
$$\|Y\|^2-n\bar{Y}^2=\|\hat{r}\|^2+\|\hat{r}\|^2-n\hat{\bar{Y}}^2$$
hence
$$R^2=\frac{\sum^n_{i=1}(\hat{Y}_i-\hat{\bar{Y}})^2}{\sum^n_{i=1}(Y_i-\bar{Y})^2}$$

_______________________________________________________________________________________
</details>
<br>

Note that both nominator and denominator are sample variance of $Y$ and $\hat{Y}$, so when $R^2$ is large, much of the outcome sample variance is "explained" by the fitted value, with $0 \leq R^2 \leq 1$. 

<script src="https://gist.github.com/Vincent-Maladiere/b3bd0d25d76896892fa681f9dd73057f.js"></script>

Let's plot the residuals for our model against fitted values $\hat{Y}_i$ (not the original outcomes $Y_i$)

<p align="center">
<img src="https://vincent-maladiere.github.io/images/residuals_vs_fitted.png" width="500" height="200"/>
</p>

<details>
<br>
<summary>Hat tip to <a href="https://emredjan.github.io/blog/2017/07/11/emulating-r-plots-in-python/">Emre Can</a> for his very neat implementation and overview of plotting techniques..</summary>
<script src="https://gist.github.com/Vincent-Maladiere/6fb9b7c2ae87822ddc68ca12125aa421.js"></script>
</details>

+ We might have non-linear relationships here because residuals are not equally spread around a horizontal line, this line has an angle.
+ In case of size effect (not enough data) or overfitting, our multiple-$R^2$ can be high even when the model fits poorly.
+ On the contrary, a model with a low $R^2$ can [still be useful](https://www.theanalysisfactor.com/small-r-squared/) to find a small relationship if the model is significant.

We assumed that we had $p<n$ and $X$ has *full rank* $p+1$. What happens this is not the case?
+ If $X$ doesn't have full rank, then $X^TX$ is not invertible, thus the optimal $\hat{\beta}$ minimizing the SSE is not unique (we have collinearity, the model is nonindentifiable).
+ If $p \approx n$, then the number of covariates is of a similar order to the number of observation (high dimensional regime).
+ If $p+1 \geq n$ we have enough degrees of freedom to perfectly fit the data (in general when $p \geq$ the model is non identifiable). Is this a good model? 


## 1-6 Interpreting regression coefficients

Suppose we completely believe our model. We might say:
+ A $1$ unit change in $X_{ij}$ is associated with a $\hat{\beta}_j$ change in $Y_i$.
+ Given $(X_{i1}, ..., X_{ip})$, we predict $Y_i$ will be $$\hat{\beta}_0 + \sum_j \hat{\beta}_j X_{ij}$$.

This section focus on helping you understand conditions under which these statements are ok, and when they aren't.

Recall `mom_work` is a canditional variable that ranges from $1$ to $4$. Does it make sense to build a model where:
`kid_score`$\approx \hat{\beta}_0+\hat{\beta}_1$ `mom_work` ? Not really.
> The regression model tries to estimate the conditional average of the outcome, given the covariates

We can get more intuition for the regression coefficients by looking at the case where there is only one continous covariate:
$$Y_i \approx \hat{\beta}_0+\hat{\beta}_1 X_i$$ (note we dropped the second index $j$ on $X_i$)

It can be shown that

$$\hat{\beta}_0=\bar{Y}-\hat{\beta}\bar{X};\;\hat{\beta}_1=\hat{\rho}\frac{\hat{\sigma}_Y}{\hat{\sigma}_X}$$

where $-1 \leq \hat{\rho} \leq 1$ is the sample correlation

$$\hat{\rho}=\frac{\frac{1}{n}\sum^n_{i=1}(X_i-\bar{X})(Y_i-\bar{Y})}{\hat{\sigma_X}\hat{\sigma_Y}}$$

Thus, if $X_i$ is $a.\hat{\sigma}_X$ larger than $\bar{X}$, then the fitted value $\hat{Y}_i$ will only be $\hat{\rho}.a.\hat{\sigma}_X$ larger than $\bar{Y}$.
On average, fitted values are closer to their mean than the covariates are to their mean (this is called mean reversion).

## 1-7 Beyond linearity

The linear regression model projects the outcomes into a hyperplane, determined by the covariates. This fit might have systematic problems because the relationship between Y and X is inherently nonlinear. 
Looking at residuals won't always suggest an issue, and therefore knowing the context is critical.

Our new [datasets](https://gist.github.com/seankross/a412dfbd88b3db70b74b) suggests that modeling our `mpg` as quadratic function of `hp` increase our $R^2$.
We now have

$$Y_i \approx \hat{\beta}_0 + \hat{\beta}_1 X_i + \hat{\beta}_2 X_i^2$$

+ Linear
<p align="center">
<img src="https://vincent-maladiere.github.io/images/linear.png" width="800" height="400"/>
</p>
$R^2$ value is $0.62$

+ Quadratic
<p align="center">
<img src="https://vincent-maladiere.github.io/images/quadratic.png" width="800" height="400"/>
</p>
$R^2$ value is $0.75$

<details>
<summary>See notebook here</summary>
<script src="https://gist.github.com/Vincent-Maladiere/b032af321fcceb36423685f4f6510274.js"></script>
</details>
<br>

Back with our kid.iq dataset, we find that

`mom_iq`$=0 \implies$ `kid_score` $\approx 25.73 + 0.56 .$`mom_iq`<br>
`mom_iq`$=1 \implies$ `kid_score` $\approx 31.68 + 0.56 .$`mom_iq`

so they have the same slope. However the plot below suggests higher slope when `mom_hs`$=0$.

<p align="center">
<img src="https://vincent-maladiere.github.io/images/need_interaction.png" width="400" height="200"/>
</p>

When changing the value of one covariate affects the coefficient of another, we need an interaction term in the model.
So our model with 2 covariates becomes 

$$Y_i \approx \hat{\beta}_0+\hat{\beta}_1 X_{i1}+\hat{\beta}_2 X_{i2}+\hat{\beta}_{1:2}X_{i1}X_{i2}$$

This time we have

`mom_iq`$=0 \implies$ `kid_score` $\approx -11.48 + 0.97 .$`mom_iq`<br>
`mom_iq`$=1 \implies$ `kid_score` $\approx 39.79 + 0.48 .$`mom_iq`

<p align="center">
<img src="https://vincent-maladiere.github.io/images/with_interaction.png" width="400" height="200"/>
</p>

You should try to include interaction terms with
+ A covariate has a large effect on the fitted value (high coefficient).
+ Covariates that describe groups of data (`mom_hs` or `mom_work` in this example).

## 1-8 Beyond minimizing SSE

Adding covariates to a model can only make $R^2$ increase, because each additional covariate 
carries even little "explanations to the variance of the outcome variable".

However, we sometimes want our regression to pick out coefficients that are "meaningful".
We achieve this by regularizing the objective function. Instead of minimizing sse, we 
minimize

+ Ridge regression

$$SSE + \lambda\sum^p_{j=1}\|\hat{\beta}_j\|^2$$

where $\lambda > 0$. It penalizes $\hat{\beta}$ vectors with "large" norms

+ Lasso

$$SSE + \lambda \sum^p_{j=1}\|\hat{\beta}_j\|$$

where $\lambda > 0$. In practice, the resulting coefficient vector will be "sparser"
than unregularized coefficient vector.

Regularized regression is often used for "kitchen sink" data analysis (including every covariate you can find, 
then run a regularized regression to pick out which covariates are important).

This raises the issue of overfitting and hasty generalizations, because the more independent 
variables are included in a regression, the greater the probability that one or more will 
be found to be significant while in fact having no causal effect on 
the outcome.

Also, linear regression can be very sensitive to outliers because SSE 
$(Y_i-\hat{Y}_i)^2$ is quadratic. Thus, small changes in $Y_i$ leads 
to large changes in SSE.

Thus, instead of SSE, one might minimize the sum of *absolute deviations*

$$\sum^n_{i=1}\|Y_i-\hat{Y}_i\|$$

Ridge and Lasso are also less vulnerable to outliers.

Very sweet tutorials on [Scikit Learn Doc](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)

## 1-9 Data transformations

+ Logarithmic transformations

Regression can sometimes leads to a model where $\hat{Y}_i$ is negative for some
$X_i$. This can be problematic for positive outcome, like a weight or a price.

So one approach is 

$$log Y_i \approx \hat{\beta}_0 + \sum^p_{j=1}\hat{\beta}_jX_{ij}$$

So

$$Y_i \approx e^{\hat{\beta}_0}\prod^p_{j=1}e^{\hat{\beta}_j X_{ij}}$$

See the housing unit vs income by county below

<p align="center">
<img src="https://vincent-maladiere.github.io/images/nolog.png" width="400" height="200"/>
</p>

which becomes

<p align="center">
<img src="https://vincent-maladiere.github.io/images/log.png" width="400" height="200"/>
</p>

Here $\hat{\beta}_1=1.14%$, so a $1$% higher median household is associated with
a $1.14$% higher number of housing units, in average.

+ Centering
  
We remove the mean

$$\tilde{X}_{ij}=X_{ij}-\bar{X}_j$$

so the model is now

$$Y_i \approx \hat{\beta}_0+\sum^p_{j=1}\hat{\beta}_j\tilde{X}_ij$$

+ Standardize 

$$\tilde{X}_{ij}=\frac{X_{ij}-\bar{X}_j}{\hat{\sigma}_j}$$

This gives all covariate a normalized dispersion.

## 1-10 Alternating Least Square

Let's broaden the scope of this article with an extension from another [Stanford course](http://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf).

We shed light now on the common use case of personalized recommendation of new products to users. We know the rates that users have given to certain items,
and our task is to predict their rating for the rest of these items.

A popular approach is to use Matrix Factorisation. We describe both $m$ users and $n$ products with vectors of size $k$ 

$$X=(x_{uj})_{1 \leq u \leq n,\; 1 \leq j \leq k} \\ Y=(y_{ij})_{1 \leq i \leq m,\; 1 \leq j \leq k}$$

To predict user $u$ rating for item $i$ we simply compute

$$r_{ui} \approx x^T_u y_i$$

So our goal is to estimate the complete rating matrix 

$$R=X^TY$$

We need to find optimal $X$ and $Y$ in order to minimizes least square error of the observed ratings

$$min_{XY}\sum_{r_{ui}observed}(r_{ui}-x^T_u y_i)^2 + \lambda(\sum_u\|x_u\|^2+\sum_i\|y_i\|^2)$$

Notice that this objective is non-convex because of the $x_u^T y_i$ term. 
In fact its NP-hard to optimize (the travelling salesman is another exemple of NP-hard problem). We can use gradiant descent as an approximation, 
but it turns out to be too slow. 

Instead, we use ALS: we fix $Y$ and optimize $X$, then fix $X$ and optimize $Y$, and repeat until convergence. 

<details>
<summary>Let's describe our ALS</summary>
<br>
$Initialize\;X, Y$.
<br>
$while\;not\;convergence:$
<br>
    $\;\;\;\;\;\;\;\;for\;u=1...n\;:$
    <br>
      $\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;x_u=(\sum_{r_{ui}\in r_{u*}}y_i y_i^T + \lambda I_k)^{-1} (\sum_{r_{ui} \in r_{u*}} r_{ui}y_i)$
<br>
    $\;\;\;\;\;\;\;\;for\;i=1...m\;:$
    <br>
      $\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;y_i=(\sum_{r_{ui}\in r_{*i}}x_u x_u^T + \lambda I_k)^{-1} (\sum_{r_{ui} \in r_{*i}} r_{ui}x_u)$
<br><br>
Updates will cost:
<ul>
<li>$O(n_u k^2 + k^3)$ for each $y_i$ (where $n_u$ are the number of items rated by user $u$)</li>
<li>$O(n_i k^2 + k^3)$ for each $x_u$ (where $n_i$ are the number of users that have rated item $i$)</li>
</ul>

_______________________________________________________________________________________
</details>
<br>

Now that we have $X$ and $Y$, we need to compute $R$. Simply predicting $r_{ui} \approx x_u^T y_i$ will cost $O(nmk)$, if we estimate every user-item pair.

While this migth be ok for small datasets, this is clearly not ok for larger ones, so we need instead Distributed ALS, which is detailed in the previously mentionned course.

<br>
# Up Next 
------

I hope that this summary gave you a nice overview about regression techniques and that this series has helped you so far.
If you didn't quench your regression thirst, have a look at [this state of this art on ALS](https://endymecy.gitbooks.io/spark-ml-source-analysis/content/%E6%8E%A8%E8%8D%90/papers/Large-scale%20Parallel%20Collaborative%20Filtering%20the%20Netflix%20Prize.pdf).
Next episode will cover prediction. Stay tuned!





