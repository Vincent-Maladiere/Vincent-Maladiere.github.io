---
title: 'Small Data [2/4] - Prediction'
date: 2020-01-01
permalink: /posts/2020/01/small-data-prediction/
tags:
  - machine learning
  - prediction
---

[Work in progress]

# Small Intro & Aknowledgment
---

Hello reader, this article is a short version of the MS&E 226 [Stanford course](http://web.stanford.edu/~rjohari/teaching/notes/)
of Ramesh Johari, who kindly let me use his materials, so all credits go to him.

This is the second part of the small data series. I wrote it after an extensive reading of this course so that you don't have to.
This series aims at mastering the skills that will help you for "small" data analysis and thus for any data analysis.

Note that all the code was written in R langage in the original papers and that I converted it into Python.


# 2 Prediction
---

## 2-1 Generalization

The observed data we have, ${\bf X}$ and ${\bf Y}$, are our sample from some population. 
We try to understand the broader population through a smaller sample.

We model the population using probability:
+ There is a probability distribution of $X=(X_1, ..., X_p)$ in the population
+ $Y$ has a conditional probability distribution given $vec{X}$

<details>
<summary>Here is an example with linear population model</summary>
<br>

$$Y=\beta_0 + \beta_1X_1 + ... + \beta_pX_p + \epsilon$$

where $\vec{X}$ is jointly multivariate normal (each $X_i$ follows a normal distribution), and
$\epsilon \sim N(0, \sigma^2)$ is independent of $\vec{X}$.

Suppose in a population that father's heights are normally distributed with mean $69$ inches
and variance $4$ inches.
If a father has height $X=x$, his child's height is normally distributed with mean $40+0.4.x$
and variance 3 inches.

Then the population model is that

$$Y=40+0.4.X + \epsilon$$

where $X \sim N(69, 4)$, $\epsilon \sim N(0, 3)$ and $X$ and $\epsilon$ are independant.
______________________________________________________________
</details>
<br>

To make generalizations, we first build a fitted model: a function $\hat{f}$ that uses
${\bf X}$ and ${\bf Y}$ to capture relationships between $\vec{X}$ and $Y$:

$$Y \approx \hat{f}(\vec{X})$$

We want to make statements using $\hat{f}$ of:
+ Prediction: Given a new $\vec{X}$ observed, what is our best guess of $Y$?
E.g. will this customer purchase or not? How much will he spend?
+ Inference: Describing the joint distribution of $\vec{X}$ and $Y$, i.e. 
interpreting the structure of $\hat{f}$,
e.g. which marketing campaign is most responsible for the customer spend?

However, how to know whether better education leads to higher earnings, or if people with 
higher earnings have chosen to acquire more schooling? We can't, because

> Prediction relies on correlation, not causation. 


## 2-2 Prediction

How to measure prediction error?

+ In regression, $Y$ is a continuous variable,
error measure includes 
  + Squared error $(Y-\hat{f}(\vec{X}))^2$
  + Absolute deviation $\|Y-\hat{f}(\vec{X})\|$
+ In classification, $Y$ is a categorical variable,
error can be defined by $0-1$ loss: error is $1$ if $Y \neq \hat{f}(\vec{X})$ and $0$ 
otherwise.

Let's focus on regression with squared error as our measure of prediction error.
We are given ${\bf X}$ and ${\bf Y}$, and we want to minimize the test error, i.e. the prediction error on 
new data:

$$E[(Y-\hat{f}(\vec{X}))^2][{\bf X}, {\bf Y}]$$

As the data is given, the randomness is in the new sample $\vec{X}$ and $Y$.

Now, if we have enough data, we can build a predictive model:
1. Split data into 3 groups: training, validation, and test
2. Use training data to fit different models $\hat{f}$'s
3. Use validation data to estimate generalization error of different model,
and select model with the lowest error.
4. Use test data to assess performance of the chosen model.

Note that the validation error of the best model is often lower the test error. 
It means that your model looks more efficient than it actually is, which may lead to overfitting.

However, how should we estimate prediction error if we don't have enough data? 
We have 2 approches:
+ Cross validation
+ Model scores

## 2-3 Cross validation

Cross validation is a simple and widely used technique, whose core idea follows the 
train-test-valid paradigm, with a twist:
+ Train model on a subset of the data, and test on the remaining data
+ Repeat with $K$ different subsets of the data

![cross_validation](https://vincent-maladiere.github.io/images/cross_validation.png)

<details>
<summary>Let's describe the algorithm</summary>
<ul>
<li>Divide data into K equal groups, called folds. $A_k$ is the set of validation $(Y_i, {\bf X_i})$ of the k'th fold</li>
<li>For $k=1, ..., K$, we fit the model $\hat{f}_k$ with all points except the ones from $A_k$</li>
<li>We finaly compute the average of the test error of each model:
$$Err_{CV}=\frac{1}{K}\sum^K_{k=1}\bigg(\frac{1}{n/K}\sum_{i \in A_k}(Y_i-\hat{f}_k({\bf X}_i))^2\bigg)$$</li>
where $n$ is the number of all data points. 
</ul>

How to choose K?
<br>
<ul>
<li>When $K=n$: the training set for each $\hat{f}_k$ is almost the entire training data.
Therefore, $Err_{CV}$ will be nearly unbiased as an estimate of $Err$, 
  and will be very sensitive to the training data.</li>
<li>When $K<<n$, fewer train data are used, so $\hat{f}_k$ has higher generalization error
and are less correlated with each other.
So, $Err_{CV}$ will overestimate $Err$, and will be less sensitive to training data.</li>
</ul>
________________________________________________
<br>
</details>
<br>

Leave-one-out (LOO) CV corresponds to the case $K=n$. This is a useful computational trick
for linear models fitted by OLS: there is no need to refit the model.

<details>
<summary>Let's see the theorem</summary>
Previously in regression, we saw that 
$${\bf \hat{Y}} = {\bf X}\hat{\beta} = {\bf X}({\bf X}^T{\bf X})^{-1}{\bf X}^T{\bf Y} = {\bf H}{\bf Y}$$
Here, ${\bf \hat{Y}}$ is the fitted values under OLS with the full training data.
We have:

$$Err_{LOOCV} = \frac{1}{n}\sum^n_{i=1}\bigg(\frac{Y_i-\hat{Y}_i}{1-H_{ii}}\bigg)^2$$
Observations with $H_{ii}$ close to $1$ have a big impact on the fit and generalization error
________________________________________________
<br>
</details>

## 2-4 Model Scores

<br>
# Up Next 
------

