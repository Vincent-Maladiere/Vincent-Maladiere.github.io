---
title: 'Small Data [2/4] - Prediction'
date: 2020-01-01
permalink: /posts/2020/01/small-data-prediction/
tags:
  - machine learning
  - prediction
---

[Work in progress]

# Small Intro
---

Hello reader, this article is the short version of a [Stanford course](http://web.stanford.edu/~rjohari/teaching/notes/). 

This is second article of the small data series. I wrote it after an extensive reading of this course so that you don't have to.
This series aims at mastering the skills that will help you for "small" data analysis and thus for any data analysis.

Note that all the code was in R langage in the original papers and that I converted it into Python.

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

> Prediction relies on correlation, not causation. 

How to know whether better education leads to higher earnings, or if people with 
higher earnings have chosen to acquire more schooling?

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

If we have enough data, we can build a predictive model as:
1. Split data into 3 groups: training, validation, and test
2. Use training data to fit different models $\hat{f}$'s
3. Use validation data to estimate generalization error of different model, 
  and select the best one

  <details>
  <br>
  <ul>
    <li>Suppose samples $(\tilde{X}_1, \tilde{Y}_1), ..., (\tilde{X}_k, \tilde{Y}_k))$ 
    in the validation set</li>
    <li>For each fitted model \hat{f}, estimate the generalization error as</li>
    $$\frac{1}{k}\sum^k_{i=1}(\tilde{Y}_i-\hat{f}(\tilde{X_i}))^2$$
    <li>We choose the model with the lowest generalization error</li>
  </ul>
  ______________________________________________________________
  </details>
  <br>
4. Use test data to assess performance of the chosen model




# Up Next 
------

