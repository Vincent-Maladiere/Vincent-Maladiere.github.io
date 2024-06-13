# 13.1 Introduction

Linear models make the strong assumption of linear relationships between inputs and outputs.

A simple way of increasing the flexibility of linear models is to perform feature transformation by replacing $\bold{x}$ by $\Phi(\bold{x})$. For example, polynomial extension in 1d uses $\Phi(x)=[1,x,x^2,\dots]$

The model now becomes:

$$
f(\bold{x};\theta)=W\phi(\bold{x})+\bold{b}
$$

This is still linear in the parameters $(W,\bold{b})$, which makes the fitting easy since the NLL is convex, but specifying the feature transform manually is very limiting.

A natural solution is to endow the feature extractor with its own parameters:

$$
f(\bold{x};\theta)=W\phi(\bold{x};\theta_2)+\bold{b}
$$

where $\theta=(\theta_1,\theta_2)$.

We can repeat this process recursively to create more complex patterns:

$$
f(\bold{x},\theta)=f_{L}(f_{L-1}(\dots(f_1(\bold{x}))\dots)
$$

where $f_\ell(\bold{x})=f(\bold{x};\theta_\ell)$ is the function at level $\ell$.

**Deep neural networks (DNN)** encompass a large family of models in which we compose differentiable functions in DAG (direct acyclic graph).

The $L$ layers above are the simplest example where the DAG is a chain: this is called **Feed Foward Neural Net (FFNN)** or **multilayer perceptron (MLP)**.

An MLP assumes the input is a fixed-dimensional vector $\bold{x}\in\mathbb{R}^D$, often called **tabular data** or **structured data**, since the data is stored into a $N\times D$ design matrix, in which each column has a specific meaning (age, length, etc).

Other kinds of DNNs are more suited to **unstructured data** (text, image), where each element (pixel or word) is meaningless alone. **Convolutional neural networks (CNN)** are historically performant on images, **transformers** on sequences, and **graph neural networks (GNN)** on graphs.