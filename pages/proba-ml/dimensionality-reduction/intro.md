# 20. Dimensionality reduction

A common form of unsupervised learning is **dimensionality reduction**, in which we learn a mapping from the high-dimensional visible space $\bold{x}\in\R^D$ to a low-dimensional latent space $\bold{z}\in \R^L$.

This mapping can either be parametric $\bold{z}=f(\bold{x};\theta)$, which can be applied to any input, or it can be a nonparametric mapping where we compute an **embeddings $\bold{z}_n$** for each input $\bold{x}_n$ in the dataset, but not for any other point.

The former is mostly used for data visualization whereas the latter can be used as a preprocessing step for other kind of learning algorithms. For example, we can produce an embedding by mapping $\bold{x}$ to $\bold{z}$, and then learn a simple linear classifier by mapping $\bold{z}$ to $y$.