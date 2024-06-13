# 17. Kernel Methods

In this chapter, we consider **nonparametric methods** for classification and regression. These methods don’t assume fixed parametric form for the prediction function, but try to estimate the function itself (rather than its parameters) directly from the data.

They key idea is to memorize a training dataset $\mathcal{D}=\{(\bold{x}_n,y_n)\}_{i=1:N}$ and compare a new input $\bold{x}_*$ to each of the training points $\{\bold{x}_n\}$. We can then predict that $f(\bold{x}_*)$ is a weighted combinations of the $\{y_n=f(\bold{x}_n)\}$ values.

In section 17.1, we explain that the similarity between $\bold{x}_*$ and each $\bold{x}_n$ is computed using a kernel function, $\mathcal{K}(\bold{x}_n,\bold{x}_*)\geq 0$. This approach is similar to RBF network, except we use the datapoints $\bold{x}_n$ themselves as anchors rather than learning centroids $\mu_n$.

In section 17.2, we discuss an approach called Gaussian processes, in which the kernel define a *prior over functions*, which we can update given data to get a *posterior over functions.*

Alternatively, we can use the kernel with a Support Vector Machines to compute the MAP estimate of the function, as explained in section 17.3.