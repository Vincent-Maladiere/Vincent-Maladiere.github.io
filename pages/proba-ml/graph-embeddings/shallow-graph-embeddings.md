# 23.3 Shallow graph embeddings

Shallow embedding methods are transductive graph embedding methods, where the encoder maps categorical nodes IDs onto a Euclidean space through an embedding matrix.

Each node $v_i\in V$ has a corresponding embedding $Z_i\in\R^L$ and the shallow encoder function is:

$$
Z=\mathrm{ENC}(\Theta^E)\triangleq \Theta^E\in \R^{N\times L}
$$

The embedding dictionary $Z$ is directly learned as model parameters.

In the unsupervised case, the embeddings are optimized to recover information about the input graph (e.g. the adjacency matrix $W$). This is similar to dimension reduction methods like PCA, but for graphs.

In the supervised case, the embedding are optimized to predict some labels, for nodes, edges and/or the whole graph.

### 23.3.1 Unsupervised embeddings

In the unsupervised case, we consider two main types of shallow embedding methods: distance-based and outer product-based.

**Distance-based** methods optimize embeddings such that nodes $i$ and $j$ which are close on the graph (as measured by some graph distance function) are embedded in $Z$ such that the pairwise distance function $d_2(Z_i,Z_j)$ is small.

$d_2$ can be customized, which lead to Euclidean or non-Euclidean embeddings. The decoder outputs:

$$
\hat{W}=\mathrm{DEC}(Z;\Theta^D)
$$

with $\hat{W}_{ij}=d_2(Z_i,Z_j)$.

**Pairwise dot-products** compute node similarities. The decoder network can be written as:

$$
\hat{W}=\mathrm{DEC}(Z;\Theta^D)=ZZ^\top
$$

In both cases, embeddings are learned by minimizing the graph regularization loss:

$$
\mathcal{L}_\mathrm{G,RECON}(W,\hat{W};\Theta)=d_1(s(W),\hat{W})
$$

where $s(W)$ is an optional transformation and $d_1$ is a pairwise distance function between matrix, which don’t need to be the same form as $d_2$.

### 23.3.2 Distance-based: Euclidean methods

Distance-based methods minimize Euclidean distance between similar (connected) nodes.

**Multi-dimensional scaling (MDS)** is equivalent to setting $s(W)$  to some distance matrix measuring the dissimilarity between nodes (e.g. proportional to pairwise shortest distance) and then defining:

$$
d_1(s(W),\hat{W})=\sum_{i,j}(s(W)_{ij}-\hat{W}_{ij})^2=||s(W)-\hat{W}||^2_F
$$

where $\hat{W}_{ij}=d_2(Z_i,Z_j)=||Z_i-Z_j||$

**Laplacian eigenmaps** learn embeddings by solving the generalized eigenvector problem:

$$
\hat{Z}=\min_{Z} \mathrm{tr}(Z^\top LZ)\quad s.t.\quad Z^\top DZ =I\quad \mathrm{and}\quad Z^\top D 1=0
$$

where $L=D-W$ is the graph Laplacian, and $D$ is the diagonal matrix of the row-wise sum of $W$.

The first constraint removes an arbitrary scaling factor in the embedding and the second remove trivial solutions corresponding to the constant eigenvector (with eigenvalue zero for connected graphs).

Further, note that:

$$
\mathrm{tr}(Z^\top LZ)=\frac{1}{2}\sum_{i,j}W_{ij}||Z_i-Z_j||^2_2
$$

where $Z_i$ is the $i$’th row of $Z$. Therefore the minimization of the objective can be written as a graph reconstruction term:

$$
d_1(s(W),\hat{W})=\sum_{ij}W_{ij}\hat{W}_{ij}
$$

where $s(W)=W$

### 23.3.3 Distance-based: non-Euclidean methods

So far, we have assumed method creating embeddings in the Euclidean space.

However, hyperbolic geometry is ideal for embedding trees and graph exhibiting a hierarchical structure.

Embedding of hierarchical graphs can be learn using the **Pointcaré model** of hyperbolic space. We only need to change $d_2$:

$$
d_2(Z_i,Z_j)=\mathrm{arcosh}(1+2\frac{||Z_i-Z_j||^2_2}{||1-Z_i||^2_2||1-Z_j||^2_2})
$$

The optimization then learns embedding that minimize the distance between connected node, and maximize the distance between disconnected nodes:

$$
d_1(W,\hat{W})=\sum_{i,j}W_{ij}\log \frac{e^{-\hat{W}_{ij}}}{\sum_{k:W_{ik}=0}e^{-\hat{W}_{ik}}}
$$

where the denominator is approximated by negative sampling.

Note that since the hyperbolic space has a manifold structure, we need to make sure that the embedding stay remain on the manifold (by using Riemannian optimization techniques).

It has been shown that variant using **Lorentz model** of hyperbolic space provides better numerical stability.

### 23.3.5 Outer product-based: Skip-gram methods

Skip-gram word embeddings are optimized to predict context words given a center word. Given a sequence of words $(w_0,\dots,w_T)$, skip-gram will optimize:

$$
\mathcal{L}=-\sum_{-K\leq i \leq K,i\neq 0}\log p(w_{k-i}|w_k)
$$

for each target word $w_k$.

This idea has been leveraged for graph embedding in the **DeepWalk** framework, since the author have proven that the frequency statistics induced by random walk in the graph is similar to that of words in natural language.

Deepwalk train node embeddings to maximize the probability of context nodes for each center node. The context are the nodes reached during a single random walk.

Each random walk starts with a nodes $v_{i_1}\in V$ and repeatedly samples the next node uniformly at random: $v_{i_{j+1}}\in \{v\in V:(v_{i_j},v)\in E\}$. The walk length is a hyperparameter.

All generated random-walks (sentences) can then be encoded by a sequence model. This has been implemented by **node2vec**.

It is common to underlying representation to use two distinct representation for each node: one for when the node is a center and one for when it is in the context.

To present DeepWalk on the GraphEDM framework, we can set:

$$
s(W)=\mathbb{E}_q[(D^{-1}W)^q]\;\mathrm{with}\;q\sim P(Q)=\mathrm{Cat}(1,\dots,T_{\max})
$$

Training DeeWalk is equivalent to minimizing:

$$
\mathcal{L}_\mathrm{G,RECON}(W,\hat{W};\Theta)=\sum_i\log \sum_j \exp(\hat{W}_{ij})-\sum_{v_i\in V,v_j\in V}s(W)_{ij}\hat{W}_{ij}
$$

where $\hat{W}=ZZ^\top$  and the left term can be approximated in $O(N)$ time by hierarchical softmax.

Skip-gram methods can be viewed as implicit matrix factorization, which can inherit benefits of efficient sparse matrix operatioons.

### 23.3.6 Supervised embeddings

In many applications, we have labeled data in addition to node features and graph structure.

While we can tackle supervised problem by first learning unsupervised embeddings and apply them to a supervised task, this is not the recommended workflow. Unsupervised node embeddings might not preserve important graph properties (e.g., node neighborhoods) that are most useful for a downstream task.

A number of methods combine these two steps, like **label propagation (LP)**, a very popular algorithm for semi-supervised node classification. The encoder is a shallow model represented by a lookup table $Z$.

LP use the label space to represent the node embedding directly (the decoder is the identity function):

$$
\hat{y}^N=\mathrm{DEC}(Z;\Theta^C)=Z
$$

Laplacian eigenmaps are used in the regularization to enforce a smoothness on labels, which represents the assumption that neighbor nodes should have the same labels:

$$
\mathcal{L}_\mathrm{G,RECON}(W,\hat{W};\Theta)=\sum_{ij}W_{ij}||y_i^N-\hat{y}_j^N||^2_2
$$

LP minimizes this loss on the space of functions that take fixed values on label nodes (i.e. $\{\hat{y}_i^N=y_j^N:\forall i \;|\;v_i \in V_L\}$ using an iterative algorithm that updates unlabeled node’s label distribution via the weighted average of its neighbors’ labels.

**Label spreading (LS)** is a variant of label propagation which minimizes the following function:

$$
\mathcal{L}_{\mathrm{G,RECON}}(W,\hat{W};\Theta)=\sum_{ij}W_{ij}||\frac{y_i^N}{\sqrt{D_i}}-\frac{\hat{y}_j^N}{\sqrt{D_j}}||^2_2
$$

where $D_i=\sum_{j}W_{ij}$ is the degree of node $v_i$.

In both methods, the supervised loss is the distance between the predicted labels and the ground truth (one-hot vectors)

$$
\mathcal{L}^N_{\mathrm{SUP}}(y_N,\hat{y}_N;\Theta)=\sum_{i|v_i\in V_L}||y^N_i-\hat{y}^N_j||^2_2
$$

These methods are expected to work well with consistent graphs, that is graph where node proximity is positively correlated with label similarity.