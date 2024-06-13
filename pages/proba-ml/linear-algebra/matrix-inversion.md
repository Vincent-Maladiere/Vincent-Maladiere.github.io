# 7.3 Matrix inversion

### 7.3.1 Square matrix

The inverse of a square matrix $A \in \mathbb{R}^n$ is $A^{-1}$. It is the unique matrix such that:

$$
A A^{-1}=I=A^{-1}A
$$

$A^{-1}$ is only defined if $\mathrm{det}(A)\neq 0$, that is if $A$ is not singular.

Some properties:

$$
\begin{align}(A^{-1})^{-1}&=A \\
(AB)^{-1}&=B^{-1}A^{-1} \\
(A^{-1})^\top &= (A^\top)^{-1}\triangleq A^{-\top}
\end{align}
$$

For a simple 2x2 matrix, we have:

$$
A=\begin{bmatrix}a & b \\c &d\end{bmatrix},\;\;A^{-1}=\frac{1}{|A|}\begin{bmatrix}d & -b \\-c &a\end{bmatrix}
$$

For a block diagonal matrix, we invert each block separately:

$$
\begin{bmatrix}A & 0 \\0 &B\end{bmatrix}^{-1}=\begin{bmatrix}A^{—1} & 0 \\0 &B^{—1}\end{bmatrix}
$$