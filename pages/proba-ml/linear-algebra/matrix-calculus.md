# 7.8 Matrix calculus

### 7.8.1 Derivatives

For a scalar-argument function $f:\mathbb{R}\rightarrow \mathbb{R}$, we define its derivative at a point $x$ the quantity:

$$
f'(x)=\lim_{h\rightarrow 0}\frac{f(x+h)-f(x)}{h}
$$

### 7.8.2 Gradients

We can extend this definition to vector-argument function $f:\mathbb{R}^n\rightarrow \mathbb{R}$, by defining the partial derivative of $f$ w.r.t $i:$

$$
\frac{\partial f}{\partial x_i}=\lim_{h\rightarrow 0} \frac{f(\bold{x}+h\bold{e}_i)-f(\bold{x})}{h}
$$

where $\bold{e}_i$ is the $i$th unit vector.

The gradient of a function is its vector of partial derivatives:

$$
g=\frac{\partial f}{\partial \bold{x}}=\nabla f
=
\begin{pmatrix}
\frac{\partial f}{\partial x_1} \\ \vdots \\
\frac{\partial f}{\partial x_n}
\end{pmatrix}
$$

Where the operator $\nabla$ maps a function $f:\mathbb{R}^n\rightarrow \mathbb{R}$ to another function $g:\mathbb{R}^n\rightarrow \mathbb{R}^n.$

The point at which the gradient is evaluated is noted:

$$
g(\bold{x}^*)\triangleq \frac{\partial f}{\partial \bold{x}}\Bigg |_{\bold{x}^*}
$$

### 7.8.3 Directional derivative

The directional derivative measure how much $f:\mathbb{R}^n\rightarrow\mathbb{R}$ changes along a direction in space $\bold{v}$:

$$
D_\bold{v}f(\bold{x})=\lim_{h\rightarrow 0}\frac{f(\bold{x}+\bold{v}h)-f(\bold{x})}{h}
$$

Note that:

$$
D_{\bold{v}}f(\bold{x})=\nabla f(\bold{x}).\bold{v}
$$

### 7.8.4 Total derivative

Suppose the function has the form $f(t,x(t),y(t))$, we define the total **derivative** w.r.t $t$ as:

$$
\frac{df}{dt}=\frac{\partial f}{\partial t}+\frac{\partial f}{\partial x}\frac{\partial x}{dt}+\frac{\partial f}{\partial y}\frac{\partial y}{dt}
$$

Multiplying by $dt$, we get the **total differential**:

$$
df=\frac{\partial f}{\partial t}dt+\frac{\partial f}{\partial x}\partial x+\frac{\partial f}{\partial y}\partial y
$$

This represents how much $f$ changes when we change $t$.

### 7.8.5 Jacobian

Consider $\bold{f}:\mathbb{R}^n\rightarrow \mathbb{R}^m$. The Jacobian of this function is an $m \times n$ matrix of partial derivatives:

$$
J_\bold{f}(\bold{x})=\frac{\partial \bold{f}}{\partial \bold{x}^\top}\triangleq
\begin{pmatrix}
\frac{\partial f_1}{\partial x_1}&\dots& \frac{\partial f_1}{\partial x_n}\\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1}&\dots& \frac{\partial f_m}{\partial x_n}
\end{pmatrix}=
\begin{pmatrix}
\nabla f_1(\bold{x})^\top \\ \vdots \\ \nabla f_m(\bold{x})^\top
\end{pmatrix}
$$

**7.8.5.1 Vector Product**

The Jacobian vector product (JVP) is right multiplying $J\in\mathbb{R}^{m\times n}$ by $v\in\mathbb{R^n}$:

$$
J_\bold{f}(\bold{x}).\bold{v}=\begin{pmatrix}
\nabla f_1(\bold{x})^\top.\bold{v} \\ \vdots \\ \nabla f_m(\bold{x})^\top.\bold{v}
\end{pmatrix}
$$

Similarly, the vector Jacobian product (VJP) is left multiplying by $u\in\mathbb{R}^m$:

$$
\bold{u}^\top J_{\bold{f}}(\bold{x})=
\bold{u}^\top\begin{pmatrix}\frac{\partial \bold{f}}{\partial x_1}&\dots & \frac{\partial \bold{f}}{\partial x_1}
\end{pmatrix}=
\begin{pmatrix}\bold{u}^\top\frac{\partial \bold{f}}{\partial x_1}&\dots & \frac{\partial \bold{f}}{\partial x_1}
\end{pmatrix}
$$

**7.8.5.2 Composition of feature**

The Jacobian of the composition of two features $h(\bold{x})=g(f(\bold{x}))$ is obtained with the chain rule:

$$
J_h(\bold{x})=J_g(f(\bold{x}))J_f(\bold{x})
$$

Let $f:\mathbb{R}\rightarrow \mathbb{R}^2$ and $g:\mathbb{R}^2\rightarrow \mathbb{R}^2$, we have:

$$
\frac{\partial \bold{g}}{\partial x}=\frac{\partial \bold{g}}{\partial \bold{f}}\frac{\partial \bold{f}}{\partial x}=
\begin{pmatrix}
\frac{\partial g_1}{\partial f_1} & \frac{\partial g_1}{\partial f_2} \\
\frac{\partial g_2}{\partial f_1} & \frac{\partial g_2}{\partial f_2} 
\end{pmatrix}
\begin{pmatrix}
\frac{\partial f_1}{\partial x} \\ 
\frac{\partial f_2}{\partial x} 
\end{pmatrix}
$$

### 7.8.6 Hessian

For $f:\mathbb{R}^n\rightarrow\mathbb{R}$ that is twice differentiable, the Hessian is the $n\times n$  symmetric matrix of second partial derivatives:

$$
H_{f}=\frac{\partial^2 f}{\partial \bold{x}^2}=
\begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \dots & 
\frac{\partial^2 f}{\partial x_1\partial x_n} \\
 & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \dots & \frac{\partial^2 f}{\partial x_n^2}

\end{pmatrix}
$$

The Hessian is the Jacobian of the gradient.

### 7.8.7 Gradients of commonly used functions

**7.8.7.2 Functions that map vectors to scalars**

$$
\begin{align}
\frac{\partial a^\top x}{\partial x}&=a \\
\frac{\partial b^\top A x}{\partial x}&=A^\top b \\
\frac{\partial x^\top Ax}{\partial x}&=(A+A^\top)x
\end{align}
$$

**7.8.7.3 Functions that map matrices to scalar**

Quadratic forms:

$$
\begin{align}
\frac{\partial a^\top Xb}{\partial X}&=ab^\top \\
\frac{\partial a^\top X^\top b}{\partial X}&=ba^\top 
\end{align}
$$

Traces:

$$
\begin{align}
\frac{\partial}{\partial X}\mathrm{tr}(AXB)&=A^\top B^\top \\
\frac{\partial}{\partial X}\mathrm{tr}(X^\top A) &=A \\
\frac{\partial}{\partial X}\mathrm{tr}(X^{-1}A)&= -X^{-T}AX^{-T} \\
\frac{\partial }{\partial X} \mathrm{tr}(X^\top AX)&=(A+A^\top)X
\end{align}
$$

Determinants:

$$
\begin{align}
\frac{\partial}{\partial X}\mathrm{det}(AXB)&=\mathrm{det}(AXB)X^{-T} \\
\frac{\partial}{\partial X}\ln(\mathrm{det(X))}&=X^{-T}
\end{align} 
$$