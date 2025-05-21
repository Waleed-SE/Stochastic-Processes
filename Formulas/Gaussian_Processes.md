# Gaussian Processes Formulas

## 1. Definition

A Gaussian Process (GP) is a collection of random variables, any finite number of which have a joint Gaussian distribution. A GP is fully specified by:

1. Mean function: $m(x) = \mathbb{E}[f(x)]$
2. Covariance (kernel) function: $k(x, x') = \mathbb{E}[(f(x) - m(x))(f(x') - m(x'))]$

## 2. Common Kernel Functions

### Squared Exponential (RBF) Kernel

$$
k_{SE}(x, x') = \sigma_f^2 \exp\left(-\frac{||x - x'||^2}{2l^2}\right)
$$

### Matérn Kernel (with $\nu=3/2$)

$$
k_{M}(x, x') = \sigma_f^2 \left(1 + \frac{\sqrt{3}|x - x'|}{l}\right) \exp\left(-\frac{\sqrt{3}|x - x'|}{l}\right)
$$

### Periodic Kernel

$$
k_{P}(x, x') = \sigma_f^2 \exp\left(-\frac{2\sin^2(\pi|x - x'|/p)}{l^2}\right)
$$

Where:

- $\sigma_f^2$ = Signal variance
- $l$ = Length scale parameter
- $p$ = Period length (for periodic kernel)

## 3. Gaussian Process Regression (GPR)

### Posterior Distribution

Given training data $X$, $y$ and test points $X_*$, the posterior distribution is:

Mean:

$$
\mu_* = K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1}y
$$

Covariance:

$$
\Sigma_* = K(X_*, X_*) - K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1}K(X, X_*)
$$

Where:

- $K(X, X')$ = Covariance matrix between points $X$ and $X'$
- $\sigma_n^2$ = Noise variance

### Predictive Distribution

For a test point $x_*$, the predictive distribution is:

$$
p(f_* | X, y, x_*) = \mathcal{N}(f_* | \mu_*, \sigma_*^2)
$$

Where:

- $\mu_* = k_*^T (K + \sigma_n^2 I)^{-1} y$
- $\sigma_*^2 = k_{**} - k_*^T (K + \sigma_n^2 I)^{-1} k_*$
- $k_* = k(X, x_*)$ is the vector of covariances between test point and training points
- $k_{**} = k(x_*, x_*)$ is the prior variance at the test point

## 4. Log Marginal Likelihood

The log marginal likelihood for hyperparameter optimization:

$$
\log p(y|X) = -\frac{1}{2} y^T (K + \sigma_n^2 I)^{-1} y - \frac{1}{2} \log |K + \sigma_n^2 I| - \frac{n}{2} \log 2\pi
$$

## 5. Confidence Intervals

The 95% confidence interval for predictions at a test point $x_*$ is:

$$
\mu_* \pm 1.96 \sqrt{\sigma_*^2}
$$
