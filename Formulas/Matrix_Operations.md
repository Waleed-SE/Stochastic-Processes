# Matrix Operations for Stochastic Processes

## 1. Matrix Power Method for Steady State

For a Markov chain with transition matrix $P$, the steady state distribution $\pi$ can be found by:

$$
\pi = \lim_{n \to \infty} \pi_0 P^n
$$

Where $\pi_0$ is any initial distribution.

## 2. Eigenvalue Decomposition for Transition Matrices

If $P$ has eigenvalue decomposition $P = VDV^{-1}$, then:

$$
P^n = VD^nV^{-1}
$$

For a transition matrix, the largest eigenvalue is 1, and the corresponding eigenvector (normalized) is the stationary distribution.

## 3. Mean First Passage Time Matrix

For a Markov chain with transition matrix $P$ and stationary distribution $\pi$, the mean first passage time matrix $M$ can be computed as:

$$
M_{ij} = 1 + \sum_{k \neq j} P_{ik} M_{kj} \quad \text{for } i \neq j
$$

$$
M_{ii} = \frac{1}{\pi_i}
$$

## 4. Fundamental Matrix

For an absorbing Markov chain with transition matrix in canonical form:

$$
P = \begin{pmatrix}
Q & R \\
0 & I
\end{pmatrix}
$$

The fundamental matrix is:

$$
N = (I - Q)^{-1}
$$

## 5. Absorption Probabilities

Using the fundamental matrix $N$, the probability of absorption in each absorbing state is:

$$
B = NR
$$

## 6. Expected Time to Absorption

The expected number of steps before absorption when starting in transient state $i$ is:

$$
t_i = \sum_{j} N_{ij}
$$

## 7. Variance of Time to Absorption

$$
Var[T_i] = \sum_{j} N_{ij}(2N_{ij} - 1) - \left(\sum_{j} N_{ij}\right)^2
$$

## 8. Steady-State Calculation for Birth-Death Processes

For a birth-death process with birth rates $\{\lambda_i\}$ and death rates $\{\mu_i\}$, the steady-state probabilities are:

$$
\pi_i = \pi_0 \prod_{j=1}^{i} \frac{\lambda_{j-1}}{\mu_j}
$$

Where $\pi_0$ is determined by normalization: $\sum_{i=0}^{\infty} \pi_i = 1$
