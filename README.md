# Stochastic Processes and Statistical Modeling

This repository contains implementations of various stochastic processes and statistical models in R, using only built-in functions. The code examples cover topics relevant to probability theory, Markov chains, Poisson processes, queueing theory, and Gaussian processes.

## Table of Contents

- [Probability Theory](#probability-theory)
- [Stochastic Processes](#stochastic-processes)
- [Discrete Markov Chains](#discrete-markov-chains)
- [Hidden Markov Models](#hidden-markov-models)
- [Poisson Processes](#poisson-processes)
- [Queueing Theory](#queueing-theory)
- [Gaussian Processes](#gaussian-processes)
- [Implementation Files](#implementation-files)

## Probability Theory

### Basic Laws of Probability

1. **Addition Law (OR)**:

   - For disjoint events: $P(A \cup B) = P(A) + P(B)$
   - For any events: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$

2. **Multiplication Law (AND)**:

   - General case: $P(A \cap B) = P(A) \cdot P(B|A)$
   - Independent events: $P(A \cap B) = P(A) \cdot P(B)$

3. **Conditional Probability**:

   - $P(A|B) = \frac{P(A \cap B)}{P(B)}$, where $P(B) > 0$

4. **Bayes' Theorem**:
   - $P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$
   - $P(A_i|B) = \frac{P(B|A_i) \cdot P(A_i)}{\sum_{j} P(B|A_j) \cdot P(A_j)}$

## Stochastic Processes

### Definition and Classification

A stochastic process $\{X(t), t \in T\}$ is a collection of random variables indexed by parameter $t$ (often time).

**Classification**:

- **State Space**: Discrete vs. Continuous
- **Parameter Space (Time)**: Discrete vs. Continuous
- **Memory**: Memoryless (Markov) vs. with Memory

### Key Properties

1. **Trajectory**: A single realization of the process
2. **Stationarity**:
   - Strict stationarity: $P(X_{t_1}, X_{t_2}, ..., X_{t_n}) = P(X_{t_1+h}, X_{t_2+h}, ..., X_{t_n+h})$ for all $h$
   - Weak stationarity: $E[X_t] = \mu$ (constant) and $Cov(X_t, X_{t+h}) = f(h)$ (depends only on lag)
3. **Ergodicity**: Time averages converge to ensemble averages

### Random Walk

A simple random walk on integers:

- $S_0 = 0$
- $S_n = \sum_{i=1}^{n} X_i$ where $X_i$ are i.i.d. random variables
- For symmetric case: $P(X_i = 1) = P(X_i = -1) = \frac{1}{2}$

## Discrete Markov Chains

### Definition

A stochastic process $\{X_n, n \geq 0\}$ with the Markov property:
$P(X_{n+1} = j | X_0 = i_0, X_1 = i_1, ..., X_n = i) = P(X_{n+1} = j | X_n = i)$

### Transition Matrix

The one-step transition probability matrix $P$:

$P = \begin{pmatrix}
p_{11} & p_{12} & \cdots & p_{1n} \\
p_{21} & p_{22} & \cdots & p_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
p_{n1} & p_{n2} & \cdots & p_{nn}
\end{pmatrix}$

where $p_{ij} = P(X_{n+1} = j | X_n = i)$

### n-Step Transition Probabilities

$p_{ij}^{(n)} = P(X_{m+n} = j | X_m = i)$

The n-step transition matrix: $P^{(n)} = P^n$ (matrix power)

### Classification of States

1. **Recurrent State**: $P(X_n = i \text{ for some } n \geq 1 | X_0 = i) = 1$

   - Expected return time for recurrent state $i$: $\mu_i = E[T_i] < \infty$ for positive recurrent
   - Null recurrent if $\mu_i = \infty$

2. **Transient State**: $P(X_n = i \text{ for some } n \geq 1 | X_0 = i) < 1$

3. **Absorbing State**: $p_{ii} = 1$ (once entered, never leaves)

4. **Periodic State**: $d(i) = \gcd\{n > 0: p_{ii}^{(n)} > 0\} > 1$
   - Aperiodic if $d(i) = 1$

### Mean First Passage and Recurrence Times

- First passage time: $T_{ij} = \min\{n \geq 1: X_n = j | X_0 = i\}$
- Mean first passage time: $m_{ij} = E[T_{ij}]$
- For $i \neq j$: $m_{ij} = 1 + \sum_{k \neq j} p_{ik} m_{kj}$
- Recurrence time: $T_{ii} = \min\{n \geq 1: X_n = i | X_0 = i\}$
- Mean recurrence time: $\mu_i = E[T_{ii}]$

### Stationary and Limiting Distributions

For an irreducible and aperiodic Markov chain:

1. **Stationary Distribution** $\pi$:

   - $\pi P = \pi$ (equivalent to $\sum_{i} \pi_i p_{ij} = \pi_j$)
   - $\sum_{i} \pi_i = 1$
   - For positive recurrent states: $\pi_i = \frac{1}{\mu_i}$

2. **Limiting Distribution**:
   - $\lim_{n \to \infty} p_{ij}^{(n)} = \pi_j$ for all $i,j$

### Absorption Probabilities

For a Markov chain with transient states $T$ and absorbing states $A$:

- $h_{ij} = P(\text{absorption in state } j | X_0 = i)$ for $i \in T, j \in A$
- $\mathbf{h} = \mathbf{N} \mathbf{R}$ where:
  - $\mathbf{N} = (I - Q)^{-1}$ is the fundamental matrix
  - $Q$ is the transition matrix restricted to transient states
  - $\mathbf{R}$ contains probabilities from transient to absorbing states

### Expected Times to Absorption

- $t_i = E[\text{steps until absorption} | X_0 = i]$ for $i \in T$
- $\mathbf{t} = \mathbf{N} \mathbf{1}$ where $\mathbf{1}$ is a column vector of 1's

### Reducible Markov Chains

A Markov chain is reducible if the state space can be partitioned into two or more communicating classes. The transition matrix can be expressed in canonical form:

$P = \begin{pmatrix}
T & R \\
0 & I
\end{pmatrix}$

where:

- $T$ describes transitions between transient states
- $R$ describes transitions from transient to recurrent states
- $0$ is a matrix of zeros (can't return from recurrent to transient)
- $I$ describes transitions between recurrent states

## Hidden Markov Models

### Definition

- Observed process: $\{Y_t\}$
- Hidden Markov process: $\{X_t\}$

### Key Components

1. **Transition Probabilities**:

   - $a_{ij} = P(X_{t+1} = j | X_t = i)$
   - Transition matrix $A = [a_{ij}]$

2. **Emission Probabilities**:

   - $b_j(k) = P(Y_t = k | X_t = j)$
   - Emission matrix $B = [b_j(k)]$

3. **Initial State Distribution**:
   - $\pi_i = P(X_1 = i)$

### Forward Algorithm

For calculating $P(Y_1, Y_2, ..., Y_T)$:

1. Initialization: $\alpha_1(i) = \pi_i b_i(Y_1)$

2. Recursion: $\alpha_{t+1}(j) = \left[\sum_{i=1}^N \alpha_t(i) a_{ij} \right] b_j(Y_{t+1})$

3. Termination: $P(Y_1, Y_2, ..., Y_T) = \sum_{i=1}^N \alpha_T(i)$

where $\alpha_t(i) = P(Y_1, Y_2, ..., Y_t, X_t = i)$

## Poisson Processes

### Definition

A continuous-time counting process $\{N(t), t \geq 0\}$ with:

1. $N(0) = 0$
2. Independent increments
3. $P(N(t+h) - N(t) = 1) = \lambda h + o(h)$ for small $h$
4. $P(N(t+h) - N(t) \geq 2) = o(h)$ for small $h$

### Key Properties

1. **Distribution of $N(t)$**:

   - $P(N(t) = n) = \frac{(\lambda t)^n e^{-\lambda t}}{n!}$, i.e., $N(t) \sim \text{Poisson}(\lambda t)$

2. **Inter-arrival Times**:

   - If $T_1, T_2, ...$ are the inter-arrival times, then $T_i \sim \text{Exp}(\lambda)$
   - $P(T_i > t) = e^{-\lambda t}$
   - $E[T_i] = \frac{1}{\lambda}$

3. **Waiting Time Until $n$th Event**:

   - If $S_n = \sum_{i=1}^n T_i$, then $S_n \sim \text{Gamma}(n, \lambda)$
   - $P(S_n \leq t) = P(N(t) \geq n)$

4. **Conditional Distribution of Arrival Times**:
   - Given $N(t) = n$, the unordered arrival times are distributed as $n$ independent uniform random variables on $[0,t]$

## Queueing Theory

### Kendall-Lee Notation

- **A/B/c/K/m/Z** where:
  - **A**: Arrival process distribution
  - **B**: Service time distribution
  - **c**: Number of servers
  - **K**: System capacity (default: ∞)
  - **m**: Population size (default: ∞)
  - **Z**: Queue discipline (default: FIFO)

Common distributions:

- **M**: Markovian (exponential)
- **D**: Deterministic
- **G**: General

### M/M/1 Queue

#### Arrival and Service Process

- Arrivals: Poisson process with rate $\lambda$
- Service times: Exponentially distributed with rate $\mu$
- Traffic intensity: $\rho = \frac{\lambda}{\mu}$

#### Birth-Death Process

For M/M/1, the birth rate $\lambda_n = \lambda$ and death rate $\mu_n = \mu$ for all $n > 0$.

#### Steady-State Probabilities

For stable queue ($\rho < 1$):

- $p_0 = 1 - \rho$
- $p_n = \rho^n p_0 = \rho^n (1 - \rho)$ for $n \geq 0$

#### Performance Measures

1. **Expected number of customers in system**:

   - $L = \frac{\rho}{1-\rho}$

2. **Expected number of customers in queue**:

   - $L_q = \frac{\rho^2}{1-\rho}$

3. **Expected waiting time in system**:

   - $W = \frac{1}{\mu - \lambda} = \frac{1}{\mu(1-\rho)}$

4. **Expected waiting time in queue**:

   - $W_q = \frac{\rho}{\mu - \lambda} = \frac{\rho}{\mu(1-\rho)}$

5. **Little's Law**:
   - $L = \lambda W$
   - $L_q = \lambda W_q$

## Gaussian Processes

### Definition

A Gaussian Process (GP) is a collection of random variables where any finite subset follows a multivariate Gaussian distribution.

A GP is fully specified by:

1. A mean function: $m(x) = E[f(x)]$
2. A covariance function (kernel): $k(x, x') = E[(f(x)-m(x))(f(x')-m(x'))]$

For inputs $X = \{x_1, x_2, ..., x_n\}$, outputs follow:
$f(X) \sim \mathcal{N}(m(X), K(X, X))$

where $K(X, X)_{ij} = k(x_i, x_j)$

### Common Kernels

1. **Squared Exponential (RBF)**:

   - $k(x, x') = \sigma_f^2 \exp\left(-\frac{||x-x'||^2}{2l^2}\right)$

2. **Matérn Kernel** (with $\nu = 3/2$):

   - $k(x, x') = \sigma_f^2 (1 + \frac{\sqrt{3}||x-x'||}{l}) \exp(-\frac{\sqrt{3}||x-x'||}{l})$

3. **Periodic Kernel**:
   - $k(x, x') = \sigma_f^2 \exp\left(-\frac{2\sin^2(\pi||x-x'||/p)}{l^2}\right)$

### Posterior Distribution

Given:

- Training data $(X, y)$ where $y = f(X) + \epsilon$ and $\epsilon \sim \mathcal{N}(0, \sigma_n^2 I)$
- Test points $X_*$

The posterior distribution is:
$f_* | X, y, X_* \sim \mathcal{N}(\bar{f}_*, \text{cov}(f_*))$

where:

- $\bar{f}_* = K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1}y$
- $\text{cov}(f_*) = K(X_*, X_*) - K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1}K(X, X_*)$

### Hyperparameter Estimation

The hyperparameters $\theta$ (kernel parameters, noise variance) can be estimated by maximizing the log marginal likelihood:

$\log p(y|X, \theta) = -\frac{1}{2}y^T(K + \sigma_n^2 I)^{-1}y - \frac{1}{2}\log|K + \sigma_n^2 I| - \frac{n}{2}\log(2\pi)$

In R, optimization is typically done using methods like L-BFGS-B.

### Prediction using Gaussian Processes

For a test point $x_*$, the predictive distribution is:
$f(x_*) | X, y, x_* \sim \mathcal{N}(\mu_*, \sigma_*^2)$

where:

- $\mu_* = k_*^T (K + \sigma_n^2 I)^{-1} y$
- $\sigma_*^2 = k(x_*, x_*) - k_*^T (K + \sigma_n^2 I)^{-1} k_*$

with $k_* = [k(x_1, x_*), k(x_2, x_*), ..., k(x_n, x_*)]^T$

## Implementation Files

This repository includes the following R scripts:

- `matrix_operations.R`: Basic matrix operations fundamental to statistical modeling
- `markov_chains.R`: Implementation of discrete Markov chains
- `markov_chains.py`: Python equivalent of the Markov chain implementation
- `hmm_and_poisson.R`: Hidden Markov Models and Poisson processes
- `queueing_theory.R`: Queueing theory models, particularly M/M/1 queues
- `gaussian_processes.R`: Gaussian process regression with hyperparameter optimization
- `poisson_process_calculator.R`: Specialized calculator for Poisson processes

Each file contains detailed implementations using only built-in R functions, with extensive comments and examples.
