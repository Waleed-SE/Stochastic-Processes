# Markov Chains Formulas

## 1. Markov Property

For a stochastic process $\{X_t\}$ with state space $S$, the Markov property states:

$$
P(X_{n+1} = j | X_n = i, X_{n-1} = i_{n-1}, ..., X_0 = i_0) = P(X_{n+1} = j | X_n = i)
$$

## 2. Transition Probability Matrix

For a Markov chain with $N$ states, the transition probability matrix $P$ is:

$$
P = \begin{pmatrix}
p_{11} & p_{12} & \cdots & p_{1N} \\
p_{21} & p_{22} & \cdots & p_{2N} \\
\vdots & \vdots & \ddots & \vdots \\
p_{N1} & p_{N2} & \cdots & p_{NN}
\end{pmatrix}
$$

Where:

- $p_{ij}$ = Probability of transitioning from state $i$ to state $j$
- Each row must sum to 1: $\sum_{j=1}^{N} p_{ij} = 1$ for all $i$

## 3. Chapman-Kolmogorov Equations

$$
p_{ij}^{(n+m)} = \sum_{k \in S} p_{ik}^{(n)} p_{kj}^{(m)}
$$

Where:

- $p_{ij}^{(n)}$ = Probability of going from state $i$ to state $j$ in exactly $n$ steps

## 4. n-Step Transition Probability

$$
P^{(n)} = P^n
$$

Where:

- $P^{(n)}$ is the transition probability matrix for $n$ steps
- $P^n$ is the matrix $P$ raised to the power of $n$

## 5. Stationary Distribution

A probability vector $\pi = [\pi_1, \pi_2, ..., \pi_N]$ is a stationary distribution if:

$$
\pi P = \pi
$$

This means:

$$
\pi_j = \sum_{i \in S} \pi_i p_{ij} \text{ for all } j \in S
$$

With the constraint:

$$
\sum_{i \in S} \pi_i = 1
$$

## 6. Mean First Passage Time

The mean first passage time $m_{ij}$ from state $i$ to state $j$ is:

$$
m_{ij} = 1 + \sum_{k \neq j} p_{ik} m_{kj}
$$

For $i \neq j$, and $m_{jj} = 0$

## 7. Absorption Probabilities

For a Markov chain with transient states $T$ and absorbing states $A$, the probability of absorption in state $j \in A$ when starting from state $i \in T$ is:

$$
b_{ij} = \sum_{k \in T} p_{ik} b_{kj} + p_{ij}
$$

## 8. Expected Time to Absorption

Starting from transient state $i$, the expected number of steps before absorption is:

$$
t_i = 1 + \sum_{j \in T} p_{ij} t_j
$$
