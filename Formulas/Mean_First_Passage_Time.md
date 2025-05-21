# Mean First Passage Time Formulas

## 1. Definition

The Mean First Passage Time (MFPT) $m_{ij}$ is the expected number of steps needed to reach state $j$ for the first time when starting from state $i$.

## 2. Basic Formula

For states $i \neq j$:

$$
m_{ij} = 1 + \sum_{k \neq j} p_{ik} m_{kj}
$$

Where:

- $p_{ik}$ = Transition probability from state $i$ to state $k$
- $m_{jj} = 0$ (by definition)

## 3. Matrix Formulation

Let $M$ be the matrix of MFPTs. For an $n$-state Markov chain with transition matrix $P$:

$$
M_{ij} =
\begin{cases}
1 + \sum_{k \neq j} P_{ik} M_{kj}, & \text{for } i \neq j \\
0, & \text{for } i = j
\end{cases}
$$

This can be written in matrix form as:

$$
(I - P)M = E - PD
$$

Where:

- $I$ = Identity matrix
- $E$ = Matrix of all 1's
- $P$ = Transition probability matrix
- $D$ = Diagonal matrix with $D_{jj} = M_{jj} = 0$

## 4. Using Fundamental Matrix

For ergodic Markov chains, let $Z = (I - P + W)^{-1}$ be the fundamental matrix, where $W$ has identical rows, each equal to the stationary distribution $\pi$.

The MFPT can be computed as:

$$
m_{ij} = \frac{z_{jj} - z_{ij}}{\pi_j}
$$

Where:

- $z_{ij}$ = $(i,j)$ element of $Z$
- $\pi_j$ = Stationary probability of state $j$

## 5. Mean Recurrence Time

The mean recurrence time (expected time to return to the same state) for an ergodic Markov chain is:

$$
m_{ii} = \frac{1}{\pi_i}
$$

## 6. Mean First Hitting Time

For a subset of states $A$, the mean first hitting time from state $i$ to set $A$ is:

$$
h_{i}^A =
\begin{cases}
0, & \text{if } i \in A \\
1 + \sum_{j} p_{ij} h_{j}^A, & \text{if } i \notin A
\end{cases}
$$

## 7. Kemeny's Constant

For an ergodic Markov chain, Kemeny's constant is the expected number of steps to reach a state chosen randomly according to the stationary distribution:

$$
K = \sum_{j} \pi_j m_{ij}
$$

Remarkably, this value is independent of the starting state $i$.
