# Poisson Process Formulas

## 1. Probability Mass Function

For a Poisson process with rate $\lambda$, the probability of $n$ events in time interval $t$ is:

$$
P(N(t) = n) = \frac{(\lambda t)^n e^{-\lambda t}}{n!}
$$

Where:

- $N(t)$ = Number of events up to time $t$
- $\lambda$ = Rate parameter (average number of events per unit time)
- $n$ = Number of events

## 2. Cumulative Distribution Function

$$
P(N(t) \leq n) = \sum_{i=0}^{n} \frac{(\lambda t)^i e^{-\lambda t}}{i!}
$$

## 3. Expected Value and Variance

$$
E[N(t)] = \lambda t
$$

$$
Var[N(t)] = \lambda t
$$

## 4. Interarrival Times

The time between consecutive events follows an exponential distribution:

$$
f_T(t) = \lambda e^{-\lambda t}, \, t \geq 0
$$

## 5. Memoryless Property

For exponential interarrival times, the memoryless property states:

$$
P(T > s+t | T > s) = P(T > t)
$$

Where $T$ is the time until the next event.

## 6. Superposition of Poisson Processes

If $N_1(t)$, $N_2(t)$, ..., $N_k(t)$ are independent Poisson processes with rates $\lambda_1$, $\lambda_2$, ..., $\lambda_k$, then their sum $N(t) = N_1(t) + N_2(t) + ... + N_k(t)$ is a Poisson process with rate $\lambda = \lambda_1 + \lambda_2 + ... + \lambda_k$.

## 7. Thinning of Poisson Process

If events in a Poisson process with rate $\lambda$ are independently classified as type 1 with probability $p$ and type 2 with probability $1-p$, then:

1. Type 1 events form a Poisson process with rate $\lambda p$
2. Type 2 events form a Poisson process with rate $\lambda(1-p)$
3. The two resulting processes are independent

## 8. Non-homogeneous Poisson Process

For a non-homogeneous Poisson process with rate function $\lambda(t)$, the probability of $n$ events in interval $(a,b)$ is:

$$
P(N(b) - N(a) = n) = \frac{\left(\int_a^b \lambda(t) dt\right)^n e^{-\int_a^b \lambda(t) dt}}{n!}
$$
