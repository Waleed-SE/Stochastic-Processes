# Queuing Theory Formulas

## 1. Basic Notation

- $\lambda$ = Arrival rate (customers per unit time)
- $\mu$ = Service rate (customers per unit time)
- $\rho = \lambda/\mu$ = Traffic intensity (for single server)
- $c$ = Number of servers
- $K$ = System capacity (maximum number of customers allowed)

## 2. M/M/1 Queue

### Utilization

$$
\rho = \frac{\lambda}{\mu}
$$

### Probability of n customers in system

$$
P_n = (1-\rho)\rho^n
$$

### Expected number of customers in system

$$
L = \frac{\rho}{1-\rho}
$$

### Expected number of customers in queue

$$
L_q = \frac{\rho^2}{1-\rho}
$$

### Expected waiting time in system

$$
W = \frac{1}{\mu - \lambda}
$$

### Expected waiting time in queue

$$
W_q = \frac{\rho}{\mu - \lambda}
$$

## 3. M/M/1/K Queue

### Probability of n customers in system

$$
P_n = \frac{(1-\rho)\rho^n}{1-\rho^{K+1}} \text{ for } \rho \neq 1
$$

$$
P_n = \frac{1}{K+1} \text{ for } \rho = 1
$$

### Expected number of customers in system

$$
L = \frac{\rho}{1-\rho} - \frac{(K+1)\rho^{K+1}}{1-\rho^{K+1}} \text{ for } \rho \neq 1
$$

$$
L = \frac{K}{2} \text{ for } \rho = 1
$$

### Blocking probability (probability that system is full)

$$
P_K = \frac{(1-\rho)\rho^K}{1-\rho^{K+1}} \text{ for } \rho \neq 1
$$

$$
P_K = \frac{1}{K+1} \text{ for } \rho = 1
$$

## 4. M/M/c Queue (See MMC.md for details)

### Utilization

$$
\rho = \frac{\lambda}{c\mu}
$$

### Probability of no customers in system

$$
P_0 = \left[ \sum_{n=0}^{c-1} \frac{(\lambda / \mu)^n}{n!} + \frac{(\lambda / \mu)^c}{c!} \frac{c\mu}{c\mu - \lambda} \right]^{-1}
$$

### Probability of customers waiting in queue

$$
P_q = \frac{P_0}{c!} \left( \frac{\lambda}{\mu} \right)^c \frac{c\mu}{c\mu - \lambda}
$$

## 5. M/M/c/K Queue

### Probability of n customers in system

$$
P_n =
\begin{cases}
P_0 \frac{(\lambda/\mu)^n}{n!}, & \text{for } 0 \leq n < c \\
P_0 \frac{(\lambda/\mu)^n}{c! c^{n-c}}, & \text{for } c \leq n \leq K
\end{cases}
$$

### Effective arrival rate (accounting for blocked customers)

$$
\lambda_{eff} = \lambda(1-P_K)
$$

## 6. M/G/1 Queue (Pollaczek-Khinchine Formula)

### Expected number of customers in queue

$$
L_q = \frac{\lambda^2 \sigma^2 + \rho^2}{2(1-\rho)}
$$

Where:

- $\sigma^2$ = Variance of service time
- $\rho = \lambda E[S]$ where $E[S]$ is the expected service time

### Expected waiting time in queue

$$
W_q = \frac{\lambda \sigma^2 + \rho^2/\lambda}{2(1-\rho)}
$$
