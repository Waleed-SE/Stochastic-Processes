# M/M/c Queuing Model Formulas

## 1. Utilization (Traffic Intensity)

$$
\rho = \frac{\lambda}{c \mu}
$$

Where:

- $\lambda$ = Arrival rate (customers per unit time)
- $\mu$ = Service rate per server (customers per unit time)
- $c$ = Number of servers

## 2. Probability of No Customers in System ($P_0$)

$$
P_0 = \left[ \sum_{n=0}^{c-1} \frac{(\lambda / \mu)^n}{n!} + \frac{(\lambda / \mu)^c}{c!} \frac{c\mu}{c\mu - \lambda} \right]^{-1}
$$

## 3. Probability of Customers Waiting in Queue ($P_q$)

$$
P_q = \frac{P_0}{c!} \left( \frac{\lambda}{\mu} \right)^c \frac{c\mu}{c\mu - \lambda}
$$

## 4. Expected Number of Customers in Queue ($L_q$)

$$
L_q = \frac{P_q \rho}{(1 - \rho)} \left( \frac{c\mu}{c\mu - \lambda} \right)
$$

## 5. Expected Number of Customers in the System ($L$)

$$
L = L_q + \frac{\lambda}{\mu}
$$

## 6. Expected Waiting Time in Queue ($W_q$)

$$
W_q = \frac{L_q}{\lambda}
$$

## 7. Expected Total Time in System ($W$)

$$
W = W_q + \frac{1}{\mu}
$$
