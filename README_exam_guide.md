# Stochastic Processes & Statistical Modeling

A comprehensive guide covering mathematical concepts for stochastic processes, probability theory, and statistical modeling with formulas and practical examples.

## Table of Contents

- [Probability Theory](#probability-theory)
- [Stochastic Processes](#stochastic-processes)
- [Discrete Markov Chains](#discrete-markov-chains)
- [Hidden Markov Models](#hidden-markov-models)
- [Poisson Processes](#poisson-processes)
- [Queueing Theory](#queueing-theory)
- [Gaussian Processes](#gaussian-processes)

## Probability Theory

### Laws of Probability

#### Addition Law (OR)

For disjoint events (cannot happen simultaneously):

- **Formula**: $P(A \cup B) = P(A) + P(B)$
- **Example**: Probability of rolling either a 1 OR a 2 on a fair die = 1/6 + 1/6 = 1/3

For any events (may overlap):

- **Formula**: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
- **Example**: Probability of drawing a King OR a Heart from a deck of cards = 4/52 + 13/52 - 1/52 = 16/52 = 4/13

#### Multiplication Law (AND)

General case:

- **Formula**: $P(A \cap B) = P(A) \cdot P(B|A)$
- **Example**: Probability of drawing two Aces in a row without replacement = 4/52 × 3/51 = 1/221

For independent events:

- **Formula**: $P(A \cap B) = P(A) \cdot P(B)$
- **Example**: Probability of rolling a 6 twice in a row = 1/6 × 1/6 = 1/36

### Conditional Probability

- **Formula**: $P(A|B) = \frac{P(A \cap B)}{P(B)}$, where $P(B) > 0$
- **Example**: If 70% of students pass math (M) and 50% pass both math and physics (P), then P(P|M) = 0.5/0.7 = 0.714 or 71.4%

### Bayes' Theorem

- **Formula**: $P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$

For multiple hypotheses:

- **Formula**: $P(A_i|B) = \frac{P(B|A_i) \cdot P(A_i)}{\sum_{j} P(B|A_j) \cdot P(A_j)}$

- **Example**:
  - 1% of people have a certain disease
  - A test is 95% accurate for people with the disease
  - The test gives false positives 10% of the time
  - If someone tests positive, the probability they actually have the disease is:
  - P(Disease|Positive) = (0.95 × 0.01)/[(0.95 × 0.01) + (0.10 × 0.99)] ≈ 0.088 or 8.8%

## Stochastic Processes

### Definition and Classification

A stochastic process is a collection of random variables {X(t), t ∈ T} indexed by a parameter t (often time).

#### Classification

**By State Space**:

- **Discrete**: Values are countable (e.g., number of customers)
- **Continuous**: Values can be any real number (e.g., temperature)

**By Parameter Space (Time)**:

- **Discrete**: Time moves in steps (e.g., daily stock prices)
- **Continuous**: Time flows continuously (e.g., radioactive decay)

**By Memory**:

- **Memoryless (Markov)**: Future depends only on the present
- **With Memory**: Future depends on past history

### Key Properties

1. **Trajectory**: A single realization of the process

   - **Example**: One specific path of stock prices over time

2. **Stationarity**:

   - **Strict stationarity**:

     - **Formula**: $P(X_{t_1}, X_{t_2}, ..., X_{t_n}) = P(X_{t_1+h}, X_{t_2+h}, ..., X_{t_n+h})$ for all $h$
     - **Example**: White noise process has the same statistical properties regardless of when you observe it

   - **Weak stationarity**:
     - Mean is constant: $E[X_t] = \mu$
     - Autocovariance depends only on time difference: $Cov(X_t, X_{t+h}) = f(h)$
     - **Example**: Seasonal temperature variations around a constant mean

3. **Ergodicity**: Time averages converge to ensemble averages
   - **Example**: Recording temperature at one location for years gives the same information as recording many locations simultaneously for one day

### Random Walk

- **Starting point**: $S_0 = 0$
- **Position after n steps**: $S_n = \sum_{i=1}^{n} X_i$ where $X_i$ are random steps
- **Example**: For a symmetric random walk where P(step right) = P(step left) = 0.5:
  - Starting at position 0, take 3 steps
  - Possible paths: RRR, RRL, RLR, RLL, LRR, LRL, LLR, LLL
  - Each path has probability 1/8
  - Expected position after 3 steps = 0 (by symmetry)
  - Variance of position after 3 steps = 3

## Discrete Markov Chains

### Definition

A stochastic process {Xn, n ≥ 0} with the Markov property:

- **Formula**: $P(X_{n+1} = j | X_0 = i_0, X_1 = i_1, ..., X_n = i) = P(X_{n+1} = j | X_n = i)$
- **Example**: Weather model where tomorrow's weather only depends on today's weather, not on weather from previous days

### Transition Matrix

The one-step transition probability matrix P with elements:

- **Formula**: $p_{ij} = P(X_{n+1} = j | X_n = i)$

- **Example**: Weather model with three states (Sunny, Cloudy, Rainy):

```
P = | 0.7  0.2  0.1 |  (Probabilities from Sunny to all states)
    | 0.3  0.4  0.3 |  (Probabilities from Cloudy to all states)
    | 0.2  0.3  0.5 |  (Probabilities from Rainy to all states)
```

- If today is sunny, there's a 70% chance tomorrow will be sunny too

### n-Step Transition Probabilities

- **Formula**: $p_{ij}^{(n)} = P(X_{m+n} = j | X_m = i)$
- The n-step transition matrix: $P^{(n)} = P^n$ (matrix power)

- **Example**: Using the weather example, the 2-step transition matrix is:

```
P² = | 0.56  0.25  0.19 |
     | 0.38  0.32  0.30 |
     | 0.33  0.32  0.35 |
```

- Starting from sunny, the probability of rain two days later is 0.19 or 19%

### Classification of States

1. **Recurrent State**: $P(X_n = i \text{ for some } n \geq 1 | X_0 = i) = 1$

   - **Example**: In the weather model, all states are recurrent as we'll eventually return to any weather condition

   - **Positive Recurrent**: Mean return time $\mu_i = E[T_i] < \infty$
   - **Null Recurrent**: Mean return time $\mu_i = \infty$
   - **Example**: For a symmetric random walk on an infinite line, all states are null recurrent

2. **Transient State**: $P(X_n = i \text{ for some } n \geq 1 | X_0 = i) < 1$

   - **Example**: In a gambler's ruin problem, all intermediate bankroll amounts are transient states

3. **Absorbing State**: $p_{ii} = 1$ (once entered, never leaves)

   - **Example**: In a birth-death process with extinction, the state "0 population" is absorbing

4. **Periodic State**: $d(i) = \gcd\{n > 0: p_{ii}^{(n)} > 0\} > 1$
   - **Example**: In a simple two-state chain where P = [0 1; 1 0], both states have period 2 as you can only return after an even number of steps

### Mean First Passage and Recurrence Times

- **First passage time**: $T_{ij} = \min\{n \geq 1: X_n = j | X_0 = i\}$
- **Mean first passage time**: $m_{ij} = E[T_{ij}]$
- **Formula**: For $i \neq j$: $m_{ij} = 1 + \sum_{k \neq j} p_{ik} m_{kj}$

- **Example**: In a weather model, if today is sunny, the expected number of days until it rains might be 4.3 days

- **Recurrence time**: $T_{ii} = \min\{n \geq 1: X_n = i | X_0 = i\}$
- **Mean recurrence time**: $\mu_i = E[T_{ii}]$

- **Example**: If it's sunny today, the expected number of days until it's sunny again might be 1.5 days

### Stationary and Limiting Distributions

For an irreducible and aperiodic Markov chain:

1. **Stationary Distribution** $\pi$:

   - **Formulas**:

     - $\pi P = \pi$ (equivalent to $\sum_{i} \pi_i p_{ij} = \pi_j$)
     - $\sum_{i} \pi_i = 1$
     - For positive recurrent states: $\pi_i = \frac{1}{\mu_i}$

   - **Example**: For our weather model, the stationary distribution might be π = [0.5, 0.3, 0.2], meaning in the long run, it's sunny 50% of days, cloudy 30%, and rainy 20%

2. **Limiting Distribution**:

   - **Formula**: $\lim_{n \to \infty} p_{ij}^{(n)} = \pi_j$ for all $i,j$

   - **Example**: No matter what today's weather is, the probability of rain 1000 days from now approaches 0.2 (from the stationary distribution)

### Absorption Probabilities

For a Markov chain with transient states T and absorbing states A:

- **Absorption probability**: $h_{ij} = P(\text{absorption in state } j | X_0 = i)$ for $i \in T, j \in A$
- **Formula**: $\mathbf{h} = \mathbf{N} \mathbf{R}$ where:

  - $\mathbf{N} = (I - Q)^{-1}$ is the fundamental matrix
  - $Q$ is the transition matrix restricted to transient states
  - $\mathbf{R}$ contains probabilities from transient to absorbing states

- **Example**: In a gambler's ruin problem with bankroll of $10 and betting $1 each time with 40% win probability:
  - Probability of reaching $20 before going bankrupt = 0.0018 (almost certainly will go bankrupt)

### Expected Times to Absorption

- **Expected time**: $t_i = E[\text{steps until absorption} | X_0 = i]$ for $i \in T$
- **Formula**: $\mathbf{t} = \mathbf{N} \mathbf{1}$ where $\mathbf{1}$ is a column vector of 1's

- **Example**: In the same gambler's ruin problem, starting with $10, the expected number of bets until either going broke or reaching $20 is about 18.7 bets

### Reducible Markov Chains

A Markov chain is reducible if the state space can be partitioned into separate communicating classes.

- **Canonical form**:

```
P = | T  R |
    | 0  I |
```

Where:

- T describes transitions between transient states
- R describes transitions from transient to recurrent states
- 0 is a matrix of zeros (can't return from recurrent to transient)
- I describes transitions between recurrent states

- **Example**: In an educational model with states (High School, College, Graduate, Employed, Retired):
  - Can move between first three states
  - Once employed, can't go back to education
  - Once retired, stay retired
  - The transition matrix would have this reducible structure

## Hidden Markov Models

### Definition

- **Observed process**: {Yt}
- **Hidden Markov process**: {Xt}

- **Example**: Speech recognition where the spoken words are hidden states, and the audio signal is the observation

### Key Components

1. **Transition Probabilities**:

   - **Formula**: $a_{ij} = P(X_{t+1} = j | X_t = i)$
   - **Example**: In a speech model, probability of the next word being "rain" given the current word is "heavy"

2. **Emission Probabilities**:

   - **Formula**: $b_j(k) = P(Y_t = k | X_t = j)$
   - **Example**: Probability of observing certain sound frequencies given the spoken word is "rain"

3. **Initial State Distribution**:
   - **Formula**: $\pi_i = P(X_1 = i)$
   - **Example**: Probability that the first word in a sentence is "the"

### Forward Algorithm

For calculating $P(Y_1, Y_2, ..., Y_T)$:

1. **Initialization**:

   - **Formula**: $\alpha_1(i) = \pi_i b_i(Y_1)$
   - **Example**: Probability of starting with word i and observing the first sound

2. **Recursion**:

   - **Formula**: $\alpha_{t+1}(j) = \left[\sum_{i=1}^N \alpha_t(i) a_{ij} \right] b_j(Y_{t+1})$
   - **Example**: Building up probability of word sequence by considering all possible previous words

3. **Termination**:
   - **Formula**: $P(Y_1, Y_2, ..., Y_T) = \sum_{i=1}^N \alpha_T(i)$
   - **Example**: Total probability of observing the entire sequence of sounds

## Poisson Processes

### Definition

A continuous-time counting process {N(t), t ≥ 0} with:

1. N(0) = 0
2. Independent increments
3. P(N(t+h) - N(t) = 1) = λh + o(h) for small h
4. P(N(t+h) - N(t) ≥ 2) = o(h) for small h

- **Example**: Customer arrivals at a store, where customers arrive independently at a constant average rate

### Key Properties

1. **Distribution of N(t)**:

   - **Formula**: $P(N(t) = n) = \frac{(\lambda t)^n e^{-\lambda t}}{n!}$
   - **Example**: If customers arrive at a rate of 3 per hour, the probability of exactly 5 customers in 2 hours is:
     - P(N(2) = 5) = (3×2)⁵ × e⁻⁶ ÷ 5! ≈ 0.1606 or 16.06%

2. **Inter-arrival Times**:

   - **Formula**:

     - $T_i \sim \text{Exp}(\lambda)$
     - $P(T_i > t) = e^{-\lambda t}$
     - $E[T_i] = \frac{1}{\lambda}$

   - **Example**: If buses arrive at 3 per hour:
     - The probability no bus arrives within 30 minutes is e⁻¹·⁵ ≈ 0.223 or 22.3%
     - The expected waiting time between buses is 20 minutes (1/3 of an hour)

3. **Waiting Time Until nth Event**:

   - **Formula**:

     - $S_n = \sum_{i=1}^n T_i \sim \text{Gamma}(n, \lambda)$
     - $P(S_n \leq t) = P(N(t) \geq n)$

   - **Example**: The time until the 5th customer arrives follows a Gamma(5, λ) distribution
     - With λ = 3 per hour, the expected time until the 5th customer is 5/3 hours or 1 hour and 40 minutes

## Queueing Theory

### Kendall-Lee Notation

- **A/B/c/K/m/Z** where:
  - A: Arrival process distribution
  - B: Service time distribution
  - c: Number of servers
  - K: System capacity (default: ∞)
  - m: Population size (default: ∞)
  - Z: Queue discipline (default: FIFO)

**Common distributions**:

- M: Markovian/exponential
- D: Deterministic
- G: General

- **Example**: M/M/1 = Single server queue with Poisson arrivals and exponential service times

### M/M/1 Queue

#### Arrival and Service Process

- **Arrivals**: Poisson process with rate λ
- **Service times**: Exponentially distributed with rate μ
- **Traffic intensity**: ρ = λ/μ

- **Example**: In a coffee shop, customers arrive at rate 20/hour, and service takes 2 minutes on average (μ = 30/hour)
  - Traffic intensity ρ = 20/30 = 2/3
  - System will be stable as ρ < 1

#### Steady-State Probabilities

For stable queue (ρ < 1):

- **Formulas**:

  - $p_0 = 1 - \rho$
  - $p_n = \rho^n p_0 = \rho^n (1 - \rho)$ for $n \geq 0$

- **Example**: In our coffee shop with ρ = 2/3:
  - Probability of empty system: p₀ = 1 - 2/3 = 1/3
  - Probability of exactly 2 customers in system: p₂ = (2/3)² × (1/3) = 4/27 ≈ 0.148 or 14.8%

#### Performance Measures

1. **Expected number of customers in system**:

   - **Formula**: $L = \frac{\rho}{1-\rho}$
   - **Example**: For our coffee shop, L = (2/3)/(1-2/3) = 2 customers on average

2. **Expected number of customers in queue**:

   - **Formula**: $L_q = \frac{\rho^2}{1-\rho}$
   - **Example**: For our coffee shop, Lq = (2/3)²/(1-2/3) = 4/3 ≈ 1.33 customers waiting on average

3. **Expected waiting time in system**:

   - **Formula**: $W = \frac{1}{\mu - \lambda} = \frac{1}{\mu(1-\rho)}$
   - **Example**: For our coffee shop, W = 1/(30-20) = 1/10 hour = 6 minutes total time

4. **Expected waiting time in queue**:

   - **Formula**: $W_q = \frac{\rho}{\mu - \lambda} = \frac{\rho}{\mu(1-\rho)}$
   - **Example**: For our coffee shop, Wq = (2/3)/10 = 1/15 hour = 4 minutes waiting in line

5. **Little's Law**:

   - **Formulas**:

     - $L = \lambda W$
     - $L_q = \lambda W_q$

   - **Example**: Verifying for our coffee shop:
     - L = 20 × (1/10) = 2 ✓
     - Lq = 20 × (1/15) = 4/3 ✓

## Gaussian Processes

### Definition

A Gaussian Process (GP) is a collection of random variables where any finite subset follows a multivariate Gaussian distribution.

A GP is fully specified by:

1. **Mean function**: $m(x) = E[f(x)]$
2. **Covariance function (kernel)**: $k(x, x') = E[(f(x)-m(x))(f(x')-m(x'))]$

- **Example**: Modeling temperatures across different locations, where:
  - Mean function represents the average temperature at each location
  - Covariance function captures how temperature at one location relates to another

### Common Kernels

1. **Squared Exponential (RBF)**:

   - **Formula**: $k(x, x') = \sigma_f^2 \exp\left(-\frac{||x-x'||^2}{2l^2}\right)$
   - **Example**: Models smooth functions where nearby points have similar values
   - Parameters: signal variance σ²ᶠ (overall variance) and length scale l (how quickly correlation drops with distance)

2. **Matérn Kernel** (with $\nu = 3/2$):

   - **Formula**: $k(x, x') = \sigma_f^2 (1 + \frac{\sqrt{3}||x-x'||}{l}) \exp(-\frac{\sqrt{3}||x-x'||}{l})$
   - **Example**: Allows for less smooth functions than RBF
   - Good for modeling natural phenomena like terrain elevation

3. **Periodic Kernel**:
   - **Formula**: $k(x, x') = \sigma_f^2 \exp\left(-\frac{2\sin^2(\pi||x-x'||/p)}{l^2}\right)$
   - **Example**: Models functions that repeat, like seasonal patterns
   - Parameter p controls the period of repetition

### Posterior Distribution

Given:

- Training data $(X, y)$ where $y = f(X) + \epsilon$ and $\epsilon \sim \mathcal{N}(0, \sigma_n^2 I)$
- Test points $X_*$

The posterior distribution is:

- **Formula**: $f_* | X, y, X_* \sim \mathcal{N}(\bar{f}_*, \text{cov}(f_*))$

Where:

- **Mean**: $\bar{f}_* = K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1}y$
- **Covariance**: $\text{cov}(f_*) = K(X_*, X_*) - K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1}K(X, X_*)$

- **Example**: Having observed temperatures at 10 locations, we can predict temperature at a new location with confidence intervals

### Hyperparameter Estimation

The hyperparameters $\theta$ (kernel parameters, noise variance) can be estimated by maximizing the log marginal likelihood:

- **Formula**: $\log p(y|X, \theta) = -\frac{1}{2}y^T(K + \sigma_n^2 I)^{-1}y - \frac{1}{2}\log|K + \sigma_n^2 I| - \frac{n}{2}\log(2\pi)$

- **Example**: Finding the optimal length scale and variance parameters that best explain our observed temperature data

### Prediction using Gaussian Processes

For a test point $x_*$, the predictive distribution is:

- **Formula**: $f(x_*) | X, y, x_* \sim \mathcal{N}(\mu_*, \sigma_*^2)$

Where:

- **Mean**: $\mu_* = k_*^T (K + \sigma_n^2 I)^{-1} y$
- **Variance**: $\sigma_*^2 = k(x_*, x_*) - k_*^T (K + \sigma_n^2 I)^{-1} k_*$
- With $k_* = [k(x_1, x_*), k(x_2, x_*), ..., k(x_n, x_*)]^T$

- **Example**: Predicting temperature at a new location:
  - Mean prediction might be 22°C
  - 95% confidence interval might be 22°C ± 1.96 × 1.2°C = [19.6°C, 24.4°C]
