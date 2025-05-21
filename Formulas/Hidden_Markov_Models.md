# Hidden Markov Model (HMM) Formulas

## 1. Basic Elements of an HMM

An HMM is characterized by:

- $S = \{s_1, s_2, ..., s_N\}$ = Set of hidden states
- $O = \{o_1, o_2, ..., o_M\}$ = Set of possible observations
- $A = \{a_{ij}\}$ = State transition probabilities, where $a_{ij} = P(q_{t+1} = s_j | q_t = s_i)$
- $B = \{b_j(k)\}$ = Observation probabilities, where $b_j(k) = P(o_k \text{ at } t | q_t = s_j)$
- $\pi = \{\pi_i\}$ = Initial state probabilities, where $\pi_i = P(q_1 = s_i)$

## 2. Three Basic Problems for HMMs

### Evaluation: Forward Algorithm

Computing the probability of an observation sequence $O = o_1, o_2, ..., o_T$ given the model $\lambda = (A, B, \pi)$, i.e., $P(O|\lambda)$

Forward variable:

$$
\alpha_t(i) = P(o_1, o_2, ..., o_t, q_t = s_i | \lambda)
$$

Recursive calculation:

$$
\alpha_1(i) = \pi_i b_i(o_1)
$$

$$
\alpha_{t+1}(j) = \left[ \sum_{i=1}^{N} \alpha_t(i) a_{ij} \right] b_j(o_{t+1})
$$

Final probability:

$$
P(O|\lambda) = \sum_{i=1}^{N} \alpha_T(i)
$$

### Evaluation: Backward Algorithm

Backward variable:

$$
\beta_t(i) = P(o_{t+1}, o_{t+2}, ..., o_T | q_t = s_i, \lambda)
$$

Recursive calculation:

$$
\beta_T(i) = 1
$$

$$
\beta_t(i) = \sum_{j=1}^{N} a_{ij} b_j(o_{t+1}) \beta_{t+1}(j)
$$

Final probability (alternative computation):

$$
P(O|\lambda) = \sum_{i=1}^{N} \pi_i b_i(o_1) \beta_1(i)
$$

### Decoding: Viterbi Algorithm

Finding the most likely sequence of hidden states given observations.

Define:

$$
\delta_t(i) = \max_{q_1, q_2, ..., q_{t-1}} P(q_1, q_2, ..., q_t = s_i, o_1, o_2, ..., o_t | \lambda)
$$

Recursive calculation:

$$
\delta_1(i) = \pi_i b_i(o_1)
$$

$$
\delta_{t+1}(j) = \max_{1 \leq i \leq N} [\delta_t(i) a_{ij}] b_j(o_{t+1})
$$

$$
\psi_{t+1}(j) = \arg\max_{1 \leq i \leq N} [\delta_t(i) a_{ij}]
$$

State sequence backtracking:

$$
q_T^* = \arg\max_{1 \leq i \leq N} [\delta_T(i)]
$$

$$
q_t^* = \psi_{t+1}(q_{t+1}^*) \text{ for } t = T-1, T-2, ..., 1
$$

### Learning: Baum-Welch Algorithm (EM for HMMs)

Define:

$$
\gamma_t(i) = P(q_t = s_i | O, \lambda) = \frac{\alpha_t(i) \beta_t(i)}{P(O|\lambda)}
$$

$$
\xi_t(i,j) = P(q_t = s_i, q_{t+1} = s_j | O, \lambda) = \frac{\alpha_t(i) a_{ij} b_j(o_{t+1}) \beta_{t+1}(j)}{P(O|\lambda)}
$$

Re-estimation formulas:

$$
\bar{\pi}_i = \gamma_1(i)
$$

$$
\bar{a}_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}
$$

$$
\bar{b}_j(k) = \frac{\sum_{t=1, o_t=k}^{T} \gamma_t(j)}{\sum_{t=1}^{T} \gamma_t(j)}
$$

## 3. Expected Value Formulas

### Expected number of transitions from state $s_i$ to $s_j$

$$
E[\text{number of transitions from } s_i \text{ to } s_j] = \sum_{t=1}^{T-1} \xi_t(i,j)
$$

### Expected number of times in state $s_i$

$$
E[\text{number of times in state } s_i] = \sum_{t=1}^{T} \gamma_t(i)
$$
