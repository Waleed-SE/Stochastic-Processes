# Stochastic Processes & Statistical Modeling: A Friendly Guide

This repository contains practical implementations of statistical models and random processes in R. The code uses only built-in R functions to demonstrate key concepts in probability, prediction, and data analysis.

## Exam Topics & Concepts Covered

### 1. Probability Theory Fundamentals

#### Basic Laws of Probability

- **Addition Law (OR rule)** ✓
  - For separate events: "Probability of A OR B = Sum of their individual probabilities"
  - For overlapping events: "Probability of A OR B = Sum of individual probabilities MINUS the overlap"
- **Multiplication Law (AND rule)** ✓
  - For any events: "Probability of A AND B = Probability of A × Probability of B given A"
  - For independent events: "Probability of A AND B = Probability of A × Probability of B"
- **Conditional Probability** ✓
  - "Probability of A given B = Probability of both A and B happening ÷ Probability of B"
- **Bayes' Theorem** ✓
  - The powerful formula for updating beliefs: "Posterior = Likelihood × Prior ÷ Evidence"
  - Written in English: "The probability of a hypothesis given new evidence equals the probability of seeing that evidence if the hypothesis is true, times your initial belief in the hypothesis, divided by the total probability of seeing that evidence regardless of hypothesis"

### 2. Stochastic Processes - Random Events Over Time

#### What are Stochastic Processes?

A stochastic process is a collection of random variables that evolve over time according to probability rules.

**Real-life examples:**

- Stock market prices
- Weather patterns
- Number of customers in a store
- Traffic flow on highways

#### Classification of Stochastic Processes:

Based on:

1. **State Space**: What values can it take?

   - **Discrete**: Only specific values (like number of people)
   - **Continuous**: Any value within a range (like temperature)

2. **Time/Parameter Space**: How does time flow?

   - **Discrete**: Time moves in steps or jumps
   - **Continuous**: Time flows smoothly

3. **Memory**: How much of the past matters?
   - **Memoryless (Markov)**: Only the present state matters
   - **With Memory**: Past history also affects the future

#### Key Properties

- **Trajectory**: One specific realization or "path" the process takes
- **Stationarity**: When statistical properties don't change over time
  - Like a river flowing at a constant average rate
- **Ergodicity**: When time averages equal space averages
  - Like saying "watching one person for a long time gives the same information as watching many people for a short time"

#### Random Walk

A random walk is the simplest stochastic process - think of a person taking random steps:

- Start at position zero
- Take steps forward or backward with equal probability
- Your position after n steps is the sum of all previous steps

Applied to finance, this is the foundation of the "efficient market hypothesis" where stock prices follow a random walk.

### 3. Discrete Markov Chains

#### The Markov Property Explained

A Markov Chain is a process where the future depends only on the present state, not on how you got there.

**Everyday example**: Think of the weather. Whether it will rain tomorrow mostly depends on today's weather, not on what happened last week.

#### Transition Matrix in Plain Language

The transition matrix shows the probabilities of moving from one state to another:

```
Transition matrix shows:
┌───────────────────────────────────────────────┐
│ Probability of going from State 1 to State 1  │  ···  │ Probability of going from State 1 to State n  │
├───────────────────────────────────────────────┼───────┼───────────────────────────────────────────────┤
│ Probability of going from State 2 to State 1  │  ···  │ Probability of going from State 2 to State n  │
├───────────────────────────────────────────────┼───────┼───────────────────────────────────────────────┤
│                     ⋮                         │   ⋱   │                      ⋮                        │
├───────────────────────────────────────────────┼───────┼───────────────────────────────────────────────┤
│ Probability of going from State n to State 1  │  ···  │ Probability of going from State n to State n  │
└───────────────────────────────────────────────┴───────┴───────────────────────────────────────────────┘
```

#### Multi-Step Transitions

The n-step transition probability is the chance of going from state i to state j in exactly n steps.

**Simple explanation**: If you multiply the transition matrix by itself n times, you get the probabilities of going from any state to any other state in n steps.

#### Types of States in Markov Chains

1. **Recurrent State**: A state you will eventually return to (probability = 1)

   - **Positive Recurrent**: The expected return time is finite
   - **Null Recurrent**: You'll return eventually, but the expected time is infinite

2. **Transient State**: A state you might leave and never return to

3. **Absorbing State**: Once you enter, you stay forever (like retirement in a career model)

4. **Periodic State**: You can only return after a specific number of steps
   - **Aperiodic**: Can return at irregular intervals

#### First Passage & Return Times

- **First Passage Time**: How long it takes to first reach state j, starting from state i
  - Think of it as "How many days until it rains, given that today is sunny?"
- **Mean First Passage Time**: The average number of steps to reach state j from state i
  - Formula in words: "One step plus the weighted average of passage times from all possible next states"
- **Recurrence Time**: How long it takes to return to the same state
  - Think of it as "If it's sunny today, how many days until it's sunny again?"
- **Mean Recurrence Time**: The average number of steps to return to the same state
  - For a positive recurrent state, this equals 1/(long-term probability of being in that state)

#### Long-Term Behavior

A **stationary distribution** tells us the long-term probability of being in each state:

- It satisfies: "The probability of being in state j equals the sum of the probabilities of coming into j from all possible states"
- The sum of all state probabilities equals 1
- For positive recurrent states: "Long-term probability = 1/(mean recurrence time)"

**In plain English**: If you run a Markov chain for a very long time, it settles into a pattern where the probability of being in each state doesn't change anymore.

#### Absorption Probabilities

For Markov chains with some absorbing states:

- **Absorption probability**: The chance of eventually ending up in absorbing state j when starting from transient state i
- **Expected time to absorption**: The average number of steps until reaching any absorbing state

**Real-world example**: In a board game like Monopoly, what's the probability of ending up in jail, and how many turns will it take on average?

#### Reducible Markov Chains

A Markov chain is reducible when it has separate "communities" of states that don't fully communicate.

The transition matrix can be rearranged to show:

- Transitions within the transient states
- Transitions from transient to recurrent states
- No transitions from recurrent back to transient states
- Transitions within recurrent states

### 4. Hidden Markov Models

#### What is a Hidden Markov Model?

In a Hidden Markov Model (HMM), we have:

- A Markov process we cannot directly observe (the "hidden" states)
- Observable outcomes that depend on those hidden states

**Real-life example**: Voice recognition - we hear sounds (observations) but need to determine the words (hidden states) that produced them.

#### Key Components

1. **Transition Probabilities**: Chances of moving between hidden states
2. **Emission Probabilities**: Chances of observing different outcomes in each hidden state
3. **Initial State Distribution**: Probabilities of starting in each hidden state

#### The Forward Algorithm

The forward algorithm calculates the probability of a sequence of observations by:

1. **Initialization**: Calculate probabilities for the first observation
2. **Recursion**: Build up probabilities step by step, considering all possible paths
3. **Termination**: Sum up probabilities to get the final answer

**In everyday terms**: "What's the chance we would see this exact sequence of observations under our model?"

### 5. Poisson Processes - Modeling Random Arrivals

#### What is a Poisson Process?

A Poisson process models random events that:

- Occur one at a time
- Happen at a constant average rate
- Occur independently of each other

**Classic examples**: Customer arrivals, website visits, radioactive decay, traffic accidents

#### Key Properties and Formulas

1. **Number of Events in a Time Period**:
   - The number of events in time t follows a Poisson distribution
   - The probability of exactly n events = (λt)ⁿ × e^(-λt) ÷ n!
   - Where λ = average rate of events per unit time
2. **Time Between Events**:
   - The times between consecutive events follow an exponential distribution
   - The probability that waiting time exceeds t = e^(-λt)
   - Average waiting time = 1/λ
3. **Waiting Time Until n Events**:
   - The time until n events occur follows a gamma distribution
   - This lets us answer questions like "How long until 5 customers arrive?"

### 6. Queueing Theory - The Science of Waiting Lines

#### Kendall-Lee Notation

Queues are described using a shorthand: **A/B/c/K/m/Z**

- **A**: How customers arrive (arrival distribution)
- **B**: How long service takes (service time distribution)
- **c**: Number of servers
- **K**: Maximum system capacity
- **m**: Size of customer population
- **Z**: Queue discipline (FIFO, LIFO, etc.)

Common distributions:

- **M**: Memoryless/Markovian (exponential)
- **D**: Deterministic (fixed time)
- **G**: General (any distribution)

#### The M/M/1 Queue Model

The simplest queue: one server, exponential arrivals and service times

- **Arrival rate**: λ customers per unit time
- **Service rate**: μ customers per unit time
- **Traffic intensity**: ρ = λ/μ (must be < 1 for a stable queue)

#### Performance Measures

For a stable M/M/1 queue:

1. **Probability of n customers in system**: p₀ × ρⁿ

   - Where p₀ = 1-ρ (probability of empty system)

2. **Expected number of customers in system**: L = ρ/(1-ρ)

   - Example: If ρ = 0.8, then L = 4 customers

3. **Expected number of customers in queue**: Lq = ρ²/(1-ρ)

4. **Expected waiting time in system**: W = 1/(μ-λ)

   - The average time from arrival to departure

5. **Expected waiting time in queue**: Wq = ρ/(μ-λ)

   - The average time spent waiting before service

6. **Little's Law**: L = λW and Lq = λWq
   - Connects the number of customers to waiting times
   - The elegance of queueing theory in one simple relationship!

### 7. Gaussian Processes - Flexible Predictive Models

#### What are Gaussian Processes?

A Gaussian Process (GP) is a collection of random variables where any subset of them follows a multi-dimensional normal distribution.

**Intuitive explanation**: Think of a GP as drawing random curves through data points. Unlike simple regression that gives one "best" curve, a GP gives you a whole distribution of possible curves.

#### Key Components

A GP is defined by:

1. **Mean function**: The expected value at each point
2. **Covariance function (kernel)**: How points relate to each other
   - Points close together are more correlated than points far apart

#### Common Kernel Functions

1. **Squared Exponential**: For smooth functions

   - Similar to saying "nearby points should have similar values"

2. **Matérn Kernel**: For less smooth functions

   - Allows for more "wiggliness" between points

3. **Periodic Kernel**: For repeating patterns
   - Good for seasonal data or cycles

#### Making Predictions

To make predictions with a GP:

1. Start with training data (known points)
2. Choose a kernel function
3. Update your belief about the function based on observed data
4. Predict values at new points with uncertainty estimates

**In plain English**: "Based on the patterns in data I've seen, here's my best guess for this new point, along with how confident I am."

#### Hyperparameter Estimation

Hyperparameters control the behavior of the GP (like how quickly correlation falls off with distance).

In R, we find the best hyperparameters by maximizing the log marginal likelihood - essentially finding the parameters that make the observed data most probable.

## Implementation Files

This repository includes several R scripts:

- `matrix_operations.R`: Basic matrix math operations
- `markov_chains.R`: Models where future only depends on present state
- `markov_chains.py`: Python version of the Markov chain implementation
- `hmm_and_poisson.R`: Models for hidden states and random event arrivals
- `queueing_theory.R`: Tools for analyzing waiting lines and service systems
- `gaussian_processes.R`: Flexible prediction models with uncertainty estimates
- `poisson_process_calculator.R`: Calculator for random arrival processes

Each file contains detailed implementations with extensive comments and practical examples.

## Getting Started

To use these implementations:

1. Clone this repository
2. Open the R scripts in RStudio or your preferred R environment
3. Run the examples to see the models in action
4. Modify the parameters to fit your specific use case

The code is well-documented to help you understand both the mathematics and the implementation details.
