# Discrete Markov Chains in R
# Using only built-in R functions (base R)

# ============================================================================
# 1. DEFINITION AND EXAMPLES OF MARKOV CHAINS
# ============================================================================
cat("\n======= MARKOV CHAIN DEFINITION =======\n")
cat("A Markov chain is a stochastic process where future states depend only on the current state\n")
cat("and not on the sequence of events that preceded it (Markov property).\n\n")

# Example 1: Simple Weather Model
cat("Example 1: Weather Model (Sunny, Cloudy, Rainy)\n")
weather_states <- c("Sunny", "Cloudy", "Rainy")

# Transition probability matrix P
# P[i,j] = Probability of moving from state i to state j
P_weather <- matrix(c(
  0.7, 0.2, 0.1,  # From Sunny: P(S->S)=0.7, P(S->C)=0.2, P(S->R)=0.1
  0.3, 0.4, 0.3,  # From Cloudy: P(C->S)=0.3, P(C->C)=0.4, P(C->R)=0.3
  0.2, 0.4, 0.4   # From Rainy: P(R->S)=0.2, P(R->C)=0.4, P(R->R)=0.4
), nrow = 3, byrow = TRUE)

# Display the transition matrix with row and column names
dimnames(P_weather) <- list(
  from = weather_states,
  to = weather_states
)
cat("\nTransition Probability Matrix:\n")
print(P_weather)

# Verify that each row sums to 1 (stochastic matrix property)
cat("\nVerify rows sum to 1:\n")
print(rowSums(P_weather))

# Example 2: Simple 2-state Markov Chain
cat("\nExample 2: Two-state Markov Chain\n")
two_states <- c("State A", "State B")
P_two_state <- matrix(c(
  0.6, 0.4,
  0.3, 0.7
), nrow = 2, byrow = TRUE)
dimnames(P_two_state) <- list(from = two_states, to = two_states)
cat("\nTransition Matrix for Two-state Chain:\n")
print(P_two_state)

# ============================================================================
# 2. TRANSITION PROBABILITY MATRIX
# ============================================================================
cat("\n\n======= TRANSITION PROBABILITY MATRIX =======\n")
cat("The transition probability matrix P contains probabilities pij of moving from state i to j.\n")
cat("Each row must sum to 1, as they represent probability distributions.\n")

# Let's create a more complex example
cat("\nExample 3: Four-state Markov Chain\n")
four_states <- c("State 1", "State 2", "State 3", "State 4")
P_four_state <- matrix(c(
  0.5, 0.3, 0.2, 0.0,
  0.0, 0.6, 0.3, 0.1,
  0.2, 0.0, 0.5, 0.3,
  0.1, 0.1, 0.0, 0.8
), nrow = 4, byrow = TRUE)
dimnames(P_four_state) <- list(from = four_states, to = four_states)
print(P_four_state)

# ============================================================================
# 3. CALCULATION OF n-STEP TRANSITION PROBABILITIES
# ============================================================================
cat("\n\n======= n-STEP TRANSITION PROBABILITIES =======\n")
cat("The n-step transition matrix P^n gives probabilities of going from state i to j in exactly n steps.\n")
cat("It is calculated by raising the transition matrix to the power n using matrix multiplication.\n")

# 2-step transition probabilities using matrix multiplication
P_weather_2step <- P_weather %*% P_weather
cat("\nTwo-step transition probabilities for the weather model:\n")
print(P_weather_2step)

# Calculate 3-step probabilities
P_weather_3step <- P_weather_2step %*% P_weather
cat("\nThree-step transition probabilities:\n")
print(P_weather_3step)

# Function to calculate n-step transition matrix
n_step_transition <- function(P, n) {
  result <- P
  for (i in 2:n) {
    result <- result %*% P
  }
  return(result)
}

# Calculate 10-step transition matrix
P_10step <- n_step_transition(P_weather, 10)
cat("\nTen-step transition probabilities:\n")
print(round(P_10step, 4))  # Round to 4 decimal places for readability

# ============================================================================
# 4. STEADY STATE PROBABILITY & STATIONARY DISTRIBUTIONS
# ============================================================================
cat("\n\n======= STEADY STATE PROBABILITY =======\n")
cat("The steady state (or stationary distribution) π is a probability vector such that π = π*P.\n")
cat("It represents the long-term probabilities of being in each state regardless of the initial state.\n")

# Method 1: Find steady state by raising the transition matrix to a high power
cat("\nMethod 1: Raising transition matrix to high power\n")
P_high_power <- n_step_transition(P_weather, 100)
cat("Approximating steady state by P^100:\n")
print(round(P_high_power[1,], 4))  # First row of the matrix

# Method 2: Solve the system of linear equations
# For a stationary distribution π, we need π = π*P and sum(π) = 1
cat("\nMethod 2: Solving the system of linear equations\n")
cat("For stationary distribution, we solve: π = π*P and sum(π) = 1\n")

# For a 3-state Markov chain:
# π₁ = π₁*p₁₁ + π₂*p₂₁ + π₃*p₃₁
# π₂ = π₁*p₁₂ + π₂*p₂₂ + π₃*p₃₂
# π₃ = π₁*p₁₃ + π₂*p₂₃ + π₃*p₃₃
# π₁ + π₂ + π₃ = 1

# We can rewrite as (I-P^T)π = 0 with additional constraint sum(π) = 1
# We'll replace one equation with sum(π) = 1 to get a unique solution

# For the Weather Example:
# Step 1: Create the system of equations
A <- t(diag(3) - P_weather)  # (I - P^T)
A[3,] <- rep(1, 3)           # Replace last row with the constraint equation
b <- c(0, 0, 1)              # Right hand side of the equations

# Step 2: Solve the system
steady_state <- solve(A, b)
cat("\nCalculated steady state distribution for weather model:\n")
names(steady_state) <- weather_states
print(round(steady_state, 4))

# Verify the result: π = π*P
verification <- steady_state %*% P_weather
cat("\nVerification - The steady state multiplied by P should equal itself:\n")
print(round(verification, 4))
cat("Difference between π and π*P:", sum(abs(steady_state - verification)))

# ============================================================================
# 5. CLASSIFICATION OF STATES
# ============================================================================
cat("\n\n======= CLASSIFICATION OF STATES =======\n")
cat("States in a Markov chain can be classified as:\n")
cat("- Recurrent: The process will return to this state with probability 1\n")
cat("- Transient: There's a non-zero probability of never returning\n")
cat("- Absorbing: Once entered, the process never leaves (p_ii = 1)\n")
cat("- Periodic: Returns occur at regular intervals\n")
cat("- Aperiodic: Returns can occur at irregular intervals\n")

# Example of an absorbing Markov chain - The Gambler's Ruin
cat("\nExample: Gambler's Ruin with $0 and $3 as absorbing states\n")
cat("States represent money: $0, $1, $2, $3\n")
gambler_states <- c("$0", "$1", "$2", "$3")
P_gambler <- matrix(c(
  1.0, 0.0, 0.0, 0.0,  # $0 is absorbing
  0.5, 0.0, 0.5, 0.0,  # From $1, equal chance of going to $0 or $2
  0.0, 0.5, 0.0, 0.5,  # From $2, equal chance of going to $1 or $3
  0.0, 0.0, 0.0, 1.0   # $3 is absorbing
), nrow = 4, byrow = TRUE)
dimnames(P_gambler) <- list(from = gambler_states, to = gambler_states)
cat("\nTransition matrix for Gambler's Ruin:\n")
print(P_gambler)

# Identifying absorbing states
cat("\nIdentifying absorbing states (p_ii = 1):\n")
is_absorbing <- diag(P_gambler) == 1
absorbing_states <- gambler_states[is_absorbing]
cat("Absorbing states:", absorbing_states, "\n")

# ============================================================================
# 6. MEAN FIRST PASSAGE AND RECURRENCE TIMES
# ============================================================================
cat("\n\n======= MEAN FIRST PASSAGE AND RECURRENCE TIMES =======\n")
cat("Mean first passage time m_ij is the expected number of steps to reach state j from state i.\n")
cat("Mean recurrence time m_ii is the expected return time to state i starting from i.\n")

# Let's use a simple 3-state example
cat("\nExample: Simple 3-state Markov chain\n")
simple_states <- c("A", "B", "C")
P_simple <- matrix(c(
  0.3, 0.6, 0.1,
  0.4, 0.2, 0.4,
  0.1, 0.5, 0.4
), nrow = 3, byrow = TRUE)
dimnames(P_simple) <- list(from = simple_states, to = simple_states)
print(P_simple)

cat("\nCalculating mean first passage times using fundamental matrix method:\n")
cat("For this calculation, we need to solve systems of linear equations.\n")

# Step 1: Find the steady state
A <- t(diag(3) - P_simple)
A[3,] <- rep(1, 3)
b <- c(0, 0, 1)
pi_simple <- solve(A, b)
names(pi_simple) <- simple_states
cat("\nSteady state distribution:\n")
print(round(pi_simple, 4))

# Step 2: Calculate mean recurrence times
# For a recurrent irreducible Markov chain, mean recurrence time = 1/π_i
recurrence_times <- 1 / pi_simple
cat("\nMean recurrence times (expected return time to each state):\n")
print(round(recurrence_times, 2))

cat("\nFor complete first passage time calculations, we would need to solve:\n")
cat("m_ij = 1 + Σ_k≠j p_ik * m_kj for all i≠j\n")
cat("This requires solving systems of linear equations.\n")

# ============================================================================
# 7. ABSORPTION PROBABILITIES AND EXPECTED TIME TO ABSORPTION
# ============================================================================
cat("\n\n======= ABSORPTION PROBABILITIES AND TIMES =======\n")
cat("In a Markov chain with absorbing states, we're often interested in:\n")
cat("1. The probability of absorption in each absorbing state\n")
cat("2. The expected number of steps before absorption\n")

# Let's use the Gambler's Ruin example
cat("\nFor Gambler's Ruin example:\n")
print(P_gambler)

# Step 1: Identify transient and absorbing states
transient_indices <- which(!is_absorbing)
absorbing_indices <- which(is_absorbing)

# Step 2: Decompose the transition matrix
# P = [ Q  R ]
#     [ 0  I ]
# Where:
# Q = transitions between transient states
# R = transitions from transient to absorbing states
# 0 = transitions from absorbing to transient (always 0)
# I = transitions between absorbing states (identity matrix)

Q <- P_gambler[transient_indices, transient_indices, drop = FALSE]
R <- P_gambler[transient_indices, absorbing_indices, drop = FALSE]

cat("\nQ matrix (transitions between transient states):\n")
print(Q)
cat("\nR matrix (transitions from transient to absorbing states):\n")
print(R)

# Step 3: Calculate the fundamental matrix N = (I-Q)^(-1)
N <- solve(diag(nrow(Q)) - Q)
cat("\nFundamental matrix N = (I-Q)^(-1):\n")
print(round(N, 4))

# Step 4: Calculate absorption probabilities B = N*R
B <- N %*% R
rownames(B) <- gambler_states[transient_indices]
colnames(B) <- gambler_states[absorbing_indices]
cat("\nAbsorption probabilities (probability of ending in each absorbing state):\n")
print(round(B, 4))

# Step 5: Calculate expected number of steps to absorption
t <- N %*% rep(1, ncol(N))
names(t) <- gambler_states[transient_indices]
cat("\nExpected number of steps before absorption from each transient state:\n")
print(round(t, 2))

# ============================================================================
# 8. LONG-RUN BEHAVIOR OF MARKOV CHAINS
# ============================================================================
cat("\n\n======= LONG-RUN BEHAVIOR OF MARKOV CHAINS =======\n")
cat("The long-run behavior depends on the chain's structure:\n")
cat("- Irreducible chains: Every state can reach every other state\n")
cat("- Aperiodic chains: Returns do not occur at fixed intervals\n")
cat("- Ergodic chains: Both irreducible and aperiodic\n\n")
cat("For ergodic chains, P^n converges to a matrix with identical rows (steady state)\n")
cat("For chains with absorbing states, the process eventually gets trapped in those states\n")

# Example of convergence to steady state
cat("\nExample: Convergence of weather model to steady state\n")
cat("Initial state: Starting in state 'Sunny'\n")
initial_state <- c(1, 0, 0)  # Start in state 'Sunny'
names(initial_state) <- weather_states

# Track evolution over time
cat("\nEvolution of state probabilities over time:\n")
state_t <- initial_state
cat("t=0:", round(state_t, 4), "\n")

for (t in 1:20) {
  state_t <- state_t %*% P_weather
  if (t %in% c(1, 2, 3, 5, 10, 20)) {
    cat("t=", t, ": ", round(state_t, 4), "\n", sep="")
  }
}

cat("\nSteady state for comparison:", round(steady_state, 4), "\n")
cat("Notice how the distribution converges to the steady state regardless of initial state.\n")

# ============================================================================
# 9. REDUCIBLE MARKOV CHAINS
# ============================================================================
cat("\n\n======= REDUCIBLE MARKOV CHAINS =======\n")
cat("A Markov chain is reducible if the state space can be divided into multiple classes\n")
cat("where transitions between some classes are impossible.\n")

# Example of a reducible Markov chain
cat("\nExample: Reducible Markov chain with two communication classes\n")
reducible_states <- c("A", "B", "C", "D", "E")
P_reducible <- matrix(c(
  0.6, 0.4, 0.0, 0.0, 0.0,  # States A,B form one class
  0.2, 0.8, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.5, 0.3, 0.2,  # States C,D,E form another class
  0.0, 0.0, 0.4, 0.4, 0.2,
  0.0, 0.0, 0.1, 0.2, 0.7
), nrow = 5, byrow = TRUE)
dimnames(P_reducible) <- list(from = reducible_states, to = reducible_states)
cat("\nTransition matrix for reducible chain:\n")
print(P_reducible)

cat("\nIn this example, states {A,B} and {C,D,E} form separate communication classes.\n")
cat("Once the process enters one class, it cannot move to the other class.\n")
cat("Each class will have its own steady-state distribution.\n")

# ============================================================================
# 10. APPLICATION AND PRACTICE EXAMPLE: PageRank Algorithm
# ============================================================================
cat("\n\n======= APPLICATION EXAMPLE: SIMPLIFIED PAGERANK =======\n")
cat("PageRank is a famous application of Markov chains used by Google to rank web pages.\n")
cat("We'll implement a simplified version for a small network of 4 web pages.\n")

# Web page network with transition matrix
pages <- c("Page A", "Page B", "Page C", "Page D")
links <- matrix(c(
  0.0, 0.5, 0.5, 0.0,  # Page A links to B and C
  0.3, 0.0, 0.3, 0.4,  # Page B links to A, C and D
  0.1, 0.8, 0.0, 0.1,  # Page C links to A, B and D
  0.5, 0.2, 0.3, 0.0   # Page D links to A, B and C
), nrow = 4, byrow = TRUE)
dimnames(links) <- list(from = pages, to = pages)
cat("\nWeb page link structure (transition matrix):\n")
print(links)

# In PageRank, we add a "damping factor" (typically 0.85) to handle dead ends
# and add random jumps to any page
cat("\nApplying damping factor (d=0.85) to create Google matrix\n")
damping <- 0.85
n <- length(pages)
teleport <- matrix(1/n, nrow = n, ncol = n)
G <- damping * links + (1 - damping) * teleport
cat("\nGoogle matrix with damping:\n")
print(round(G, 3))

# Find the steady state (PageRank) by power iteration
cat("\nCalculating PageRank using power iteration\n")
pagerank <- rep(1/n, n)  # Start with uniform distribution
names(pagerank) <- pages

for (i in 1:50) {
  pagerank_new <- pagerank %*% G
  # Check for convergence
  if (max(abs(pagerank_new - pagerank)) < 1e-6) {
    cat("Converged after", i, "iterations\n")
    break
  }
  pagerank <- pagerank_new
}

cat("\nPageRank values (importance of each page):\n")
print(round(pagerank, 4))
cat("\nPages sorted by importance:\n")
print(sort(pagerank, decreasing = TRUE))

cat("\n\n======= END OF MARKOV CHAIN EXAMPLES =======\n")
cat("This file demonstrates key concepts of Markov chains using only built-in R functions.\n")
cat("For more advanced functionality, consider packages like 'markovchain' or 'DTMCPack'.\n")
