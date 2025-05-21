# Mean First Passage Time (MFPT) in Markov Chains
# Using only built-in R functions (no external packages)

# =====================================================================
# INTRODUCTION TO MEAN FIRST PASSAGE TIME
# =====================================================================

cat("\n======= MEAN FIRST PASSAGE TIME IN MARKOV CHAINS =======\n")
cat("The mean first passage time m_ij is the expected number of steps\n")
cat("to reach state j for the first time, starting from state i.\n")
cat("For i = j, we define the mean recurrence time m_ii as the expected\n")
cat("number of steps to return to state i, having started from state i.\n\n")

# =====================================================================
# METHOD 1: DIRECT CALCULATION USING LINEAR EQUATIONS
# =====================================================================

cat("METHOD 1: DIRECT CALCULATION USING LINEAR EQUATIONS\n")
cat("For a Markov chain with transition matrix P, the mean first passage times\n")
cat("satisfy the system of linear equations:\n")
cat("m_ij = 1 + Σ_{k≠j} p_ik * m_kj for all i≠j\n")
cat("For recurrent states, the mean recurrence time m_ii = 1/π_i, where π_i\n")
cat("is the steady-state probability of state i.\n\n")

# Function to calculate mean first passage times directly
calculate_mfpt_direct <- function(P) {
  n <- nrow(P)
  states <- rownames(P)
  if (is.null(states)) states <- 1:n
  
  # Calculate steady state distribution
  # We solve (I - P^T)π = 0 with constraint sum(π) = 1
  A <- t(diag(n) - P)
  A[n, ] <- rep(1, n)
  b <- c(rep(0, n-1), 1)
  pi <- solve(A, b)
  
  # Initialize mean first passage time matrix
  M <- matrix(0, nrow = n, ncol = n)
  
  # Set mean recurrence times (diagonal elements)
  for (i in 1:n) {
    M[i, i] <- 1 / pi[i]
  }
  
  # For each target state j, calculate m_ij for all i≠j
  for (j in 1:n) {
    # Create system of equations for m_ij (i≠j)
    indices <- setdiff(1:n, j)
    n_eq <- length(indices)
    
    # Each equation has form: m_ij = 1 + Σ_{k≠j} p_ik * m_kj
    # Rearranging: Σ_{k≠j} p_ik * m_kj - m_ij = -1
    # This gives us system: A*x = b where x contains the unknowns m_ij
    
    A <- matrix(0, nrow = n_eq, ncol = n_eq)
    b <- rep(-1, n_eq)
    
    for (idx in 1:n_eq) {
      i <- indices[idx]
      # Fill row of coefficient matrix
      for (kdx in 1:n_eq) {
        k <- indices[kdx]
        if (idx == kdx) {
          # Coefficient of m_ij is -1
          A[idx, kdx] <- P[i, k] - 1
        } else {
          # Coefficient of m_kj is p_ik
          A[idx, kdx] <- P[i, k]
        }
      }
      # Add contribution from transitions to state j
      b[idx] <- b[idx] - P[i, j] * 0  # m_jj = 0 for this calculation
    }
    
    # Solve system for m_ij (i≠j)
    m_ij <- solve(A, b)
    
    # Fill in the results
    for (idx in 1:n_eq) {
      i <- indices[idx]
      M[i, j] <- m_ij[idx]
    }
  }
  
  # Set row and column names
  rownames(M) <- states
  colnames(M) <- states
  
  return(list(M = M, pi = pi))
}

# =====================================================================
# METHOD 2: USING THE FUNDAMENTAL MATRIX APPROACH
# =====================================================================

cat("METHOD 2: USING THE FUNDAMENTAL MATRIX APPROACH\n")
cat("For an ergodic Markov chain, we can calculate MFPT using the fundamental matrix Z:\n")
cat("Z = (I - P + W)^(-1), where W has all rows equal to the steady state π\n")
cat("Then m_ij = (z_jj - z_ij) / π_j\n\n")

# Function to calculate mean first passage times using fundamental matrix
calculate_mfpt_fundamental <- function(P) {
  n <- nrow(P)
  states <- rownames(P)
  if (is.null(states)) states <- 1:n
  
  # Calculate steady state distribution
  A <- t(diag(n) - P)
  A[n, ] <- rep(1, n)
  b <- c(rep(0, n-1), 1)
  pi <- solve(A, b)
  
  # Create matrix W with rows equal to π
  W <- matrix(pi, nrow = n, ncol = n, byrow = TRUE)
  
  # Calculate fundamental matrix Z = (I - P + W)^(-1)
  Z <- solve(diag(n) - P + W)
  
  # Calculate mean first passage times
  M <- matrix(0, nrow = n, ncol = n)
  for (i in 1:n) {
    for (j in 1:n) {
      if (i != j) {
        M[i, j] <- (Z[j, j] - Z[i, j]) / pi[j]
      } else {
        M[i, i] <- 1 / pi[i]
      }
    }
  }
  
  # Set row and column names
  rownames(M) <- states
  colnames(M) <- states
  
  return(list(M = M, pi = pi, Z = Z))
}

# =====================================================================
# EXAMPLES AND APPLICATIONS
# =====================================================================

cat("EXAMPLE 1: WEATHER MODEL (SUNNY, CLOUDY, RAINY)\n\n")

# Define transition matrix
weather_states <- c("Sunny", "Cloudy", "Rainy")
P_weather <- matrix(c(
  0.7, 0.2, 0.1,  # Sunny to Sunny, Cloudy, Rainy
  0.3, 0.4, 0.3,  # Cloudy to Sunny, Cloudy, Rainy
  0.2, 0.4, 0.4   # Rainy to Sunny, Cloudy, Rainy
), nrow = 3, byrow = TRUE)
dimnames(P_weather) <- list(from = weather_states, to = weather_states)

cat("Transition Matrix P:\n")
print(P_weather)

# Calculate mean first passage times using both methods
mfpt_direct <- calculate_mfpt_direct(P_weather)
mfpt_fundamental <- calculate_mfpt_fundamental(P_weather)

cat("\nSteady State Distribution:\n")
print(round(mfpt_direct$pi, 4))

cat("\nMean First Passage Times (Method 1 - Direct):\n")
print(round(mfpt_direct$M, 2))

cat("\nMean First Passage Times (Method 2 - Fundamental Matrix):\n")
print(round(mfpt_fundamental$M, 2))

cat("\nInterpretation of Results:\n")
cat("- M[i,j] represents the expected number of steps to reach state j from state i\n")
cat("- The diagonal elements M[i,i] are the mean recurrence times\n")
cat("- For example, starting from Sunny, it takes on average", 
    round(mfpt_direct$M[1, 3], 2), "days until it rains\n")
cat("- Starting from Rainy, it takes on average", 
    round(mfpt_direct$M[3, 1], 2), "days until it's sunny\n")

# =====================================================================
# EXAMPLE 2: GAMBLER'S RUIN PROBLEM
# =====================================================================

cat("\n\nEXAMPLE 2: MODIFIED GAMBLER'S RUIN PROBLEM\n\n")
cat("A gambler starts with $2 and plays a game where:\n")
cat("- With probability 0.4, they win $1\n")
cat("- With probability 0.6, they lose $1\n")
cat("The gambler stops if they reach $0 (broke) or $4 (target).\n\n")

# Define states and transition matrix
gambler_states <- c("$0", "$1", "$2", "$3", "$4")
P_gambler <- matrix(c(
  1.0, 0.0, 0.0, 0.0, 0.0,  # $0 is absorbing
  0.6, 0.0, 0.4, 0.0, 0.0,  # From $1: lose or win
  0.0, 0.6, 0.0, 0.4, 0.0,  # From $2: lose or win
  0.0, 0.0, 0.6, 0.0, 0.4,  # From $3: lose or win
  0.0, 0.0, 0.0, 0.0, 1.0   # $4 is absorbing
), nrow = 5, byrow = TRUE)
dimnames(P_gambler) <- list(from = gambler_states, to = gambler_states)

cat("Transition Matrix P:\n")
print(P_gambler)

# For absorbing chains, we need a different approach for MFPT
# We'll calculate expected time to absorption using the fundamental matrix

cat("\nFor absorbing Markov chains, we calculate expected time to absorption using fundamental matrix:\n")

# Identify transient and absorbing states
is_absorbing <- diag(P_gambler) == 1
transient_indices <- which(!is_absorbing)
absorbing_indices <- which(is_absorbing)

cat("Transient states:", gambler_states[transient_indices], "\n")
cat("Absorbing states:", gambler_states[absorbing_indices], "\n")

# Extract Q (transitions between transient states) and R (transitions to absorbing states)
Q <- P_gambler[transient_indices, transient_indices, drop = FALSE]
R <- P_gambler[transient_indices, absorbing_indices, drop = FALSE]

cat("\nQ matrix (transitions between transient states):\n")
print(Q)

cat("\nR matrix (transitions to absorbing states):\n")
print(R)

# Calculate fundamental matrix N = (I-Q)^(-1)
N <- solve(diag(nrow(Q)) - Q)
rownames(N) <- gambler_states[transient_indices]
colnames(N) <- gambler_states[transient_indices]

cat("\nFundamental matrix N = (I-Q)^(-1):\n")
print(round(N, 4))

# Calculate absorption probabilities B = N*R
B <- N %*% R
rownames(B) <- gambler_states[transient_indices]
colnames(B) <- gambler_states[absorbing_indices]

cat("\nAbsorption probabilities:\n")
print(round(B, 4))

# Calculate expected steps to absorption
t <- N %*% rep(1, ncol(N))
names(t) <- gambler_states[transient_indices]

cat("\nExpected number of steps until absorption:\n")
print(round(t, 2))

cat("\nInterpretation of results for starting with $2:\n")
cat("- Probability of going broke ($0):", round(B[2, 1], 4), "\n")
cat("- Probability of reaching target ($4):", round(B[2, 2], 4), "\n")
cat("- Expected number of plays until either going broke or reaching target:", round(t[2], 2), "\n")

# =====================================================================
# EXAMPLE 3: LAND COVER CHANGE MARKOV MODEL
# =====================================================================

cat("\n\nEXAMPLE 3: LAND COVER CHANGE MODEL\n\n")
cat("A landscape has three types of land cover: Forest, Agriculture, and Urban.\n")
cat("The annual transition probabilities between them are given in the matrix P.\n\n")

# Define states and transition matrix
land_states <- c("Forest", "Agriculture", "Urban")
P_land <- matrix(c(
  0.95, 0.04, 0.01,  # Forest to Forest, Agriculture, Urban
  0.03, 0.92, 0.05,  # Agriculture to Forest, Agriculture, Urban
  0.00, 0.01, 0.99   # Urban to Forest, Agriculture, Urban
), nrow = 3, byrow = TRUE)
dimnames(P_land) <- list(from = land_states, to = land_states)

cat("Transition Matrix P:\n")
print(P_land)

# Calculate MFPT using the fundamental matrix approach
mfpt_land <- calculate_mfpt_fundamental(P_land)

cat("\nSteady State Distribution:\n")
print(round(mfpt_land$pi, 4))

cat("\nMean First Passage Times:\n")
print(round(mfpt_land$M, 2))

cat("\nInterpretation of Results:\n")
cat("- Starting from Forest land, it takes on average", 
    round(mfpt_land$M[1, 3], 2), "years until it becomes Urban\n")
cat("- Starting from Agriculture, it takes on average", 
    round(mfpt_land$M[2, 1], 2), "years until it becomes Forest\n")
cat("- Once Urban, it takes on average", 
    round(mfpt_land$M[3, 1], 2), "years until it returns to Forest\n")
cat("- The mean recurrence time for Forest is", 
    round(mfpt_land$M[1, 1], 2), "years, meaning a Forest patch\n  will change to something else and then back to Forest in that time on average\n")

# =====================================================================
# VISUALIZATION OF MEAN FIRST PASSAGE TIMES
# =====================================================================

cat("\n\nVISUALIZING MEAN FIRST PASSAGE TIMES\n")
cat("Creating a simple visualization of MFPT matrix as a heatmap\n")

# Create a simple text-based visualization function
visualize_matrix <- function(M, title) {
  n <- nrow(M)
  max_val <- max(M)
  cat("\n", title, "\n\n")
  
  # Print column headers
  cat("        ")
  for (j in 1:n) {
    col_name <- substr(colnames(M)[j], 1, 5)
    cat(sprintf("%-7s", col_name))
  }
  cat("\n")
  
  # Print rows
  for (i in 1:n) {
    row_name <- substr(rownames(M)[i], 1, 5)
    cat(sprintf("%-7s", row_name))
    for (j in 1:n) {
      val <- M[i, j]
      intensity <- min(9, max(0, round(9 * val / max_val)))
      if (i == j) {
        cat(sprintf("\033[1m[%5.2f]\033[0m", val))
      } else {
        cat(sprintf(" %5.2f ", val))
      }
    }
    cat("\n")
  }
}

# Visualize the MFPT matrices for our examples
visualize_matrix(round(mfpt_weather <- mfpt_fundamental$M, 2), "Weather Model MFPT Heatmap")
visualize_matrix(round(mfpt_land$M, 2), "Land Cover Model MFPT Heatmap")

# =====================================================================
# COMPUTING FIRST PASSAGE TIME DISTRIBUTION (NOT JUST THE MEAN)
# =====================================================================

cat("\n\nCOMPUTING FIRST PASSAGE TIME DISTRIBUTIONS\n")
cat("Mean first passage time gives only the expected value.\n")
cat("We can also compute the entire probability distribution of the first passage time.\n")

# Function to compute the probability distribution of first passage time
# from state i to state j up to max_steps
compute_fpt_distribution <- function(P, i, j, max_steps) {
  n <- nrow(P)
  
  # f_ij^(k) = Probability that starting from i, the first visit to j occurs at step k
  fpt_dist <- numeric(max_steps)
  
  # Initial case: f_ij^(1) = p_ij
  fpt_dist[1] <- P[i, j]
  
  # For k > 1, we need to exclude paths that visit j before step k
  # f_ij^(k) = sum_{r≠j} (p_ir × f_rj^(k-1))
  
  # Create modified transition matrix where j is absorbing
  P_mod <- P
  P_mod[j, ] <- 0
  P_mod[j, j] <- 1
  
  # Matrix of current k-1 step transitions, excluding paths through j
  P_current <- P_mod
  
  for (k in 2:max_steps) {
    # Move one more step
    P_current <- P_current %*% P_mod
    
    # Probability of reaching j from i in exactly k steps without visiting j before
    fpt_dist[k] <- sum(P[i, -j] * P_current[-j, j])
  }
  
  return(fpt_dist)
}

# Example using the Weather model: First passage from Sunny to Rainy
i <- 1  # Sunny
j <- 3  # Rainy
max_steps <- 20

cat("\nComputing first passage time distribution from Sunny to Rainy:\n")
fpt_dist <- compute_fpt_distribution(P_weather, i, j, max_steps)

# Display results
cat("\nProbability that first rain occurs on day k:\n")
cat("Day   Probability  Cumulative\n")
cum_prob <- 0
for (k in 1:max_steps) {
  cum_prob <- cum_prob + fpt_dist[k]
  stars <- paste(rep("*", round(fpt_dist[k] * 50)), collapse = "")
  if (fpt_dist[k] > 0.001) {
    cat(sprintf("%2d    %5.3f      %5.3f    %s\n", k, fpt_dist[k], cum_prob, stars))
  }
}

# Calculate the expected value from the distribution as a check
expected_fpt <- sum((1:max_steps) * fpt_dist)
cat("\nExpected first passage time from distribution:", round(expected_fpt, 2), "\n")
cat("Mean first passage time from matrix:", round(mfpt_weather[i, j], 2), "\n")

cat("\n\n======= END OF MEAN FIRST PASSAGE TIME EXAMPLES =======\n")
