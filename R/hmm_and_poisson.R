# Hidden Markov Models and Continuous-Time Markov Processes
# Using only built-in R functions (base R)

# ============================================================================
# 1. HIDDEN MARKOV MODELS (HMM) - INTRODUCTION
# ============================================================================
cat("\n======= HIDDEN MARKOV MODELS =======\n")
cat("A Hidden Markov Model (HMM) consists of:\n")
cat("1. A set of hidden states following a Markov process\n")
cat("2. Observable outputs (emissions) dependent on the hidden states\n")
cat("3. Transition probabilities between hidden states\n")
cat("4. Emission probabilities for observations given each state\n")

# Example: Weather as hidden states (Sunny, Cloudy, Rainy) 
# with observable outcomes of ice cream sales (High, Medium, Low)
cat("\nExample: Weather (hidden) affects Ice Cream Sales (observable)\n")
weather_states <- c("Sunny", "Cloudy", "Rainy")
sales_observations <- c("High", "Medium", "Low")

# ============================================================================
# 2. TRANSITION AND EMISSION PROBABILITIES
# ============================================================================
cat("\n\n======= TRANSITION AND EMISSION PROBABILITIES =======\n")

# Transition probability matrix A (weather transitions)
A <- matrix(c(
  0.7, 0.2, 0.1,  # From Sunny: P(S->S)=0.7, P(S->C)=0.2, P(S->R)=0.1
  0.3, 0.5, 0.2,  # From Cloudy: P(C->S)=0.3, P(C->C)=0.5, P(C->R)=0.2
  0.2, 0.4, 0.4   # From Rainy: P(R->S)=0.2, P(R->C)=0.4, P(R->R)=0.4
), nrow = 3, byrow = TRUE)
dimnames(A) <- list(from = weather_states, to = weather_states)

cat("Transition probability matrix (for hidden weather states):\n")
print(A)

# Emission probability matrix B (weather -> ice cream sales)
B <- matrix(c(
  0.6, 0.3, 0.1,  # Sunny: P(High)=0.6, P(Medium)=0.3, P(Low)=0.1
  0.3, 0.4, 0.3,  # Cloudy: P(High)=0.3, P(Medium)=0.4, P(Low)=0.3
  0.1, 0.3, 0.6   # Rainy: P(High)=0.1, P(Medium)=0.3, P(Low)=0.6
), nrow = 3, byrow = TRUE)
dimnames(B) <- list(state = weather_states, observation = sales_observations)

cat("\nEmission probability matrix (sales observations given weather):\n")
print(B)

# Initial state distribution π
pi <- c(0.4, 0.4, 0.2)  # Initially: 40% Sunny, 40% Cloudy, 20% Rainy
names(pi) <- weather_states
cat("\nInitial state distribution:\n")
print(pi)

# ============================================================================
# 3. SIMULATION OF HIDDEN MARKOV MODEL
# ============================================================================
cat("\n\n======= SIMULATION OF HMM =======\n")
cat("Simulating a sequence of hidden states and observations.\n")

# Function to simulate a sequence from an HMM
simulate_hmm <- function(A, B, pi, n_steps) {
  states <- numeric(n_steps)
  observations <- numeric(n_steps)
  
  # Sample initial state based on initial distribution
  states[1] <- sample(1:length(pi), 1, prob = pi)
  
  # Sample observation based on state's emission probabilities
  observations[1] <- sample(1:ncol(B), 1, prob = B[states[1], ])
  
  # Generate the remaining states and observations
  for (t in 2:n_steps) {
    # Sample next state based on transition from current state
    states[t] <- sample(1:nrow(A), 1, prob = A[states[t-1], ])
    
    # Sample observation based on new state's emission probabilities
    observations[t] <- sample(1:ncol(B), 1, prob = B[states[t], ])
  }
  
  return(list(
    states = states,
    observations = observations
  ))
}

# Run simulation for 10 days
set.seed(123)  # For reproducibility
n_days <- 10
simulation <- simulate_hmm(A, B, pi, n_days)

# Convert numeric states and observations to their names
state_names <- weather_states[simulation$states]
observation_names <- sales_observations[simulation$observations]

# Display the results
cat("\nSimulation results for", n_days, "days:\n")
cat("Day  Hidden State  Observation\n")
cat("---  ------------  -----------\n")
for (i in 1:n_days) {
  cat(sprintf("%3d  %-12s  %s\n", i, state_names[i], observation_names[i]))
}

# ============================================================================
# 4. FORWARD ALGORITHM (FILTERING)
# ============================================================================
cat("\n\n======= FORWARD ALGORITHM FOR HMM =======\n")
cat("The forward algorithm calculates P(observations) and can compute\n")
cat("the probability of being in each state given the observations so far.\n")

# Forward algorithm implementation
forward_algorithm <- function(A, B, pi, observations) {
  n_states <- nrow(A)
  n_steps <- length(observations)
  
  # Initialize alpha matrix (forward probabilities)
  alpha <- matrix(0, nrow = n_states, ncol = n_steps)
  
  # Compute alpha for t=1 (initialization)
  for (i in 1:n_states) {
    alpha[i, 1] <- pi[i] * B[i, observations[1]]
  }
    # Compute alpha for t=2,...,T (induction)
  for (t in 2:n_steps) {
    for (j in 1:n_states) {
      # Sum over all possible previous states
      sum_val <- 0
      for (i in 1:n_states) {
        sum_val <- sum_val + alpha[i, t-1] * A[i, j]
      }
      alpha[j, t] <- sum_val * B[j, observations[t]]
    }
  }
  
  # Total probability of the observation sequence
  prob_observations <- sum(alpha[, n_steps])
  
  return(list(
    alpha = alpha,
    prob_observations = prob_observations
  ))
}

# Let's use a sequence of observations to demonstrate
# 1=High, 2=Medium, 3=Low
observation_sequence <- simulation$observations
cat("\nObservation sequence for forward algorithm:\n")
cat(sales_observations[observation_sequence], "\n")

# Run forward algorithm
forward_results <- forward_algorithm(A, B, pi, observation_sequence)

# Calculate state probabilities at each time step
state_probs <- t(apply(forward_results$alpha, 2, function(x) x / sum(x)))
colnames(state_probs) <- weather_states

cat("\nProbability of observations:", forward_results$prob_observations, "\n")
cat("\nFiltered state probabilities (what is the weather given sales so far):\n")
cat("Day  P(Sunny)  P(Cloudy)  P(Rainy)\n")
cat("---  --------  ---------  --------\n")
for (t in 1:n_days) {
  cat(sprintf("%3d  %8.4f  %9.4f  %8.4f\n", 
             t, state_probs[t, 1], state_probs[t, 2], state_probs[t, 3]))
}

# Compare to actual hidden states
cat("\nActual hidden states were:", state_names, "\n")

# ============================================================================
# 5. MARKOV PROCESSES IN CONTINUOUS TIME
# ============================================================================
cat("\n\n======= CONTINUOUS-TIME MARKOV PROCESSES =======\n")
cat("In continuous-time Markov processes, transitions can occur at any time.\n")
cat("The Poisson process is a fundamental continuous-time Markov process.\n")

# ============================================================================
# 6. POISSON PROCESSES
# ============================================================================
cat("\n\n======= POISSON PROCESSES =======\n")
cat("A Poisson process models random events occurring independently at a constant rate λ.\n")
cat("Key properties:\n")
cat("- The number of events in any time interval follows a Poisson distribution\n")
cat("- The waiting time between events follows an exponential distribution\n")
cat("- Events occur independently of each other\n")

# Simulate a Poisson process
lambda <- 3  # Rate parameter: average of 3 events per time unit
time_period <- 10  # Observe for 10 time units

# Simulate the number of events in the time period
n_events <- rpois(1, lambda * time_period)
cat("\nSimulated Poisson process with rate λ =", lambda, "per time unit\n")
cat("Number of events in", time_period, "time units:", n_events, "\n")

# Simulate the event times (arrival times)
# In a Poisson process, arrival times in [0,T] are distributed as ordered uniform random variables
event_times <- sort(runif(n_events, 0, time_period))
cat("\nEvent times:", round(event_times, 2), "\n")

# Calculate inter-arrival times
inter_arrival_times <- diff(c(0, event_times))
cat("\nInter-arrival times:", round(inter_arrival_times, 2), "\n")

# ============================================================================
# 7. DISTRIBUTIONS ASSOCIATED WITH POISSON PROCESSES
# ============================================================================
cat("\n\n======= DISTRIBUTIONS ASSOCIATED WITH POISSON PROCESSES =======\n")

# Poisson Distribution: probability of k events in a time interval
cat("\nPoisson Distribution - Probability of k events in a time interval\n")
k_values <- 0:10
poisson_probs <- dpois(k_values, lambda = lambda)

cat("k (events)  P(X = k)\n")
cat("-----------  --------\n")
for (i in seq_along(k_values)) {
  cat(sprintf("%11d  %8.4f\n", k_values[i], poisson_probs[i]))
}

# Expected number of events in time t
t <- 5
cat("\nExpected number of events in time t =", t, "is λt =", lambda * t, "\n")

# Exponential Distribution: probability density of waiting time between events
cat("\nExponential Distribution - Waiting time between events\n")
cat("Mean waiting time: 1/λ =", 1/lambda, "\n")

# Probability of waiting more than t time units for the next event
t_values <- c(0.5, 1, 1.5, 2, 3)
exp_probs <- pexp(t_values, rate = lambda, lower.tail = FALSE)

cat("t (time)  P(T > t)\n")
cat("---------  --------\n")
for (i in seq_along(t_values)) {
  cat(sprintf("%9.1f  %8.4f\n", t_values[i], exp_probs[i]))
}

# ============================================================================
# 8. APPLICATION EXAMPLE: QUEUEING SYSTEM
# ============================================================================
cat("\n\n======= APPLICATION: SIMPLE QUEUEING SYSTEM =======\n")
cat("A basic M/M/1 queue with Poisson arrivals and exponential service times\n")

# Parameters
arrival_rate <- 2    # Customers arrive at rate of 2 per hour
service_rate <- 3    # Server can handle 3 customers per hour
time_horizon <- 8    # Simulate for 8 hours

cat("Arrival rate (λ):", arrival_rate, "customers per hour\n")
cat("Service rate (μ):", service_rate, "customers per hour\n")

# Theoretical results from queueing theory
rho <- arrival_rate / service_rate  # Traffic intensity
L <- rho / (1 - rho)                # Expected number of customers in system
W <- 1 / (service_rate - arrival_rate) # Expected time in system
Lq <- L - rho                       # Expected queue length
Wq <- W - 1/service_rate            # Expected waiting time
Ls <- L - Lq                        # Expected number of customers in service
Ws <- W - Wq                        # Expected time in service

cat("\nTraffic intensity (ρ) =", rho, "\n")
cat("Expected number in system (L) =", L, "customers\n")
cat("Expected time in system (W) =", W, "hours\n")
cat("Expected queue length (Lq) =", Lq, "customers\n")
cat("Expected waiting time (Wq) =", Wq, "hours\n")

# Simulation
cat("\nSimulating the queue for", time_horizon, "hours...\n")

# Generate customer arrivals (Poisson process)
n_arrivals <- rpois(1, arrival_rate * time_horizon)
arrival_times <- sort(runif(n_arrivals, 0, time_horizon))

# Generate service times (Exponential distribution)
service_times <- rexp(n_arrivals, service_rate)

# Calculate departure times and waiting times
departure_times <- numeric(n_arrivals)
waiting_times <- numeric(n_arrivals)
queue_length <- numeric(n_arrivals)

# Process the first customer
departure_times[1] <- arrival_times[1] + service_times[1]
waiting_times[1] <- 0  # First customer doesn't wait
queue_length[1] <- 0   # No one in queue when first customer arrives

# Process remaining customers
for (i in 2:n_arrivals) {
  # Customer i can only be served after customer i-1 departs
  service_start <- max(arrival_times[i], departure_times[i-1])
  waiting_times[i] <- service_start - arrival_times[i]
  departure_times[i] <- service_start + service_times[i]
  
  # Count customers in queue when this customer arrives
  queue_length[i] <- sum(arrival_times <= arrival_times[i] & 
                        departure_times > arrival_times[i]) - 1
}

# Calculate simulation results
avg_waiting_time <- mean(waiting_times)
avg_time_in_system <- mean(departure_times - arrival_times)
avg_queue_length <- mean(queue_length)
utilization <- sum(service_times) / time_horizon

cat("\nSimulation Results:\n")
cat("Number of customers:", n_arrivals, "\n")
cat("Average waiting time:", round(avg_waiting_time, 3), "hours\n")
cat("Average time in system:", round(avg_time_in_system, 3), "hours\n")
cat("Average queue length:", round(avg_queue_length, 3), "customers\n")
cat("Server utilization:", round(utilization, 3), "\n")

cat("\n\n======= END OF HMM AND CONTINUOUS-TIME MARKOV EXAMPLES =======\n")
cat("This file demonstrates Hidden Markov Models and Continuous-Time Markov processes\n")
cat("using only built-in R functions.\n")
