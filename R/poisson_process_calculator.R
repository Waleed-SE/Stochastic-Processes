# POISSON PROCESS CALCULATOR
# This file provides specialized functions for working with Poisson processes
# Using only built-in R functions (base R)

# ============================================================================
# 1. INTRODUCTION TO POISSON PROCESSES
# ============================================================================
cat("\n======= POISSON PROCESS CALCULATOR =======\n")
cat("A Poisson process is a stochastic process that counts the number of events\n")
cat("occurring in a given time interval, where events occur continuously and\n")
cat("independently at a constant average rate (λ).\n")
cat("\nThis calculator provides functions for working with Poisson processes.\n")

# ============================================================================
# 2. PROBABILITY OF EVENTS IN TIME PERIOD
# ============================================================================

# Function to calculate probability of exactly n events in time period t with rate lambda
poisson_probability <- function(n, lambda, t) {
  # n: number of events
  # lambda: rate parameter (events per unit time)
  # t: time period
  
  # Expected number of events in time period t
  expected_events <- lambda * t
  
  # Probability of exactly n events
  prob <- dpois(n, lambda = expected_events)
  
  return(prob)
}

# Function to calculate probability of at most n events in time period t
poisson_probability_at_most <- function(n, lambda, t) {
  expected_events <- lambda * t
  prob <- ppois(n, lambda = expected_events)
  return(prob)
}

# Function to calculate probability of at least n events in time period t
poisson_probability_at_least <- function(n, lambda, t) {
  expected_events <- lambda * t
  # P(X ≥ n) = 1 - P(X < n) = 1 - P(X ≤ n-1)
  prob <- 1 - ppois(n - 1, lambda = expected_events)
  return(prob)
}

# Function to calculate probability of between n1 and n2 events (inclusive)
poisson_probability_between <- function(n1, n2, lambda, t) {
  expected_events <- lambda * t
  # P(n1 ≤ X ≤ n2) = P(X ≤ n2) - P(X ≤ n1-1)
  prob <- ppois(n2, lambda = expected_events) - ppois(n1 - 1, lambda = expected_events)
  return(prob)
}

# ============================================================================
# 3. WAITING TIME PREDICTIONS
# ============================================================================

# Function to predict next arrival time given the last arrival time
predict_next_arrival <- function(last_arrival, lambda) {
  # last_arrival: time of the last arrival
  # lambda: rate parameter (events per unit time)
  
  # In a Poisson process, inter-arrival times follow an exponential distribution
  # with rate parameter lambda
  inter_arrival_time <- rexp(1, rate = lambda)
  
  next_arrival <- last_arrival + inter_arrival_time
  return(next_arrival)
}

# Function to predict the distribution of waiting time until the next n events
predict_waiting_time <- function(n, lambda) {
  # n: number of events to wait for
  # lambda: rate parameter (events per unit time)
  
  # In a Poisson process, the waiting time until the nth event follows
  # a Gamma distribution with shape=n and rate=lambda
  mean_waiting_time <- n / lambda
  variance <- n / (lambda^2)
  std_dev <- sqrt(variance)
  
  # Return the parameters of the waiting time distribution
  return(list(
    distribution = "Gamma",
    shape = n,
    rate = lambda,
    mean = mean_waiting_time,
    standard_deviation = std_dev
  ))
}

# Function to calculate probability of waiting less than t time units for the next event
prob_waiting_less_than <- function(t, lambda) {
  # P(T < t) = 1 - e^(-lambda*t)
  prob <- 1 - exp(-lambda * t)
  return(prob)
}

# Function to calculate probability of waiting more than t time units for the next event
prob_waiting_more_than <- function(t, lambda) {
  # P(T > t) = e^(-lambda*t)
  prob <- exp(-lambda * t)
  return(prob)
}

# ============================================================================
# 4. DEMONSTRATION OF CALCULATOR FUNCTIONS
# ============================================================================

# Example usage
cat("\n======= EXAMPLE CALCULATIONS =======\n")

# Set parameters for demonstration
lambda <- 3  # Events per hour
time_period <- 2  # Hours

cat("Poisson process with rate λ =", lambda, "events per hour\n")
cat("Time period =", time_period, "hours\n\n")

# Probabilities of different numbers of events
cat("Probability calculations:\n")
cat("------------------------\n")
for (i in 0:10) {
  p <- poisson_probability(i, lambda, time_period)
  cat(sprintf("P(X = %d) = %.4f\n", i, p))
}

cat("\nProbability of at most 5 events: ", 
    poisson_probability_at_most(5, lambda, time_period), "\n")
cat("Probability of at least 5 events: ", 
    poisson_probability_at_least(5, lambda, time_period), "\n")
cat("Probability of between 3 and 7 events: ", 
    poisson_probability_between(3, 7, lambda, time_period), "\n")

# Waiting time information
cat("\nWaiting time information:\n")
cat("------------------------\n")
cat("Mean time until next event: ", 1/lambda, " hours\n")

next_times <- c(0.5, 1, 2)
for (t in next_times) {
  p_less <- prob_waiting_less_than(t, lambda)
  p_more <- prob_waiting_more_than(t, lambda)
  cat(sprintf("P(waiting time < %.1f hours) = %.4f\n", t, p_less))
  cat(sprintf("P(waiting time > %.1f hours) = %.4f\n", t, p_more))
}

# Waiting time for multiple events
waiting_3 <- predict_waiting_time(3, lambda)
cat("\nWaiting time until 3 events:\n")
cat("Mean: ", waiting_3$mean, " hours\n")
cat("Standard deviation: ", waiting_3$standard_deviation, " hours\n")

# ============================================================================
# 5. SIMULATION OF A POISSON PROCESS
# ============================================================================

simulate_poisson_process <- function(lambda, end_time) {
  # lambda: rate parameter (events per unit time)
  # end_time: end of simulation time period
  
  # Expected number of events in the time period
  expected_count <- lambda * end_time
  
  # Generate number of events according to Poisson distribution
  n_events <- rpois(1, expected_count)
  
  # Arrival times are uniformly distributed over [0, end_time]
  # and then sorted to get ordered arrival times
  arrival_times <- sort(runif(n_events, 0, end_time))
  
  # Calculate inter-arrival times
  inter_arrival_times <- diff(c(0, arrival_times))
  
  return(list(
    arrival_times = arrival_times,
    inter_arrival_times = inter_arrival_times,
    n_events = n_events
  ))
}

# Simulate a Poisson process
cat("\n======= SIMULATION EXAMPLE =======\n")
simulation_time <- 5  # hours
cat("Simulating a Poisson process with rate λ =", lambda, "events per hour\n")
cat("Simulation period:", simulation_time, "hours\n\n")

set.seed(42)  # For reproducibility
sim_result <- simulate_poisson_process(lambda, simulation_time)

cat("Number of events:", sim_result$n_events, "\n\n")
cat("Arrival times (hours):\n")
print(round(sim_result$arrival_times, 2))

cat("\nInter-arrival times (hours):\n")
print(round(sim_result$inter_arrival_times, 2))

cat("\nSummarizing inter-arrival times:\n")
cat("Mean:", round(mean(sim_result$inter_arrival_times), 4), 
    "(theoretical: ", round(1/lambda, 4), ")\n")
cat("Std Dev:", round(sd(sim_result$inter_arrival_times), 4),
    "(theoretical: ", round(1/lambda, 4), ")\n")

# Predict next arrival after the last event
if (length(sim_result$arrival_times) > 0) {
  last_arrival <- max(sim_result$arrival_times)
  next_arrival <- predict_next_arrival(last_arrival, lambda)
  cat("\nLast event occurred at:", round(last_arrival, 2), "hours\n")
  cat("Predicted next event at:", round(next_arrival, 2), "hours\n")
  cat("(", round(next_arrival - last_arrival, 2), "hours after the last event)\n")
}

cat("\n======= END OF POISSON PROCESS CALCULATOR =======\n")
cat("This file provides specialized functions for working with Poisson processes.\n")
