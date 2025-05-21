# Queueing Theory in R
# Using only built-in R functions (base R)

# ============================================================================
# 1. INTRODUCTION TO QUEUEING THEORY
# ============================================================================
cat("\n======= QUEUEING THEORY INTRODUCTION =======\n")
cat("Queueing theory is the mathematical study of waiting lines or queues.\n")
cat("It analyzes the behavior of systems that provide service to randomly arriving customers.\n")
cat("Applications include call centers, computer systems, traffic flow, and healthcare.\n")

# ============================================================================
# 2. ARRIVAL PROCESS
# ============================================================================
cat("\n\n======= ARRIVAL PROCESS =======\n")
cat("The arrival process describes how customers enter the system.\n")
cat("Key characteristics include:\n")
cat("- Arrival rate (λ): Average number of arrivals per unit time\n")
cat("- Inter-arrival time distribution: Time between consecutive arrivals\n")
cat("- Common models: Poisson arrivals, deterministic arrivals, general distributions\n")

# Poisson arrival process (most common model)
cat("\nPoisson Arrival Process:\n")
cat("- Arrivals occur randomly and independently\n")
cat("- The number of arrivals in time interval t follows Poisson(λt) distribution\n")
cat("- Inter-arrival times follow Exponential(λ) distribution\n")

# Demonstrate Poisson arrivals
lambda <- 5  # Arrival rate: 5 customers per hour
hours <- 8   # Observe for 8 hours

# Expected number of arrivals in 8 hours
expected_arrivals <- lambda * hours
cat("\nWith arrival rate λ =", lambda, "per hour:\n")
cat("Expected number of arrivals in", hours, "hours:", expected_arrivals, "\n")

# Generate simulated number of arrivals
set.seed(123)  # For reproducibility
n_arrivals <- rpois(1, lambda * hours)
cat("Simulated number of arrivals:", n_arrivals, "\n")

# Simulate inter-arrival times (exponential distribution)
inter_arrival_times <- rexp(n_arrivals, lambda)
cat("First 5 inter-arrival times (hours):", round(head(inter_arrival_times, 5), 3), "...\n")

# Calculate arrival times
arrival_times <- cumsum(inter_arrival_times)
cat("First 5 arrival times (hours):", round(head(arrival_times, 5), 3), "...\n")

# ============================================================================
# 3. SERVICE PROCESS
# ============================================================================
cat("\n\n======= SERVICE PROCESS =======\n")
cat("The service process describes how customers are served.\n")
cat("Key characteristics include:\n")
cat("- Service rate (μ): Average number of customers served per unit time\n")
cat("- Service time distribution: Time to serve each customer\n")
cat("- Number of servers: Single or multiple parallel servers\n")

# Common service time distributions
cat("\nCommon service time distributions:\n")
cat("- Exponential: Completely random service times (memoryless)\n")
cat("- Deterministic: Fixed service times\n")
cat("- General: Any arbitrary distribution\n")

# Exponential service times (common model)
mu <- 6  # Service rate: 6 customers per hour

cat("\nWith service rate μ =", mu, "per hour:\n")
cat("Average service time: 1/μ =", round(1/mu, 3), "hours\n")

# Generate simulated service times
service_times <- rexp(n_arrivals, mu)
cat("First 5 service times (hours):", round(head(service_times, 5), 3), "...\n")

# ============================================================================
# 4. QUEUE DISCIPLINE
# ============================================================================
cat("\n\n======= QUEUE DISCIPLINE =======\n")
cat("Queue discipline defines the rules for selecting customers for service.\n")
cat("Common queue disciplines include:\n")
cat("- FIFO (First-In-First-Out): Serve in order of arrival (most common)\n")
cat("- LIFO (Last-In-First-Out): Serve most recent arrival first\n")
cat("- SIRO (Service-In-Random-Order): Serve randomly from the queue\n")
cat("- Priority: Serve based on assigned priorities\n")
cat("- Processor Sharing: All customers receive service simultaneously\n")

cat("\nIn this demonstration, we will use FIFO discipline for all models.\n")

# ============================================================================
# 5. MODELLING ARRIVAL TIME
# ============================================================================
cat("\n\n======= MODELLING ARRIVAL TIME =======\n")
cat("Arrival time modelling involves choosing appropriate distributions.\n")

cat("\nPoisson Process - Mathematical properties:\n")
cat("1. The number of arrivals N(t) in time [0,t] follows Poisson(λt)\n")
cat("2. Inter-arrival times T follow Exponential(λ)\n")
cat("3. P(N(t) = n) = e^(-λt) * (λt)^n / n!\n")
cat("4. E[N(t)] = Var[N(t)] = λt\n")
cat("5. E[T] = 1/λ, Var[T] = 1/λ^2\n")

# Demonstrate properties of Poisson process
t <- 2  # Time interval (2 hours)
n_values <- 0:15  # Number of arrivals to consider

# Calculate probability of n arrivals in t hours
arrival_probs <- dpois(n_values, lambda * t)
df_arrivals <- data.frame(
  n = n_values,
  probability = arrival_probs
)

cat("\nProbability of n arrivals in", t, "hours (λ =", lambda, "):\n")
for (i in 1:10) {  # Show first 10 values
  cat(sprintf("P(N(%d) = %d) = %.4f\n", t, df_arrivals$n[i], df_arrivals$probability[i]))
}

# Verify mean and variance
expected_mean <- lambda * t
expected_var <- lambda * t
cat("\nTheoretical mean number of arrivals in", t, "hours:", expected_mean, "\n")
cat("Theoretical variance of arrivals in", t, "hours:", expected_var, "\n")

# ============================================================================
# 6. KENDALL-LEE NOTATION
# ============================================================================
cat("\n\n======= KENDALL-LEE NOTATION =======\n")
cat("Kendall-Lee notation is a standard classification system for queueing models.\n")
cat("It has the form A/B/C/K/N/D where:\n")
cat("- A: Arrival process distribution\n")
cat("- B: Service time distribution\n")
cat("- C: Number of servers\n")
cat("- K: System capacity (max customers in system, default = ∞)\n")
cat("- N: Population size (default = ∞)\n")
cat("- D: Queue discipline (default = FIFO)\n")

cat("\nCommon distribution codes:\n")
cat("- M: Markovian (exponential distribution / Poisson process)\n")
cat("- D: Deterministic (constant times)\n")
cat("- G: General (arbitrary distribution)\n")
cat("- Ek: Erlang distribution with parameter k\n")

cat("\nCommon queueing models:\n")
cat("- M/M/1: Single server with Poisson arrivals and exponential service times\n")
cat("- M/M/c: c servers with Poisson arrivals and exponential service times\n")
cat("- M/G/1: Single server with Poisson arrivals and general service times\n")
cat("- G/G/1: Single server with general arrivals and service times\n")

# ============================================================================
# 7. BIRTH-DEATH PROCESS
# ============================================================================
cat("\n\n======= BIRTH-DEATH PROCESS =======\n")
cat("A birth-death process is a continuous-time Markov chain where:\n")
cat("- The system can only move to adjacent states\n")
cat("- State n → n+1 is a 'birth' with rate λn\n")
cat("- State n → n-1 is a 'death' with rate μn\n")
cat("Many queueing systems can be modeled as birth-death processes.\n")

cat("\nFor a simple queueing model with constant rates λ and μ:\n")
cat("- Birth = customer arrival (with rate λ)\n")
cat("- Death = service completion (with rate μ)\n")
cat("- State n = number of customers in the system\n")

# Demonstrate state transition diagram
n_states <- 5  # Number of states to display (0, 1, 2, ..., n_states-1)
cat("\nState transition rates for the first", n_states, "states:\n")
for (i in 0:(n_states-2)) {
  cat(sprintf("State %d → State %d: rate λ = %.1f\n", i, i+1, lambda))
}
for (i in 1:(n_states-1)) {
  cat(sprintf("State %d → State %d: rate μ = %.1f\n", i, i-1, mu))
}

# ============================================================================
# 8. M/M/1 PROCESS
# ============================================================================
cat("\n\n======= M/M/1 QUEUEING PROCESS =======\n")
cat("The M/M/1 queue has:\n")
cat("- Poisson arrivals (rate λ)\n")
cat("- Exponential service times (rate μ)\n")
cat("- Single server\n")
cat("- Infinite capacity and population\n")
cat("- FIFO discipline\n")

# Check stability condition
rho <- lambda / mu  # Traffic intensity
cat("\nTraffic intensity ρ = λ/μ =", rho, "\n")

if (rho >= 1) {
  cat("Warning: System is unstable (ρ ≥ 1). Queue will grow indefinitely.\n")
} else {
  cat("System is stable (ρ < 1). A steady state exists.\n")
}

# ============================================================================
# 9. STEADY STATE PROBABILITY IN M/M/1 PROCESS
# ============================================================================
cat("\n\n======= STEADY STATE PROBABILITY IN M/M/1 =======\n")
cat("In steady state, the probability of having n customers in an M/M/1 system is:\n")
cat("P(n) = (1 - ρ) * ρ^n\n")
cat("where ρ = λ/μ is the traffic intensity\n")

# Calculate steady-state probabilities
max_n <- 10  # Calculate for n = 0, 1, 2, ..., max_n
p_n <- (1 - rho) * rho^(0:max_n)

cat("\nSteady-state probabilities for the first", max_n+1, "states:\n")
for (n in 0:max_n) {
  cat(sprintf("P(%d) = %.4f\n", n, p_n[n+1]))
}

# Plot cumulative probability
cum_prob <- cumsum(p_n)
cat("\nCumulative probability for the first", max_n+1, "states:\n")
for (n in 0:max_n) {
  cat(sprintf("P(≤ %d) = %.4f\n", n, cum_prob[n+1]))
}

# ============================================================================
# 10. AVERAGE NUMBER OF CUSTOMERS AND EXPECTED TIME IN M/M/1
# ============================================================================
cat("\n\n======= AVERAGE MEASURES IN M/M/1 =======\n")
cat("Key performance measures for M/M/1 queues:\n")

# Calculate performance measures
L <- rho / (1 - rho)           # Average number in system
Lq <- rho^2 / (1 - rho)        # Average number in queue
W <- 1 / (mu - lambda)         # Average time in system
Wq <- rho / (mu - lambda)      # Average waiting time in queue

cat("\nTheoretical performance measures (λ =", lambda, ", μ =", mu, "):\n")
cat("L = Average number of customers in system =", round(L, 3), "\n")
cat("Lq = Average number of customers in queue =", round(Lq, 3), "\n")
cat("W = Average time in system =", round(W, 3), "hours\n")
cat("Wq = Average waiting time in queue =", round(Wq, 3), "hours\n")

# Verify Little's Law: L = λW and Lq = λWq
cat("\nVerifying Little's Law:\n")
cat("L = λW:", round(L, 3), "=", round(lambda * W, 3), "\n")
cat("Lq = λWq:", round(Lq, 3), "=", round(lambda * Wq, 3), "\n")

# ============================================================================
# 11. SIMULATION OF M/M/1 QUEUE
# ============================================================================
cat("\n\n======= M/M/1 QUEUE SIMULATION =======\n")
cat("Let's simulate an M/M/1 queue and compare with theoretical results.\n")

# Parameters
sim_time <- 1000  # Simulation time (hours)
set.seed(456)     # For reproducibility

# Generate arrivals
n_arrivals_sim <- rpois(1, lambda * sim_time)
arrival_times_sim <- sort(runif(n_arrivals_sim, 0, sim_time))
service_times_sim <- rexp(n_arrivals_sim, mu)

# Process the queue
departure_times <- numeric(n_arrivals_sim)
waiting_times <- numeric(n_arrivals_sim)
system_times <- numeric(n_arrivals_sim)

# First customer
departure_times[1] <- arrival_times_sim[1] + service_times_sim[1]
waiting_times[1] <- 0  # First customer doesn't wait
system_times[1] <- service_times_sim[1]

# Remaining customers
for (i in 2:n_arrivals_sim) {
  # Start service after arrival or after previous customer departs
  service_start <- max(arrival_times_sim[i], departure_times[i-1])
  waiting_times[i] <- service_start - arrival_times_sim[i]
  departure_times[i] <- service_start + service_times_sim[i]
  system_times[i] <- departure_times[i] - arrival_times_sim[i]
}

# Calculate simulation statistics
avg_waiting_time <- mean(waiting_times)
avg_system_time <- mean(system_times)

# Count customers in system at various times
sample_times <- seq(10, sim_time, by = 10)  # Sample every 10 time units
system_counts <- numeric(length(sample_times))

for (i in 1:length(sample_times)) {
  t <- sample_times[i]
  system_counts[i] <- sum(arrival_times_sim <= t & departure_times > t)
}

avg_system_count <- mean(system_counts)
avg_queue_length <- mean(system_counts - 1) # Subtract customer in service
avg_queue_length <- max(0, avg_queue_length) # Ensure non-negative

cat("\nSimulation results (", n_arrivals_sim, "customers over", sim_time, "hours):\n")
cat("Average waiting time (Wq):", round(avg_waiting_time, 3), "hours\n")
cat("Average time in system (W):", round(avg_system_time, 3), "hours\n")
cat("Average number in system (L):", round(avg_system_count, 3), "\n")
cat("Average queue length (Lq):", round(avg_queue_length, 3), "\n")

cat("\nComparison with theoretical values:\n")
cat(sprintf("%-30s %-15s %-15s\n", "Measure", "Theoretical", "Simulation"))
cat(sprintf("%-30s %-15.3f %-15.3f\n", "Waiting time (Wq)", Wq, avg_waiting_time))
cat(sprintf("%-30s %-15.3f %-15.3f\n", "Time in system (W)", W, avg_system_time))
cat(sprintf("%-30s %-15.3f %-15.3f\n", "Number in system (L)", L, avg_system_count))
cat(sprintf("%-30s %-15.3f %-15.3f\n", "Queue length (Lq)", Lq, avg_queue_length))

# ============================================================================
# 12. EFFECT OF UTILIZATION ON QUEUE PERFORMANCE
# ============================================================================
cat("\n\n======= EFFECT OF UTILIZATION ON QUEUE PERFORMANCE =======\n")
cat("As utilization (ρ = λ/μ) approaches 1, queue performance deteriorates rapidly.\n")

# Calculate performance measures for different utilization levels
rho_values <- c(0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
L_values <- rho_values / (1 - rho_values)
Lq_values <- rho_values^2 / (1 - rho_values)
W_values <- 1 / (mu * (1 - rho_values))
Wq_values <- rho_values / (mu * (1 - rho_values))

# Display results in a table
cat("\nPerformance measures for different utilization levels (μ =", mu, "):\n")
cat(sprintf("%-8s %-12s %-12s %-12s %-12s %-12s\n", 
          "ρ", "λ", "L", "Lq", "W", "Wq"))
cat(sprintf("%-8s %-12s %-12s %-12s %-12s %-12s\n", 
          "", "arrivals/hr", "customers", "customers", "hours", "hours"))
cat(rep("-", 70), "\n")

for (i in 1:length(rho_values)) {
  rho <- rho_values[i]
  lambda_i <- rho * mu
  cat(sprintf("%-8.2f %-12.2f %-12.3f %-12.3f %-12.3f %-12.3f\n", 
            rho, lambda_i, L_values[i], Lq_values[i], W_values[i], Wq_values[i]))
}

cat("\nNotice how waiting times and queue lengths grow exponentially as ρ approaches 1.\n")
cat("This is why high-utilization systems often experience severe congestion.\n")

cat("\n\n======= END OF QUEUEING THEORY EXAMPLES =======\n")
cat("This file demonstrates fundamental concepts of queueing theory\n")
cat("using only built-in R functions.\n")
