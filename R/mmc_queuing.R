# M/M/c Queuing Systems in R
# Using only built-in R functions (no external packages)

# =====================================================================
# INTRODUCTION TO M/M/C QUEUING SYSTEMS
# =====================================================================

cat("\n======= M/M/c QUEUING SYSTEMS =======\n")
cat("M/M/c queues have:\n")
cat("- M: Markovian (exponential) arrivals with rate λ\n")
cat("- M: Markovian (exponential) service times with rate μ per server\n")
cat("- c: Number of parallel servers\n\n")
cat("The system can be in states 0, 1, 2, ... representing the number of customers\n")
cat("Special cases include:\n")
cat("- M/M/1: Single server queue\n")
cat("- M/M/∞: Infinite server queue\n")
cat("- M/M/c/K: Multi-server queue with finite capacity K\n\n")

# =====================================================================
# PERFORMANCE METRICS FOR M/M/c QUEUES
# =====================================================================

cat("KEY PERFORMANCE METRICS:\n")
cat("- ρ: Traffic intensity = λ/(c*μ)\n")
cat("- P0: Probability of empty system\n")
cat("- Pn: Probability of n customers in system\n")
cat("- L: Expected number of customers in system\n")
cat("- Lq: Expected number of customers in queue\n")
cat("- W: Expected time in system\n")
cat("- Wq: Expected waiting time in queue\n")
cat("- Pc: Probability of waiting (all servers busy)\n\n")

# =====================================================================
# M/M/1 QUEUE IMPLEMENTATION
# =====================================================================

cat("IMPLEMENTING M/M/1 QUEUE MODEL\n")
cat("The simplest queueing system with a single server\n\n")

# Function to analyze M/M/1 queue
analyze_mm1 <- function(lambda, mu) {
  # Check for stability
  rho <- lambda/mu
  if (rho >= 1) {
    warning("System is unstable (ρ ≥ 1). Queue will grow without bound.")
  }
  
  # Calculate performance measures
  P0 <- 1 - rho  # Probability of empty system
  L <- rho/(1 - rho)  # Expected number in system
  Lq <- rho^2/(1 - rho)  # Expected number in queue
  W <- 1/(mu - lambda)  # Expected time in system
  Wq <- rho/(mu - lambda)  # Expected waiting time in queue
  
  # Calculate state probabilities (geometric distribution)
  n_states <- min(50, ceiling(20/(1-rho)))  # Calculate enough states
  Pn <- numeric(n_states + 1)
  for (n in 0:n_states) {
    Pn[n+1] <- (1 - rho) * rho^n
  }
  
  # Return results
  return(list(
    lambda = lambda,
    mu = mu,
    rho = rho,
    P0 = P0,
    L = L,
    Lq = Lq,
    W = W,
    Wq = Wq,
    Pn = Pn
  ))
}

# Example usage for M/M/1
lambda1 <- 5  # Arrivals per hour
mu1 <- 8      # Service rate per hour

cat("Example: Store with λ =", lambda1, "customers/hour, μ =", mu1, "customers/hour\n\n")

mm1_results <- analyze_mm1(lambda1, mu1)

cat("Performance metrics for M/M/1 queue:\n")
cat("Traffic intensity (ρ):", round(mm1_results$rho, 4), "\n")
cat("Probability of empty system (P0):", round(mm1_results$P0, 4), "\n")
cat("Expected number in system (L):", round(mm1_results$L, 4), "\n")
cat("Expected number in queue (Lq):", round(mm1_results$Lq, 4), "\n")
cat("Expected time in system (W):", round(mm1_results$W, 4), "hours =", 
    round(mm1_results$W * 60, 2), "minutes\n")
cat("Expected waiting time in queue (Wq):", round(mm1_results$Wq, 4), "hours =", 
    round(mm1_results$Wq * 60, 2), "minutes\n")

# =====================================================================
# M/M/c QUEUE IMPLEMENTATION
# =====================================================================

cat("\n\nIMPLEMENTING M/M/c QUEUE MODEL\n")
cat("Multi-server queue with c parallel servers\n\n")

# Function to analyze M/M/c queue
analyze_mmc <- function(lambda, mu, c) {
  # Check for stability
  rho <- lambda/(c*mu)
  if (rho >= 1) {
    warning("System is unstable (ρ ≥ 1). Queue will grow without bound.")
  }
  
  # Calculate P0 (probability of empty system)
  sum_term <- 0
  for (n in 0:(c-1)) {
    sum_term <- sum_term + (1/factorial(n)) * (lambda/mu)^n
  }
  
  P0 <- 1 / (sum_term + (1/factorial(c)) * (lambda/mu)^c * (1/(1-rho)))
  
  # Calculate Pc (probability that all servers are busy)
  Pc <- P0 * (lambda/mu)^c / (factorial(c) * (1-rho))
  
  # Calculate performance measures
  Lq <- Pc * rho / (1 - rho)^2  # Expected queue length
  L <- Lq + lambda/mu  # Expected number in system
  Wq <- Lq / lambda  # Expected waiting time in queue
  W <- Wq + 1/mu  # Expected time in system
  
  # Calculate state probabilities
  n_states <- min(50, ceiling(c + 20/(1-rho)))  # Calculate enough states
  Pn <- numeric(n_states + 1)
  
  # States 0 to c-1
  for (n in 0:(c-1)) {
    Pn[n+1] <- P0 * (lambda/mu)^n / factorial(n)
  }
  
  # States c and beyond
  for (n in c:n_states) {
    Pn[n+1] <- P0 * (lambda/mu)^n / (factorial(c) * c^(n-c))
  }
  
  # Return results
  return(list(
    lambda = lambda,
    mu = mu,
    c = c,
    rho = rho,
    P0 = P0,
    Pc = Pc,
    L = L,
    Lq = Lq,
    W = W,
    Wq = Wq,
    Pn = Pn
  ))
}

# Example usage for M/M/c
lambda2 <- 15  # Arrivals per hour
mu2 <- 6       # Service rate per server per hour
c2 <- 3        # Number of servers

cat("Example: Call center with λ =", lambda2, "calls/hour, μ =", mu2, "calls/hour/agent, c =", c2, "agents\n\n")

mmc_results <- analyze_mmc(lambda2, mu2, c2)

cat("Performance metrics for M/M/c queue:\n")
cat("Traffic intensity (ρ):", round(mmc_results$rho, 4), "\n")
cat("Probability of empty system (P0):", round(mmc_results$P0, 4), "\n")
cat("Probability all servers busy (Pc):", round(mmc_results$Pc, 4), "\n")
cat("Expected number in system (L):", round(mmc_results$L, 4), "\n")
cat("Expected number in queue (Lq):", round(mmc_results$Lq, 4), "\n")
cat("Expected time in system (W):", round(mmc_results$W, 4), "hours =", 
    round(mmc_results$W * 60, 2), "minutes\n")
cat("Expected waiting time in queue (Wq):", round(mmc_results$Wq, 4), "hours =", 
    round(mmc_results$Wq * 60, 2), "minutes\n")

# =====================================================================
# M/M/c/K QUEUE WITH FINITE CAPACITY
# =====================================================================

cat("\n\nIMPLEMENTING M/M/c/K QUEUE MODEL\n")
cat("Multi-server queue with c servers and maximum capacity of K customers\n\n")

# Function to analyze M/M/c/K queue
analyze_mmck <- function(lambda, mu, c, K) {
  rho <- lambda/mu
  
  # Calculate P0 (probability of empty system)
  sum_term1 <- 0
  for (n in 0:(c-1)) {
    sum_term1 <- sum_term1 + (rho^n)/factorial(n)
  }
  
  sum_term2 <- 0
  for (n in c:K) {
    sum_term2 <- sum_term2 + (rho^n)/(factorial(c) * c^(n-c))
  }
  
  P0 <- 1 / (sum_term1 + sum_term2)
  
  # Calculate state probabilities
  Pn <- numeric(K + 1)
  
  # States 0 to c-1
  for (n in 0:(c-1)) {
    Pn[n+1] <- P0 * (rho^n) / factorial(n)
  }
  
  # States c to K
  for (n in c:K) {
    Pn[n+1] <- P0 * (rho^n) / (factorial(c) * c^(n-c))
  }
  
  # Calculate performance measures
  # Expected number in queue
  Lq <- P0 * (rho^c) * rho / (factorial(c) * (1 - rho/c)^2) * 
        (1 - (rho/c)^(K-c+1) - (K-c+1) * (rho/c)^(K-c) * (1 - rho/c))
  
  # Effective arrival rate (accounting for blocked customers)
  lambda_eff <- lambda * (1 - Pn[K+1])
  
  # Expected number in system
  L <- Lq + lambda_eff / mu
  
  # Expected waiting times
  W <- L / lambda_eff
  Wq <- Lq / lambda_eff
  
  # Probability of being blocked (system is full)
  Pb <- Pn[K+1]
  
  # Return results
  return(list(
    lambda = lambda,
    mu = mu,
    c = c,
    K = K,
    P0 = P0,
    Pn = Pn,
    L = L,
    Lq = Lq,
    W = W,
    Wq = Wq,
    Pb = Pb,
    lambda_eff = lambda_eff
  ))
}

# Example usage for M/M/c/K
lambda3 <- 20  # Arrivals per hour
mu3 <- 5       # Service rate per server per hour
c3 <- 4        # Number of servers
K3 <- 10       # System capacity (including those in service)

cat("Example: Hospital emergency room with λ =", lambda3, "patients/hour, μ =", 
    mu3, "patients/hour/doctor, c =", c3, "doctors, K =", K3, "capacity\n\n")

mmck_results <- analyze_mmck(lambda3, mu3, c3, K3)

cat("Performance metrics for M/M/c/K queue:\n")
cat("Probability of empty system (P0):", round(mmck_results$P0, 4), "\n")
cat("Probability system is full (Pb):", round(mmck_results$Pb, 4), "\n")
cat("Effective arrival rate:", round(mmck_results$lambda_eff, 4), "per hour\n")
cat("Expected number in system (L):", round(mmck_results$L, 4), "\n")
cat("Expected number in queue (Lq):", round(mmck_results$Lq, 4), "\n")
cat("Expected time in system (W):", round(mmck_results$W, 4), "hours =", 
    round(mmck_results$W * 60, 2), "minutes\n")
cat("Expected waiting time in queue (Wq):", round(mmck_results$Wq, 4), "hours =", 
    round(mmck_results$Wq * 60, 2), "minutes\n")

# =====================================================================
# M/M/∞ QUEUE WITH INFINITE SERVERS
# =====================================================================

cat("\n\nIMPLEMENTING M/M/∞ QUEUE MODEL\n")
cat("Queue with infinite servers (no waiting)\n\n")

# Function to analyze M/M/∞ queue
analyze_mminf <- function(lambda, mu) {
  rho <- lambda/mu
  
  # In M/M/∞, customers never wait, so:
  # - Pn follows Poisson distribution with mean rho
  # - L = rho
  # - Lq = 0 (no queue)
  # - W = 1/mu (only service time)
  # - Wq = 0 (no waiting)
  
  # Calculate state probabilities (Poisson distribution)
  n_states <- min(50, ceiling(rho + 4*sqrt(rho)))  # Enough states for Poisson
  Pn <- numeric(n_states + 1)
  for (n in 0:n_states) {
    Pn[n+1] <- exp(-rho) * rho^n / factorial(n)
  }
  
  # Return results
  return(list(
    lambda = lambda,
    mu = mu,
    rho = rho,
    P0 = exp(-rho),
    L = rho,
    Lq = 0,
    W = 1/mu,
    Wq = 0,
    Pn = Pn
  ))
}

# Example usage for M/M/∞
lambda4 <- 12  # Arrivals per hour
mu4 <- 3       # Service rate per hour

cat("Example: Self-service kiosks with λ =", lambda4, "customers/hour, μ =", mu4, "customers/hour\n\n")

mminf_results <- analyze_mminf(lambda4, mu4)

cat("Performance metrics for M/M/∞ queue:\n")
cat("Traffic intensity (ρ):", round(mminf_results$rho, 4), "\n")
cat("Probability of empty system (P0):", round(mminf_results$P0, 4), "\n")
cat("Expected number in system (L):", round(mminf_results$L, 4), "\n")
cat("Expected number in queue (Lq):", mminf_results$Lq, "\n")
cat("Expected time in system (W):", round(mminf_results$W, 4), "hours =", 
    round(mminf_results$W * 60, 2), "minutes\n")
cat("Expected waiting time in queue (Wq):", mminf_results$Wq, "\n")

# =====================================================================
# VISUALIZATION AND COMPARISON OF DIFFERENT M/M/c SYSTEMS
# =====================================================================

cat("\n\nCOMPARING DIFFERENT M/M/c SYSTEMS\n")
cat("How does performance change with different numbers of servers?\n\n")

# Function to compare performance across different number of servers
compare_servers <- function(lambda, mu, max_servers) {
  metrics <- data.frame(
    Servers = 1:max_servers,
    Rho = rep(0, max_servers),
    P0 = rep(0, max_servers),
    L = rep(0, max_servers),
    Lq = rep(0, max_servers),
    W = rep(0, max_servers),
    Wq = rep(0, max_servers),
    Pc = rep(0, max_servers)
  )
  
  for (c in 1:max_servers) {
    result <- analyze_mmc(lambda, mu, c)
    metrics$Rho[c] <- result$rho
    metrics$P0[c] <- result$P0
    metrics$L[c] <- result$L
    metrics$Lq[c] <- result$Lq
    metrics$W[c] <- result$W
    metrics$Wq[c] <- result$Wq
    metrics$Pc[c] <- result$Pc
  }
  
  return(metrics)
}

# Example comparison
lambda_comp <- 30  # Arrivals per hour
mu_comp <- 10      # Service rate per server per hour
max_servers <- 6   # Maximum number of servers to consider

cat("Comparing performance for λ =", lambda_comp, "customers/hour, μ =", mu_comp, "per server\n\n")

comparison <- compare_servers(lambda_comp, mu_comp, max_servers)

# Display results
cat("Performance metrics by number of servers:\n")
cat("Servers | Traffic | Empty Sys | Busy Prob | Queue Len | Wait Time (min)\n")
cat("--------|---------|-----------|-----------|-----------|---------------\n")
for (c in 1:max_servers) {
  cat(sprintf("%7d | %7.3f | %9.3f | %9.3f | %9.3f | %15.2f\n",
              c, comparison$Rho[c], comparison$P0[c], comparison$Pc[c], 
              comparison$Lq[c], comparison$Wq[c] * 60))
}

# Simple text-based visualization
cat("\nWaiting time in queue (minutes) by number of servers:\n")
for (c in 1:max_servers) {
  wq_min <- round(comparison$Wq[c] * 60, 2)
  stars <- paste(rep("*", ceiling(wq_min)), collapse = "")
  cat(sprintf("c = %d: %5.2f min %s\n", c, wq_min, stars))
}

# =====================================================================
# PRACTICAL APPLICATIONS AND PROBLEM SOLVING
# =====================================================================

cat("\n\nPRACTICAL APPLICATIONS\n")
cat("Using queueing theory to solve real-world problems\n\n")

# Example 1: Finding minimum number of servers needed to meet service level
cat("EXAMPLE 1: DETERMINING REQUIRED STAFFING LEVEL\n")
cat("A bank wants to ensure customers wait less than 5 minutes on average.\n")
cat("Customers arrive at rate λ = 25 per hour and each teller serves at rate μ = 8 per hour.\n")
cat("How many tellers are needed?\n\n")

lambda_bank <- 25  # Customers per hour
mu_bank <- 8       # Customers per teller per hour
target_wait <- 5/60  # 5 minutes converted to hours

# Find minimum number of servers to meet the target
c_min <- ceiling(lambda_bank/mu_bank) # Start with minimum for stability
found <- FALSE

while (!found && c_min <= 20) {  # Reasonable upper limit
  result <- analyze_mmc(lambda_bank, mu_bank, c_min)
  if (result$Wq <= target_wait) {
    found <- TRUE
  } else {
    c_min <- c_min + 1
  }
}

if (found) {
  cat("Solution: The bank needs at least", c_min, "tellers.\n")
  cat("With", c_min, "tellers, the expected wait time is", 
      round(analyze_mmc(lambda_bank, mu_bank, c_min)$Wq * 60, 2), "minutes.\n")
} else {
  cat("Could not find a solution with up to 20 servers.\n")
}

# Example 2: Evaluating capacity expansion
cat("\nEXAMPLE 2: EVALUATING CAPACITY EXPANSION\n")
cat("A restaurant has capacity for 20 customers (including those being served).\n")
cat("Customers arrive at rate λ = 40 per hour and service time is 15 minutes.\n")
cat("The restaurant is considering adding 5 more seats.\n")
cat("How will this impact the number of customers lost due to capacity?\n\n")

lambda_rest <- 40  # Customers per hour
mu_rest <- 4       # Customers per hour (15 minutes = 1/4 hour)
c_rest <- 5        # Number of servers (tables)
K_current <- 20    # Current capacity
K_new <- 25        # New capacity

# Analyze current and expanded capacity
result_current <- analyze_mmck(lambda_rest, mu_rest, c_rest, K_current)
result_new <- analyze_mmck(lambda_rest, mu_rest, c_rest, K_new)

# Calculate customers lost per hour
lost_current <- lambda_rest * result_current$Pb
lost_new <- lambda_rest * result_new$Pb
improvement <- lost_current - lost_new

cat("Current capacity: K =", K_current, "\n")
cat("- Probability of turning customers away:", round(result_current$Pb, 4), "\n")
cat("- Customers lost per hour:", round(lost_current, 2), "\n")
cat("- Effective arrival rate:", round(result_current$lambda_eff, 2), "per hour\n\n")

cat("New capacity: K =", K_new, "\n")
cat("- Probability of turning customers away:", round(result_new$Pb, 4), "\n")
cat("- Customers lost per hour:", round(lost_new, 2), "\n")
cat("- Effective arrival rate:", round(result_new$lambda_eff, 2), "per hour\n\n")

cat("By adding 5 more seats, the restaurant would serve an additional", 
    round(improvement, 2), "customers per hour.\n")
cat("This is a", round(improvement/lost_current*100, 1), 
    "% reduction in lost customers.\n")

# =====================================================================
# ECONOMIC ANALYSIS: COST-BENEFIT TRADEOFFS
# =====================================================================

cat("\n\nECONOMIC ANALYSIS OF QUEUING SYSTEMS\n")
cat("Balancing service cost against waiting cost\n\n")

cat("EXAMPLE: DETERMINING OPTIMAL NUMBER OF SERVERS\n")
cat("A service facility has the following costs:\n")
cat("- Each server costs $25 per hour\n")
cat("- Customer waiting time is valued at $40 per hour\n")
cat("- Customers arrive at rate λ = 20 per hour\n")
cat("- Each server can process μ = 6 customers per hour\n")
cat("What is the optimal number of servers to minimize total cost?\n\n")

lambda_opt <- 20   # Arrivals per hour
mu_opt <- 6        # Service rate per server
cost_server <- 25  # Cost per server per hour
cost_wait <- 40    # Cost per customer waiting hour

# Function to calculate total cost
calc_total_cost <- function(c, lambda, mu, cost_server, cost_wait) {
  # Check if the system is stable
  if (lambda/(c*mu) >= 1) {
    return(Inf)  # Unstable system has infinite cost
  }
  
  # Calculate service metrics
  result <- analyze_mmc(lambda, mu, c)
  
  # Calculate costs
  server_cost <- c * cost_server
  waiting_cost <- lambda * result$W * cost_wait
  total_cost <- server_cost + waiting_cost
  
  return(list(
    servers = c,
    server_cost = server_cost,
    waiting_cost = waiting_cost,
    total_cost = total_cost,
    wait_time = result$W
  ))
}

# Find optimal number of servers
min_servers <- ceiling(lambda_opt/mu_opt)  # Minimum for stability
max_servers <- min_servers + 10            # Reasonable upper limit

costs <- data.frame(
  Servers = min_servers:max_servers,
  ServerCost = rep(0, max_servers - min_servers + 1),
  WaitingCost = rep(0, max_servers - min_servers + 1),
  TotalCost = rep(0, max_servers - min_servers + 1),
  WaitTime = rep(0, max_servers - min_servers + 1)
)

for (i in 1:nrow(costs)) {
  c <- costs$Servers[i]
  result <- calc_total_cost(c, lambda_opt, mu_opt, cost_server, cost_wait)
  costs$ServerCost[i] <- result$server_cost
  costs$WaitingCost[i] <- result$waiting_cost
  costs$TotalCost[i] <- result$total_cost
  costs$WaitTime[i] <- result$wait_time
}

# Find optimal solution
opt_index <- which.min(costs$TotalCost)
opt_servers <- costs$Servers[opt_index]

cat("Cost analysis by number of servers:\n")
cat("Servers | Server Cost | Waiting Cost | Total Cost | Wait Time (min)\n")
cat("--------|-------------|--------------|------------|---------------\n")
for (i in 1:nrow(costs)) {
  cat(sprintf("%7d | %11.2f | %12.2f | %10.2f | %15.2f\n",
              costs$Servers[i], costs$ServerCost[i], costs$WaitingCost[i], 
              costs$TotalCost[i], costs$WaitTime[i] * 60))
}

cat("\nOptimal solution: Use", opt_servers, "servers\n")
cat("- Server cost: $", round(costs$ServerCost[opt_index], 2), "per hour\n", sep="")
cat("- Waiting cost: $", round(costs$WaitingCost[opt_index], 2), "per hour\n", sep="")
cat("- Total cost: $", round(costs$TotalCost[opt_index], 2), "per hour\n", sep="")
cat("- Average wait time:", round(costs$WaitTime[opt_index] * 60, 2), "minutes\n")

# =====================================================================
# SIMULATION OF M/M/c QUEUE
# =====================================================================

cat("\n\nSIMULATION OF M/M/c QUEUE\n")
cat("Validating the analytical results through simulation\n\n")

# Function to simulate an M/M/c queue for a given time period
simulate_mmc_queue <- function(lambda, mu, c, sim_time) {
  # Initialize variables
  time <- 0              # Current simulation time
  num_in_system <- 0     # Number of customers in system
  next_arrival <- rexp(1, lambda)  # Generate first arrival time
  departures <- numeric(0)  # No departures scheduled initially
  
  # Statistical counters
  total_customers <- 0
  total_wait <- 0
  total_time_in_system <- 0
  area_under_n <- 0      # For calculating L
  area_under_nq <- 0     # For calculating Lq
  last_event_time <- 0   # Time of last event
  
  # State tracking
  state_time <- numeric(1000)  # Time spent in each state
  max_observed <- 0            # Maximum observed state
  
  # Main simulation loop
  while (time < sim_time) {
    # Update state time
    area_under_n <- area_under_n + num_in_system * (time - last_event_time)
    area_under_nq <- area_under_nq + max(0, num_in_system - c) * (time - last_event_time)
    if (num_in_system <= length(state_time)) {
      state_time[num_in_system + 1] <- state_time[num_in_system + 1] + (time - last_event_time)
    } else if (num_in_system > max_observed) {
      state_time <- c(state_time, numeric(num_in_system - length(state_time) + 1))
      state_time[num_in_system + 1] <- time - last_event_time
      max_observed <- num_in_system
    }
    last_event_time <- time
    
    # Determine next event
    if (length(departures) == 0 || next_arrival < min(departures)) {
      # Process arrival
      time <- next_arrival
      num_in_system <- num_in_system + 1
      total_customers <- total_customers + 1
      
      # Schedule next arrival
      next_arrival <- time + rexp(1, lambda)
      
      # If servers available, schedule service completion
      if (num_in_system <= c) {
        service_time <- rexp(1, mu)
        departures <- c(departures, time + service_time)
      }
      
      # Calculate waiting time (if any)
      wait <- max(0, (num_in_system - c)/c) * (1/mu)
      total_wait <- total_wait + wait
      total_time_in_system <- total_time_in_system + wait + (1/mu)
      
    } else {
      # Process departure
      time <- min(departures)
      departure_idx <- which.min(departures)
      departures <- departures[-departure_idx]
      num_in_system <- num_in_system - 1
      
      # If customers waiting, move one to service
      if (num_in_system >= c) {
        service_time <- rexp(1, mu)
        departures <- c(departures, time + service_time)
      }
    }
  }
  
  # Calculate performance measures
  L <- area_under_n / time
  Lq <- area_under_nq / time
  W <- total_time_in_system / total_customers
  Wq <- total_wait / total_customers
  
  # Calculate state probabilities
  total_time <- sum(state_time[1:(max_observed+1)])
  Pn <- state_time[1:(max_observed+1)] / total_time
  
  # Return results
  return(list(
    lambda = lambda,
    mu = mu,
    c = c,
    L = L,
    Lq = Lq,
    W = W,
    Wq = Wq,
    Pn = Pn,
    total_customers = total_customers
  ))
}

# Example simulation
lambda_sim <- 15    # Arrivals per hour
mu_sim <- 6         # Service rate per server per hour
c_sim <- 3          # Number of servers
sim_time <- 1000    # Simulation time (hours)

cat("Running simulation with λ =", lambda_sim, ", μ =", mu_sim, ", c =", c_sim, "\n")
cat("Simulation time:", sim_time, "hours\n\n")

# Start timer to measure computation time
start_time <- Sys.time()

# Run simulation
sim_results <- simulate_mmc_queue(lambda_sim, mu_sim, c_sim, sim_time)

# End timer
end_time <- Sys.time()
sim_run_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat("Simulation completed in", round(sim_run_time, 2), "seconds.\n")
cat("Processed", sim_results$total_customers, "customers.\n\n")

# Calculate theoretical results for comparison
theory_results <- analyze_mmc(lambda_sim, mu_sim, c_sim)

cat("Comparison of Simulation vs. Theoretical Results:\n")
cat("Metric          | Simulation | Theoretical | Difference (%)\n")
cat("----------------|------------|-------------|---------------\n")
cat(sprintf("L (system size)  | %10.4f | %11.4f | %13.2f%%\n",
            sim_results$L, theory_results$L, 
            abs(sim_results$L - theory_results$L)/theory_results$L*100))
cat(sprintf("Lq (queue size)  | %10.4f | %11.4f | %13.2f%%\n",
            sim_results$Lq, theory_results$Lq, 
            abs(sim_results$Lq - theory_results$Lq)/theory_results$Lq*100))
cat(sprintf("W (system time)  | %10.4f | %11.4f | %13.2f%%\n",
            sim_results$W, theory_results$W, 
            abs(sim_results$W - theory_results$W)/theory_results$W*100))
cat(sprintf("Wq (wait time)   | %10.4f | %11.4f | %13.2f%%\n",
            sim_results$Wq, theory_results$Wq, 
            abs(sim_results$Wq - theory_results$Wq)/theory_results$Wq*100))

# Compare state probabilities
cat("\nState probabilities for first few states:\n")
cat("State | Simulation | Theoretical | Difference (%)\n")
cat("------|------------|-------------|---------------\n")
for (n in 0:min(5, length(sim_results$Pn)-1)) {
  if (n < length(theory_results$Pn)) {
    cat(sprintf("%5d | %10.4f | %11.4f | %13.2f%%\n",
                n, sim_results$Pn[n+1], theory_results$Pn[n+1], 
                abs(sim_results$Pn[n+1] - theory_results$Pn[n+1])/theory_results$Pn[n+1]*100))
  } else {
    cat(sprintf("%5d | %10.4f | %11s | %13s\n", n, sim_results$Pn[n+1], "N/A", "N/A"))
  }
}

cat("\n\n======= END OF M/M/c QUEUING SYSTEMS EXAMPLES =======\n")
