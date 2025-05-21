# Gaussian Processes in R
# Using only built-in R functions (base R)

# ============================================================================
# 1. INTRODUCTION TO GAUSSIAN PROCESSES
# ============================================================================
cat("\n======= GAUSSIAN PROCESSES INTRODUCTION =======\n")
cat("A Gaussian Process (GP) is a collection of random variables where any finite\n")
cat("subset follows a multivariate Gaussian distribution.\n")
cat("GPs are fully specified by:\n")
cat("1. A mean function: m(x)\n")
cat("2. A covariance (kernel) function: k(x, x')\n")
cat("They are powerful non-parametric models for regression and classification.\n")

# ============================================================================
# 2. COVARIANCE (KERNEL) FUNCTIONS
# ============================================================================
cat("\n\n======= COVARIANCE FUNCTIONS (KERNELS) =======\n")
cat("The kernel function defines the similarity between points and shapes the GP behavior.\n")
cat("Common kernels include:\n")
cat("- Squared Exponential (RBF): k(x,x') = σ² exp(-||x-x'||²/2l²)\n")
cat("- Matérn: Generalizes RBF with additional smoothness parameter\n")
cat("- Periodic: For data with repeating patterns\n")
cat("- Linear: For linear relationships\n")

# Implementation of squared exponential (RBF) kernel
rbf_kernel <- function(x1, x2, l = 1, sigma_f = 1) {
  # x1, x2: vectors of input points
  # l: length scale parameter
  # sigma_f: signal variance
  
  sqdist <- outer(x1, x2, function(a, b) (a - b)^2)
  return(sigma_f^2 * exp(-0.5 * sqdist / l^2))
}

# Implementation of Matérn kernel (nu = 3/2)
matern_kernel <- function(x1, x2, l = 1, sigma_f = 1) {
  # Matérn kernel with nu = 3/2
  sqdist <- outer(x1, x2, function(a, b) abs(a - b))
  d <- sqrt(3) * sqdist / l
  return(sigma_f^2 * (1 + d) * exp(-d))
}

# Implementation of periodic kernel
periodic_kernel <- function(x1, x2, l = 1, sigma_f = 1, p = 1) {
  # p: period length
  dist <- outer(x1, x2, function(a, b) sin(pi * abs(a - b) / p)^2)
  return(sigma_f^2 * exp(-2 * dist / l^2))
}

# Demonstration of kernels
cat("\nDemonstrating covariance kernels on a 1D example:\n")

# Generate some points
x_demo <- seq(-5, 5, length.out = 11)

# Calculate covariance matrices using different kernels
K_rbf <- rbf_kernel(x_demo, x_demo, l = 1, sigma_f = 1)
K_matern <- matern_kernel(x_demo, x_demo, l = 1, sigma_f = 1)
K_periodic <- periodic_kernel(x_demo, x_demo, l = 1, sigma_f = 1, p = 2)

# Display portions of the covariance matrices
cat("\nSquared Exponential (RBF) Kernel (first 5x5 elements):\n")
print(round(K_rbf[1:5, 1:5], 3))

cat("\nMatérn Kernel (first 5x5 elements):\n")
print(round(K_matern[1:5, 1:5], 3))

cat("\nPeriodic Kernel (first 5x5 elements):\n")
print(round(K_periodic[1:5, 1:5], 3))

# ============================================================================
# 3. SAMPLING FROM A GAUSSIAN PROCESS
# ============================================================================
cat("\n\n======= SAMPLING FROM A GAUSSIAN PROCESS =======\n")
cat("Sampling from a GP draws functions from the prior distribution.\n")
cat("This helps visualize the behavior of different kernels.\n")

# Function to sample from a GP
sample_gp <- function(x, kernel_func, n_samples = 3, mean_func = NULL, ...) {
  # x: Vector of input points
  # kernel_func: Covariance function
  # n_samples: Number of sample functions to draw
  # mean_func: Mean function (default = 0)
  # ...: Additional parameters for kernel_func
  
  n <- length(x)
  
  # Calculate covariance matrix
  K <- kernel_func(x, x, ...)
  
  # Add small value to diagonal for numerical stability
  K <- K + diag(1e-10, n)
  
  # Calculate mean vector
  if (is.null(mean_func)) {
    mu <- rep(0, n)
  } else {
    mu <- sapply(x, mean_func)
  }
  
  # Cholesky decomposition of covariance matrix
  L <- chol(K)
  
  # Generate samples
  samples <- matrix(rnorm(n * n_samples), nrow = n)
  samples <- mu + t(L) %*% samples
  
  return(list(x = x, samples = samples))
}

# Generate samples from different kernels
set.seed(123)  # For reproducibility
x_plot <- seq(-5, 5, length.out = 100)

# RBF kernel
samples_rbf <- sample_gp(x_plot, rbf_kernel, n_samples = 3, l = 1, sigma_f = 1)

# Matérn kernel
samples_matern <- sample_gp(x_plot, matern_kernel, n_samples = 3, l = 1, sigma_f = 1)

# Periodic kernel
samples_periodic <- sample_gp(x_plot, periodic_kernel, n_samples = 3, l = 1, sigma_f = 1, p = 2)

# Display sample values at specific points
cat("\nSample values from RBF kernel GP at x = {-4, -2, 0, 2, 4}:\n")
idx <- c(11, 31, 51, 71, 91)  # Indices of selected points
cat("Sample 1:", round(samples_rbf$samples[1, idx], 3), "\n")
cat("Sample 2:", round(samples_rbf$samples[2, idx], 3), "\n")
cat("Sample 3:", round(samples_rbf$samples[3, idx], 3), "\n")

# ============================================================================
# 4. GAUSSIAN PROCESS REGRESSION (GPR)
# ============================================================================
cat("\n\n======= GAUSSIAN PROCESS REGRESSION =======\n")
cat("GPR predicts values at new points by conditioning the GP on observed data.\n")
cat("The posterior distribution provides both predictions and uncertainty estimates.\n")

# Function for GP regression
gp_regression <- function(X_train, y_train, X_test, kernel_func, noise_var = 0.1, ...) {
  # X_train, y_train: Training data
  # X_test: Test points
  # kernel_func: Covariance function
  # noise_var: Observation noise variance
  # ...: Additional parameters for kernel_func
  
  n_train <- length(X_train)
  n_test <- length(X_test)
  
  # Compute covariance matrices
  K <- kernel_func(X_train, X_train, ...)
  K_s <- kernel_func(X_train, X_test, ...)
  K_ss <- kernel_func(X_test, X_test, ...)
  
  # Add noise to training covariance
  K_y <- K + diag(noise_var, n_train)
  
  # Compute posterior mean and covariance
  K_y_inv <- solve(K_y)
  mean_post <- t(K_s) %*% K_y_inv %*% y_train
  cov_post <- K_ss - t(K_s) %*% K_y_inv %*% K_s
  
  # Extract variance (diagonal of covariance matrix)
  var_post <- diag(cov_post)
  
  # Return posterior mean and variance
  return(list(
    mean = mean_post,
    var = var_post,
    cov = cov_post
  ))
}

# ============================================================================
# 5. POSTERIOR DISTRIBUTION OF GAUSSIAN PROCESS
# ============================================================================
cat("\n\n======= POSTERIOR DISTRIBUTION OF GAUSSIAN PROCESS =======\n")
cat("The posterior distribution of a GP after conditioning on observations is also a GP.\n")
cat("It provides a probability distribution over possible functions that fit the data.\n")

# Generate synthetic data
set.seed(456)
gen_synthetic_data <- function(n = 20, noise_var = 0.1) {
  # Generate n points with sine function and some noise
  x <- seq(-5, 5, length.out = n)
  f_true <- sin(x)
  y <- f_true + rnorm(n, sd = sqrt(noise_var))
  return(list(x = x, y = y, f_true = f_true))
}

data <- gen_synthetic_data(n = 20, noise_var = 0.1)
X_train <- data$x
y_train <- data$y
f_true <- data$f_true

# Test points for prediction
X_test <- seq(-6, 6, length.out = 100)

# Perform GP regression with RBF kernel
gp_rbf <- gp_regression(X_train, y_train, X_test, rbf_kernel, 
                        noise_var = 0.1, l = 1, sigma_f = 1)

# Calculate 95% confidence intervals
ci_lower <- gp_rbf$mean - 1.96 * sqrt(gp_rbf$var)
ci_upper <- gp_rbf$mean + 1.96 * sqrt(gp_rbf$var)

# Display results at specific points
cat("\nPosterior distribution at selected test points:\n")
test_idx <- c(10, 25, 50, 75, 90)
cat("x:", round(X_test[test_idx], 2), "\n")
cat("Predicted mean:", round(gp_rbf$mean[test_idx], 3), "\n")
cat("Predicted std:", round(sqrt(gp_rbf$var[test_idx]), 3), "\n")
cat("95% CI lower:", round(ci_lower[test_idx], 3), "\n")
cat("95% CI upper:", round(ci_upper[test_idx], 3), "\n")

# ============================================================================
# 6. HYPERPARAMETER ESTIMATION IN GAUSSIAN PROCESSES
# ============================================================================
cat("\n\n======= HYPERPARAMETER ESTIMATION IN GAUSSIAN PROCESSES =======\n")
cat("Hyperparameters (e.g., length scale, signal variance) control GP behavior.\n")
cat("They can be estimated by maximizing the marginal log-likelihood.\n")

# Log marginal likelihood function
log_marginal_likelihood <- function(params, X, y, kernel_func, kernel_params_names) {
  # params: Vector of hyperparameters
  # X, y: Training data
  # kernel_func: Covariance function
  # kernel_params_names: Names of the kernel parameters
  
  # Extract parameters
  noise_var <- params[1]
  kernel_params <- params[-1]
  
  # Create named list of kernel parameters
  kernel_args <- as.list(kernel_params)
  names(kernel_args) <- kernel_params_names
  
  # Calculate covariance matrix with current parameters
  K <- do.call(kernel_func, c(list(X, X), kernel_args))
  n <- length(y)
  
  # Add noise to diagonal
  K_y <- K + diag(noise_var, n)
  
  # Compute log marginal likelihood
  # log(p(y|X)) = -0.5 * (y^T * K_y^-1 * y + log|K_y| + n*log(2π))
  
  # Cholesky decomposition for stable inversion
  L <- chol(K_y)
  alpha <- solve(t(L), solve(L, y))
  
  # Calculate terms
  term1 <- -0.5 * t(y) %*% alpha
  term2 <- -sum(log(diag(L)))  # log determinant
  term3 <- -0.5 * n * log(2 * pi)
  
  log_ml <- term1 + term2 + term3
  
  return(as.numeric(log_ml))
}

# Function to optimize hyperparameters
optimize_hyperparameters <- function(X, y, kernel_func, 
                                     kernel_params_names, 
                                     initial_params,
                                     lower_bounds, 
                                     upper_bounds) {
  # X, y: Training data
  # kernel_func: Covariance function
  # kernel_params_names: Names of kernel parameters
  # initial_params: Initial values for [noise_var, kernel_params]
  # lower_bounds, upper_bounds: Bounds for optimization
  
  # Negative log marginal likelihood for minimization
  neg_log_ml <- function(params) {
    -log_marginal_likelihood(params, X, y, kernel_func, kernel_params_names)
  }
  
  # Optimize using L-BFGS-B method
  opt_result <- optim(
    par = initial_params,
    fn = neg_log_ml,
    method = "L-BFGS-B",
    lower = lower_bounds,
    upper = upper_bounds,
    control = list(maxit = 100)
  )
  
  # Extract optimized parameters
  opt_params <- opt_result$par
  opt_value <- -opt_result$value  # Convert back to positive log-likelihood
  
  return(list(
    noise_var = opt_params[1],
    kernel_params = opt_params[-1],
    log_ml = opt_value,
    convergence = opt_result$convergence,
    message = opt_result$message
  ))
}

# Demonstrate hyperparameter optimization for RBF kernel
cat("\nOptimizing hyperparameters for RBF kernel GP:\n")

# Initial and bound values for [noise_var, l, sigma_f]
initial_params <- c(0.1, 1.0, 1.0)
lower_bounds <- c(1e-5, 0.1, 0.1)
upper_bounds <- c(1.0, 10.0, 10.0)
kernel_params_names <- c("l", "sigma_f")

# Perform optimization
opt_result <- optimize_hyperparameters(
  X_train, y_train, rbf_kernel, 
  kernel_params_names, 
  initial_params, 
  lower_bounds, 
  upper_bounds
)

cat("Initial parameters: noise_var =", initial_params[1], 
    ", l =", initial_params[2], 
    ", sigma_f =", initial_params[3], "\n")

cat("Optimized parameters: noise_var =", round(opt_result$noise_var, 4),
    ", l =", round(opt_result$kernel_params[1], 4),
    ", sigma_f =", round(opt_result$kernel_params[2], 4), "\n")

cat("Log marginal likelihood:", round(opt_result$log_ml, 4), "\n")
cat("Convergence status:", opt_result$convergence, 
    "(0 indicates successful convergence)\n")

# ============================================================================
# 7. PREDICTION USING GAUSSIAN PROCESSES
# ============================================================================
cat("\n\n======= PREDICTION USING GAUSSIAN PROCESSES =======\n")
cat("Using optimized hyperparameters for prediction improves model performance.\n")
cat("The GP provides both point predictions and uncertainty estimates.\n")

# Perform GP regression with optimized hyperparameters
noise_var_opt <- opt_result$noise_var
l_opt <- opt_result$kernel_params[1]
sigma_f_opt <- opt_result$kernel_params[2]

gp_opt <- gp_regression(X_train, y_train, X_test, rbf_kernel, 
                       noise_var = noise_var_opt, 
                       l = l_opt, 
                       sigma_f = sigma_f_opt)

# Calculate 95% confidence intervals
ci_lower_opt <- gp_opt$mean - 1.96 * sqrt(gp_opt$var)
ci_upper_opt <- gp_opt$mean + 1.96 * sqrt(gp_opt$var)

# Calculate RMSE for default and optimized models
rmse_default <- sqrt(mean((gp_rbf$mean - sin(X_test))^2))
rmse_optimized <- sqrt(mean((gp_opt$mean - sin(X_test))^2))

cat("\nPerformance comparison on true function (sine):\n")
cat("RMSE with default hyperparameters:", round(rmse_default, 4), "\n")
cat("RMSE with optimized hyperparameters:", round(rmse_optimized, 4), "\n")

cat("\nPredictions at selected test points using optimized hyperparameters:\n")
cat("x:", round(X_test[test_idx], 2), "\n")
cat("True values (sine):", round(sin(X_test[test_idx]), 3), "\n")
cat("Predicted mean:", round(gp_opt$mean[test_idx], 3), "\n")
cat("Predicted std:", round(sqrt(gp_opt$var[test_idx]), 3), "\n")
cat("95% CI lower:", round(ci_lower_opt[test_idx], 3), "\n")
cat("95% CI upper:", round(ci_upper_opt[test_idx], 3), "\n")

# ============================================================================
# 8. APPLICATION EXAMPLE: NOISY FUNCTION APPROXIMATION
# ============================================================================
cat("\n\n======= APPLICATION: NOISY FUNCTION APPROXIMATION =======\n")
cat("Applying GP regression to a more complex function with limited data.\n")

# Generate synthetic data from a more complex function
set.seed(789)
gen_complex_data <- function(n = 15, noise_var = 0.3) {
  # Generate n points with a complex function and noise
  x <- runif(n, -3, 3)  # Randomly spaced points
  f_true <- x^3 - 5*x^2 + 3*x + 2*sin(2.5*x)
  y <- f_true + rnorm(n, sd = sqrt(noise_var))
  return(list(x = x, y = y, f_true = function(x) x^3 - 5*x^2 + 3*x + 2*sin(2.5*x)))
}

complex_data <- gen_complex_data(n = 15, noise_var = 0.3)
X_complex <- complex_data$x
y_complex <- complex_data$y
f_complex <- complex_data$f_true

# Test points for prediction
X_test_complex <- seq(-3.5, 3.5, length.out = 100)
y_true_complex <- sapply(X_test_complex, f_complex)

# Optimize hyperparameters
initial_params_complex <- c(0.3, 1.0, 1.0)
lower_bounds_complex <- c(1e-5, 0.1, 0.1)
upper_bounds_complex <- c(2.0, 10.0, 10.0)

opt_result_complex <- optimize_hyperparameters(
  X_complex, y_complex, rbf_kernel, 
  kernel_params_names, 
  initial_params_complex, 
  lower_bounds_complex, 
  upper_bounds_complex
)

cat("\nOptimized hyperparameters for complex function:\n")
cat("noise_var =", round(opt_result_complex$noise_var, 4),
    ", l =", round(opt_result_complex$kernel_params[1], 4),
    ", sigma_f =", round(opt_result_complex$kernel_params[2], 4), "\n")

# Perform prediction
gp_complex <- gp_regression(
  X_complex, y_complex, X_test_complex, rbf_kernel, 
  noise_var = opt_result_complex$noise_var, 
  l = opt_result_complex$kernel_params[1], 
  sigma_f = opt_result_complex$kernel_params[2]
)

# Calculate 95% confidence intervals
ci_lower_complex <- gp_complex$mean - 1.96 * sqrt(gp_complex$var)
ci_upper_complex <- gp_complex$mean + 1.96 * sqrt(gp_complex$var)

# Calculate RMSE for the complex function
rmse_complex <- sqrt(mean((gp_complex$mean - y_true_complex)^2))
cat("RMSE on complex function:", round(rmse_complex, 4), "\n")

# Calculate fraction of true values within 95% CI
within_ci <- mean(y_true_complex >= ci_lower_complex & 
                 y_true_complex <= ci_upper_complex)
cat("Fraction of true values within 95% CI:", round(within_ci, 4), "\n")

cat("\n\n======= END OF GAUSSIAN PROCESSES EXAMPLES =======\n")
cat("This file demonstrates Gaussian Process regression concepts\n")
cat("using only built-in R functions.\n")
