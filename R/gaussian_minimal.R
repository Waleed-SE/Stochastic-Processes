# Minimal Gaussian Processes in R
# Contains only core functionality for GP regression

# ============================================================================
# 1. CORE FUNCTIONS
# ============================================================================

# RBF (Squared Exponential) kernel function
rbf_kernel <- function(x1, x2, l = 1, sigma_f = 1) {
    # x1, x2: vectors of input points
    # l: length scale parameter
    # sigma_f: signal variance

    sqdist <- outer(x1, x2, function(a, b) (a - b)^2)
    return(sigma_f^2 * exp(-0.5 * sqdist / l^2))
}

# Gaussian Process Regression function
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
# 2. EXAMPLE USAGE
# ============================================================================

# Generate some synthetic data
set.seed(123)
gen_data <- function(n = 20, noise_var = 0.1) {
    x <- seq(-5, 5, length.out = n)
    f_true <- sin(x)
    y <- f_true + rnorm(n, sd = sqrt(noise_var))
    return(list(x = x, y = y, f_true = f_true))
}

# Generate data
data <- gen_data()
X_train <- data$x
y_train <- data$y

# Test points for prediction
X_test <- seq(-6, 6, length.out = 100)

# Perform GP regression
gp_result <- gp_regression(X_train, y_train, X_test, rbf_kernel,
    noise_var = 0.1, l = 1, sigma_f = 1
)

# Calculate 95% confidence intervals
ci_lower <- gp_result$mean - 1.96 * sqrt(gp_result$var)
ci_upper <- gp_result$mean + 1.96 * sqrt(gp_result$var)

# Output some results
cat("Predictions at selected test points:\n")
test_idx <- c(10, 50, 90)
cat("x:", round(X_test[test_idx], 2), "\n")
cat("Predicted mean:", round(gp_result$mean[test_idx], 3), "\n")
cat("Predicted std:", round(sqrt(gp_result$var[test_idx]), 3), "\n")


P1 <- matrix(c(
    0.5, 0.2, 0.3,
    0.4, 0.5, 0.1,
    0.7, 0.1, 0.2
), nrow = 3, byrow = TRUE)

P2 <- matrix(c(
    0.5, 0.2, 0.3,
    0.4, 0.5, 0.1,
    0, 0, 1
), nrow = 3, byrow = TRUE)

P3 <- matrix(c(
    0.5, 0.2,
    0.4, 0.5
), nrow = 2, byrow = TRUE)
