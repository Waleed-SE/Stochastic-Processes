forward_algorithm <- function(A, B, pi, Observations) {
    n <- nrow(A)
    n_obs <- length(Observations)

    alpha <- matrix(0, nrow = n, ncol = n_obs)

    for (i in 1:n) {
        alpha[i, 1] <- pi[i] * B[i, Observations[1]]
    }

    for (t in 1:n_obs) {
        for (j in 1:n) {
            sum <- 0
            for (i in 1:n) {
                sum <- sum + alpha[i, t - 1] * A[i, j]
            }
            alpha[j, t] <- sum * B[j, Observations[t]]
        }
    }
    return(alpha)
}


mfpt <- function(P) {
    n <- nrow(P)

    ss <- eigen(t(P))$vectors[, 1] / sum(eigen(t(P))$vectors[, 1])

    W <- matrix(ss, nrow = n, ncol = n, byrow = TRUE)
    Z <- solve(diag(n) - P + W)

    M <- matrix(0, nrow = n, ncol = n)
    for (i in 1:n) {
        for (j in 1:n) {
            if (i != j) {
                M[i, j] <- (Z[j, j] - Z[i, j]) / ss[j]
            } else {
                M[i, j] <- 1 / ss[i]
            }
        }
    }
    return(M)
}
