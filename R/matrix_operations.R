# Matrix Operations in R

# Method 1: Initialize matrices using matrix() function
# Syntax: matrix(data, nrow, ncol, byrow)
# data: vector of elements
# nrow: number of rows
# ncol: number of columns
# byrow: logical value. If TRUE, the matrix is filled by rows

# Creating a 2x3 matrix filled by rows
matrix_A <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3, byrow = TRUE)
print("Matrix A:")
print(matrix_A)

# Creating a 3x2 matrix filled by columns
matrix_B <- matrix(c(7, 8, 9, 10, 11, 12), nrow = 3, ncol = 2)
print("Matrix B:")
print(matrix_B)

# Method 2: Initialize matrices with specific values
matrix_C <- matrix(0, nrow = 2, ncol = 2)  # 2x2 matrix of zeros
print("Matrix C (zeros):")
print(matrix_C)

matrix_D <- matrix(1, nrow = 3, ncol = 3)  # 3x3 matrix of ones
print("Matrix D (ones):")
print(matrix_D)

# Method 3: Create diagonal matrices
diag_matrix <- diag(c(1, 2, 3, 4))  # Diagonal matrix
print("Diagonal Matrix:")
print(diag_matrix)

# Method 4: Create an identity matrix
identity_matrix <- diag(3)  # 3x3 identity matrix
print("Identity Matrix:")
print(identity_matrix)

# Matrix Multiplication
# For matrix multiplication, the number of columns in the first matrix
# must equal the number of rows in the second matrix

# Creating matrices for multiplication
A <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)
B <- matrix(c(5, 6, 7, 8), nrow = 2, ncol = 2)

print("Matrix A for multiplication:")
print(A)
print("Matrix B for multiplication:")
print(B)

# Matrix multiplication using the %*% operator
result <- A %*% B
print("Result of A %*% B:")
print(result)

# Element-wise multiplication using the * operator
element_wise <- A * B
print("Element-wise multiplication A * B:")
print(element_wise)

# Additional matrix operations
# Transpose a matrix using t()
A_transpose <- t(A)
print("Transpose of matrix A:")
print(A_transpose)

# Calculate the determinant using det()
A_det <- det(A)
print("Determinant of matrix A:")
print(A_det)

# Calculate the inverse using solve()
A_inverse <- solve(A)
print("Inverse of matrix A:")
print(A_inverse)

# Verify that A * A⁻¹ = I (identity matrix)
verify <- A %*% A_inverse
print("A multiplied by its inverse (should be identity matrix):")
print(round(verify, digits = 10))  # Round to handle floating point imprecisions
