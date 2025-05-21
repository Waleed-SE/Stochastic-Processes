# Discrete Markov Chains in Python
# Using only built-in Python libraries (no NumPy, SciPy, etc.)

import math
import random
from functools import reduce
from operator import mul

# Helper functions for matrix operations (since we're not using NumPy)
def matrix_multiply(A, B):
    """Multiply two matrices."""
    # Ensure dimensions are compatible
    if len(A[0]) != len(B):
        raise ValueError("Matrix dimensions incompatible for multiplication")
    
    # Initialize result matrix with zeros
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    
    # Perform matrix multiplication
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

def vector_matrix_multiply(v, A):
    """Multiply a vector (row) by a matrix."""
    # Initialize result vector with zeros
    result = [0] * len(A[0])
    
    # Perform vector-matrix multiplication
    for j in range(len(A[0])):
        for i in range(len(v)):
            result[j] += v[i] * A[i][j]
    
    return result

def matrix_power(A, n):
    """Raise a matrix to power n (A^n)."""
    if n == 1:
        return A
    
    # Initialize result with identity matrix
    result = A.copy()
    
    # Multiply A by itself n-1 times
    for _ in range(n-1):
        result = matrix_multiply(result, A)
    
    return result

def identity_matrix(n):
    """Create an n x n identity matrix."""
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def matrix_subtract(A, B):
    """Subtract matrix B from matrix A."""
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def matrix_inverse(A):
    """
    Calculate the inverse of a square matrix using Gaussian elimination.
    This is a simplified implementation for small matrices.
    """
    n = len(A)
    
    # Create augmented matrix [A|I]
    augmented = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(A)]
    
    # Gaussian elimination (forward elimination)
    for i in range(n):
        # Find pivot
        max_index = i
        max_value = abs(augmented[i][i])
        
        for j in range(i+1, n):
            if abs(augmented[j][i]) > max_value:
                max_index = j
                max_value = abs(augmented[j][i])
        
        # Swap rows if necessary
        if max_index != i:
            augmented[i], augmented[max_index] = augmented[max_index], augmented[i]
        
        # Scale pivot row
        pivot = augmented[i][i]
        if pivot == 0:
            raise ValueError("Matrix is singular and cannot be inverted")
        
        for j in range(i, 2*n):
            augmented[i][j] /= pivot
        
        # Eliminate below
        for j in range(i+1, n):
            factor = augmented[j][i]
            for k in range(i, 2*n):
                augmented[j][k] -= factor * augmented[i][k]
    
    # Back substitution
    for i in range(n-1, -1, -1):
        for j in range(i):
            factor = augmented[j][i]
            for k in range(i, 2*n):
                augmented[j][k] -= factor * augmented[i][k]
    
    # Extract inverse matrix
    inverse = [row[n:] for row in augmented]
    
    return inverse

def transpose(A):
    """Transpose a matrix."""
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def round_matrix(A, digits=4):
    """Round all elements in a matrix to specified number of digits."""
    return [[round(val, digits) for val in row] for row in A]

def sum_abs_diff(v1, v2):
    """Calculate the sum of absolute differences between two vectors."""
    return sum(abs(a - b) for a, b in zip(v1, v2))

def solve_linear_system(A, b):
    """
    Solve the linear system Ax = b.
    Returns the solution vector x.
    """
    # Calculate A^-1
    A_inv = matrix_inverse(A)
    # Calculate x = A^-1 * b
    x = [sum(A_inv[i][j] * b[j] for j in range(len(b))) for i in range(len(A_inv))]
    return x

def print_matrix(matrix, row_names=None, col_names=None, digits=4):
    """Print a matrix with optional row and column names."""
    if row_names and len(row_names) != len(matrix):
        raise ValueError("Number of row names doesn't match matrix dimensions")
    if col_names and len(col_names) != len(matrix[0]):
        raise ValueError("Number of column names doesn't match matrix dimensions")
    
    # Print column names if provided
    if col_names:
        print("     ", end="")
        for name in col_names:
            print(f"{name:>10}", end="")
        print()
    
    # Print matrix with row names
    for i, row in enumerate(matrix):
        if row_names:
            print(f"{row_names[i]:<5}", end="")
        else:
            print(f"[{i}] ", end="")
        
        for val in row:
            print(f"{round(val, digits):>10.{digits}f}", end="")
        print()

# ============================================================================
# 1. DEFINITION AND EXAMPLES OF MARKOV CHAINS
# ============================================================================
print("\n======= MARKOV CHAIN DEFINITION =======")
print("A Markov chain is a stochastic process where future states depend only on the current state")
print("and not on the sequence of events that preceded it (Markov property).\n")

# Example 1: Simple Weather Model
print("Example 1: Weather Model (Sunny, Cloudy, Rainy)")
weather_states = ["Sunny", "Cloudy", "Rainy"]

# Transition probability matrix P
# P[i][j] = Probability of moving from state i to state j
P_weather = [
    [0.7, 0.2, 0.1],  # From Sunny: P(S->S)=0.7, P(S->C)=0.2, P(S->R)=0.1
    [0.3, 0.4, 0.3],  # From Cloudy: P(C->S)=0.3, P(C->C)=0.4, P(C->R)=0.3
    [0.2, 0.4, 0.4]   # From Rainy: P(R->S)=0.2, P(R->C)=0.4, P(R->R)=0.4
]

print("\nTransition Probability Matrix:")
print_matrix(P_weather, weather_states, weather_states)

# Verify that each row sums to 1 (stochastic matrix property)
print("\nVerify rows sum to 1:")
row_sums = [sum(row) for row in P_weather]
for i, s in enumerate(weather_states):
    print(f"{s}: {row_sums[i]}")

# Example 2: Simple 2-state Markov Chain
print("\nExample 2: Two-state Markov Chain")
two_states = ["State A", "State B"]
P_two_state = [
    [0.6, 0.4],
    [0.3, 0.7]
]
print("\nTransition Matrix for Two-state Chain:")
print_matrix(P_two_state, two_states, two_states)

# ============================================================================
# 2. TRANSITION PROBABILITY MATRIX
# ============================================================================
print("\n\n======= TRANSITION PROBABILITY MATRIX =======")
print("The transition probability matrix P contains probabilities pij of moving from state i to j.")
print("Each row must sum to 1, as they represent probability distributions.")

# Let's create a more complex example
print("\nExample 3: Four-state Markov Chain")
four_states = ["State 1", "State 2", "State 3", "State 4"]
P_four_state = [
    [0.5, 0.3, 0.2, 0.0],
    [0.0, 0.6, 0.3, 0.1],
    [0.2, 0.0, 0.5, 0.3],
    [0.1, 0.1, 0.0, 0.8]
]
print_matrix(P_four_state, four_states, four_states)

# ============================================================================
# 3. CALCULATION OF n-STEP TRANSITION PROBABILITIES
# ============================================================================
print("\n\n======= n-STEP TRANSITION PROBABILITIES =======")
print("The n-step transition matrix P^n gives probabilities of going from state i to j in exactly n steps.")
print("It is calculated by raising the transition matrix to the power n using matrix multiplication.")

# 2-step transition probabilities using matrix multiplication
P_weather_2step = matrix_multiply(P_weather, P_weather)
print("\nTwo-step transition probabilities for the weather model:")
print_matrix(P_weather_2step, weather_states, weather_states)

# Calculate 3-step probabilities
P_weather_3step = matrix_multiply(P_weather_2step, P_weather)
print("\nThree-step transition probabilities:")
print_matrix(P_weather_3step, weather_states, weather_states)

# Function to calculate n-step transition matrix
def n_step_transition(P, n):
    result = P
    for _ in range(1, n):
        result = matrix_multiply(result, P)
    return result

# Calculate 10-step transition matrix
P_10step = n_step_transition(P_weather, 10)
print("\nTen-step transition probabilities:")
print_matrix(P_10step, weather_states, weather_states)

# ============================================================================
# 4. STEADY STATE PROBABILITY & STATIONARY DISTRIBUTIONS
# ============================================================================
print("\n\n======= STEADY STATE PROBABILITY =======")
print("The steady state (or stationary distribution) π is a probability vector such that π = π*P.")
print("It represents the long-term probabilities of being in each state regardless of the initial state.")

# Method 1: Find steady state by raising the transition matrix to a high power
print("\nMethod 1: Raising transition matrix to high power")
P_high_power = n_step_transition(P_weather, 100)
print("Approximating steady state by P^100:")
print([f"{weather_states[i]}: {round(P_high_power[0][i], 4)}" for i in range(len(weather_states))])

# Method 2: Solve the system of linear equations
# For a stationary distribution π, we need π = π*P and sum(π) = 1
print("\nMethod 2: Solving the system of linear equations")
print("For stationary distribution, we solve: π = π*P and sum(π) = 1")

# For a 3-state Markov chain:
# π₁ = π₁*p₁₁ + π₂*p₂₁ + π₃*p₃₁
# π₂ = π₁*p₁₂ + π₂*p₂₂ + π₃*p₃₂
# π₃ = π₁*p₁₃ + π₂*p₂₃ + π₃*p₃₃
# π₁ + π₂ + π₃ = 1

# We can rewrite as (I-P^T)π = 0 with additional constraint sum(π) = 1
# We'll replace one equation with sum(π) = 1 to get a unique solution

# For the Weather Example:
# Step 1: Create the system of equations
I = identity_matrix(3)
P_weather_t = transpose(P_weather)
A = matrix_subtract(I, P_weather_t)  # (I - P^T)
A[2] = [1, 1, 1]           # Replace last row with the constraint equation
b = [0, 0, 1]              # Right hand side of the equations

# Step 2: Solve the system
steady_state = solve_linear_system(A, b)
print("\nCalculated steady state distribution for weather model:")
for i, state in enumerate(weather_states):
    print(f"{state}: {round(steady_state[i], 4)}")

# Verify the result: π = π*P
verification = vector_matrix_multiply(steady_state, P_weather)
print("\nVerification - The steady state multiplied by P should equal itself:")
for i, state in enumerate(weather_states):
    print(f"{state}: {round(verification[i], 4)}")
print(f"Difference between π and π*P: {sum(abs(steady_state[i] - verification[i]) for i in range(len(steady_state)))}")

# ============================================================================
# 5. CLASSIFICATION OF STATES
# ============================================================================
print("\n\n======= CLASSIFICATION OF STATES =======")
print("States in a Markov chain can be classified as:")
print("- Recurrent: The process will return to this state with probability 1")
print("- Transient: There's a non-zero probability of never returning")
print("- Absorbing: Once entered, the process never leaves (p_ii = 1)")
print("- Periodic: Returns occur at regular intervals")
print("- Aperiodic: Returns can occur at irregular intervals")

# Example of an absorbing Markov chain - The Gambler's Ruin
print("\nExample: Gambler's Ruin with $0 and $3 as absorbing states")
print("States represent money: $0, $1, $2, $3")
gambler_states = ["$0", "$1", "$2", "$3"]
P_gambler = [
    [1.0, 0.0, 0.0, 0.0],  # $0 is absorbing
    [0.5, 0.0, 0.5, 0.0],  # From $1, equal chance of going to $0 or $2
    [0.0, 0.5, 0.0, 0.5],  # From $2, equal chance of going to $1 or $3
    [0.0, 0.0, 0.0, 1.0]   # $3 is absorbing
]
print("\nTransition matrix for Gambler's Ruin:")
print_matrix(P_gambler, gambler_states, gambler_states)

# Identifying absorbing states
print("\nIdentifying absorbing states (p_ii = 1):")
is_absorbing = [P_gambler[i][i] == 1 for i in range(len(P_gambler))]
absorbing_states = [gambler_states[i] for i in range(len(is_absorbing)) if is_absorbing[i]]
print(f"Absorbing states: {', '.join(absorbing_states)}")

# ============================================================================
# 6. MEAN FIRST PASSAGE AND RECURRENCE TIMES
# ============================================================================
print("\n\n======= MEAN FIRST PASSAGE AND RECURRENCE TIMES =======")
print("Mean first passage time m_ij is the expected number of steps to reach state j from state i.")
print("Mean recurrence time m_ii is the expected return time to state i starting from i.")

# Let's use a simple 3-state example
print("\nExample: Simple 3-state Markov chain")
simple_states = ["A", "B", "C"]
P_simple = [
    [0.3, 0.6, 0.1],
    [0.4, 0.2, 0.4],
    [0.1, 0.5, 0.4]
]
print_matrix(P_simple, simple_states, simple_states)

print("\nCalculating mean first passage times using fundamental matrix method:")
print("For this calculation, we need to solve systems of linear equations.")

# Step 1: Find the steady state
I = identity_matrix(3)
P_simple_t = transpose(P_simple)
A = matrix_subtract(I, P_simple_t)
A[2] = [1, 1, 1]
b = [0, 0, 1]
pi_simple = solve_linear_system(A, b)
print("\nSteady state distribution:")
for i, state in enumerate(simple_states):
    print(f"{state}: {round(pi_simple[i], 4)}")

# Step 2: Calculate mean recurrence times
# For a recurrent irreducible Markov chain, mean recurrence time = 1/π_i
recurrence_times = [1 / p for p in pi_simple]
print("\nMean recurrence times (expected return time to each state):")
for i, state in enumerate(simple_states):
    print(f"{state}: {round(recurrence_times[i], 2)}")

print("\nFor complete first passage time calculations, we would need to solve:")
print("m_ij = 1 + Σ_k≠j p_ik * m_kj for all i≠j")
print("This requires solving systems of linear equations.")

# ============================================================================
# 7. ABSORPTION PROBABILITIES AND EXPECTED TIME TO ABSORPTION
# ============================================================================
print("\n\n======= ABSORPTION PROBABILITIES AND TIMES =======")
print("In a Markov chain with absorbing states, we're often interested in:")
print("1. The probability of absorption in each absorbing state")
print("2. The expected number of steps before absorption")

# Let's use the Gambler's Ruin example
print("\nFor Gambler's Ruin example:")
print_matrix(P_gambler, gambler_states, gambler_states)

# Step 1: Identify transient and absorbing states
transient_indices = [i for i in range(len(is_absorbing)) if not is_absorbing[i]]
absorbing_indices = [i for i in range(len(is_absorbing)) if is_absorbing[i]]

# Step 2: Decompose the transition matrix
# P = [ Q  R ]
#     [ 0  I ]
# Where:
# Q = transitions between transient states
# R = transitions from transient to absorbing states
# 0 = transitions from absorbing to transient (always 0)
# I = transitions between absorbing states (identity matrix)

Q = [[P_gambler[i][j] for j in transient_indices] for i in transient_indices]
R = [[P_gambler[i][j] for j in absorbing_indices] for i in transient_indices]

print("\nQ matrix (transitions between transient states):")
print_matrix(Q, [gambler_states[i] for i in transient_indices], [gambler_states[i] for i in transient_indices])
print("\nR matrix (transitions from transient to absorbing states):")
print_matrix(R, [gambler_states[i] for i in transient_indices], [gambler_states[i] for i in absorbing_indices])

# Step 3: Calculate the fundamental matrix N = (I-Q)^(-1)
I_Q = identity_matrix(len(Q))
I_minus_Q = matrix_subtract(I_Q, Q)
N = matrix_inverse(I_minus_Q)
print("\nFundamental matrix N = (I-Q)^(-1):")
print_matrix(N, [gambler_states[i] for i in transient_indices], [gambler_states[i] for i in transient_indices])

# Step 4: Calculate absorption probabilities B = N*R
B = matrix_multiply(N, R)
print("\nAbsorption probabilities (probability of ending in each absorbing state):")
print_matrix(B, [gambler_states[i] for i in transient_indices], [gambler_states[i] for i in absorbing_indices])

# Step 5: Calculate expected number of steps to absorption
ones = [[1] for _ in range(len(transient_indices))]
t = matrix_multiply(N, ones)
print("\nExpected number of steps before absorption from each transient state:")
for i, state in enumerate([gambler_states[i] for i in transient_indices]):
    print(f"{state}: {round(t[i][0], 2)}")

# ============================================================================
# 8. LONG-RUN BEHAVIOR OF MARKOV CHAINS
# ============================================================================
print("\n\n======= LONG-RUN BEHAVIOR OF MARKOV CHAINS =======")
print("The long-run behavior depends on the chain's structure:")
print("- Irreducible chains: Every state can reach every other state")
print("- Aperiodic chains: Returns do not occur at fixed intervals")
print("- Ergodic chains: Both irreducible and aperiodic\n")
print("For ergodic chains, P^n converges to a matrix with identical rows (steady state)")
print("For chains with absorbing states, the process eventually gets trapped in those states")

# Example of convergence to steady state
print("\nExample: Convergence of weather model to steady state")
print("Initial state: Starting in state 'Sunny'")
initial_state = [1, 0, 0]  # Start in state 'Sunny'

# Track evolution over time
print("\nEvolution of state probabilities over time:")
state_t = initial_state.copy()
print(f"t=0: {[round(p, 4) for p in state_t]}")

for t in range(1, 21):
    state_t = vector_matrix_multiply(state_t, P_weather)
    if t in [1, 2, 3, 5, 10, 20]:
        state_display = [f"{weather_states[i]}:{round(state_t[i], 4)}" for i in range(len(state_t))]
        print(f"t={t}: {state_display}")

print(f"\nSteady state for comparison: {[f'{weather_states[i]}:{round(steady_state[i], 4)}' for i in range(len(steady_state))]}")
print("Notice how the distribution converges to the steady state regardless of initial state.")

# ============================================================================
# 9. REDUCIBLE MARKOV CHAINS
# ============================================================================
print("\n\n======= REDUCIBLE MARKOV CHAINS =======")
print("A Markov chain is reducible if the state space can be divided into multiple classes")
print("where transitions between some classes are impossible.")

# Example of a reducible Markov chain
print("\nExample: Reducible Markov chain with two communication classes")
reducible_states = ["A", "B", "C", "D", "E"]
P_reducible = [
    [0.6, 0.4, 0.0, 0.0, 0.0],  # States A,B form one class
    [0.2, 0.8, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.3, 0.2],  # States C,D,E form another class
    [0.0, 0.0, 0.4, 0.4, 0.2],
    [0.0, 0.0, 0.1, 0.2, 0.7]
]
print("\nTransition matrix for reducible chain:")
print_matrix(P_reducible, reducible_states, reducible_states)

print("\nIn this example, states {A,B} and {C,D,E} form separate communication classes.")
print("Once the process enters one class, it cannot move to the other class.")
print("Each class will have its own steady-state distribution.")

# ============================================================================
# 10. APPLICATION AND PRACTICE EXAMPLE: PageRank Algorithm
# ============================================================================
print("\n\n======= APPLICATION EXAMPLE: SIMPLIFIED PAGERANK =======")
print("PageRank is a famous application of Markov chains used by Google to rank web pages.")
print("We'll implement a simplified version for a small network of 4 web pages.")

# Web page network with transition matrix
pages = ["Page A", "Page B", "Page C", "Page D"]
links = [
    [0.0, 0.5, 0.5, 0.0],  # Page A links to B and C
    [0.3, 0.0, 0.3, 0.4],  # Page B links to A, C and D
    [0.1, 0.8, 0.0, 0.1],  # Page C links to A, B and D
    [0.5, 0.2, 0.3, 0.0]   # Page D links to A, B and C
]
print("\nWeb page link structure (transition matrix):")
print_matrix(links, pages, pages)

# In PageRank, we add a "damping factor" (typically 0.85) to handle dead ends
# and add random jumps to any page
print("\nApplying damping factor (d=0.85) to create Google matrix")
damping = 0.85
n = len(pages)
teleport = [[1/n for _ in range(n)] for _ in range(n)]

# G = damping * links + (1 - damping) * teleport
G = [[damping * links[i][j] + (1 - damping) * teleport[i][j] for j in range(n)] for i in range(n)]
print("\nGoogle matrix with damping:")
print_matrix(G, pages, pages, 3)

# Find the steady state (PageRank) by power iteration
print("\nCalculating PageRank using power iteration")
pagerank = [1/n] * n  # Start with uniform distribution

for i in range(1, 51):
    pagerank_new = vector_matrix_multiply(pagerank, G)
    # Check for convergence
    if sum(abs(pagerank_new[j] - pagerank[j]) for j in range(n)) < 1e-6:
        print(f"Converged after {i} iterations")
        break
    pagerank = pagerank_new.copy()

print("\nPageRank values (importance of each page):")
for i, page in enumerate(pages):
    print(f"{page}: {round(pagerank[i], 4)}")

# Sort pages by PageRank
sorted_pages = sorted(zip(pages, pagerank), key=lambda x: x[1], reverse=True)
print("\nPages sorted by importance:")
for page, rank in sorted_pages:
    print(f"{page}: {round(rank, 4)}")

print("\n\n======= END OF MARKOV CHAIN EXAMPLES =======")
print("This file demonstrates key concepts of Markov chains using only built-in Python functions.")
print("For more advanced functionality, consider packages like 'numpy', 'scipy' or 'markov_chain'.")
