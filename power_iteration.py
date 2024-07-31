import numpy as np
from msr_utils import get_columns, msr_matrix_vector_product, msr_to_dense
import time
import matplotlib.pyplot as plt

def power_iteration_msr(msr_matrix, tol=1e-8, maxiter=1000):
    """
    Perform power iteration to find the largest eigenvalue and its corresponding eigenvector.
    
    Parameters:
    msr_matrix (string): Path to the .txt where A is stored in MSR format. A is symmetric positive definite
    maxiter (int, optional): Maximum number of iterations
    tol (float, optional): tolerance for convergence
    
    Returns:
    eigenvalue (float): The largest eigenvalue
    eigenvector (numpy.ndarray): The corresponding eigenvector
    errors (list): List of errors at each iteration
    iteration_indices (numpy.ndarray): Array of iteration indices
    runtime (float): Total time taken for the computation loop
    """
    # Extract information from MSR format file
    jm, vm = get_columns(msr_matrix)
    n = int(jm[0])
    
    # Initialization
    q_k = np.ones(n) / np.sqrt(n)
    lambda_k = 0.0
    errors = []
    runtimes = []

    start_time = time.time()
    
    # Initial matrix-by-vector product
    q_k1 = msr_matrix_vector_product(jm[1:], vm[1:], q_k, symmetric=True)

    for _ in range(maxiter):
        # Take a power method step and compute Rayleigh quotient
        q_k = q_k1 / np.linalg.norm(q_k1)
        q_k1 = msr_matrix_vector_product(jm[1:], vm[1:], q_k, symmetric=True)
        lambda_k1 = np.dot(q_k.conj().T, q_k1)
        
        # Check for convergence
        err = lambda_k1 - lambda_k
        if np.abs(lambda_k1 - lambda_k) < tol:
            break
        
        # Update step
        lambda_k = lambda_k1
        runtimes.append(time.time() - start_time)
        errors.append(np.abs(err))
    
    overall_runtime = time.time() - start_time
    iteration_indices = np.arange(len(errors))

    return lambda_k, q_k, errors, iteration_indices, runtimes, overall_runtime

def power_iteration_tridiagonal(diagonal, off_diagonal, tol=1e-8, maxiter=4000):
    """
    Perform power iteration to find the largest eigenvalue and its corresponding eigenvector.
    
    Parameters:
    diagonal (np.array): The main diagonal of the matrix.
    off_diagonal (np.array): The lower and upper diagonal of the matrix (they are the same)
    maxiter (int, optional): Maximum number of iterations
    tol (float, optional): tolerance for convergence
    
    Returns:
    eigenvalue (float): The largest eigenvalue
    eigenvector (numpy.ndarray): The corresponding eigenvector
    """
    n = len(diagonal)
    
    # Initialization
    # For clarification, q_k correspond to q_k-1 and q_k1 to q_k.
    q_k = np.ones(n) / np.sqrt(n)
    lambda_k = 0.0
    
    # Initial matrix-by-vector product
    q_k1 = tridiagonal_matrix_vector_product(diagonal, off_diagonal, q_k)
    
    for iter in range(maxiter):
        # Take a power method step and compute Rayleigh quotient
        q_k = q_k1 / np.linalg.norm(q_k1)
        q_k1 = tridiagonal_matrix_vector_product(diagonal, off_diagonal, q_k)
        lambda_k1 = np.dot(q_k.conj().T, q_k1)
        
        # Check for convergence
        if np.abs(lambda_k1 - lambda_k) < tol:
            break
        
        # Update step
        lambda_k = lambda_k1

    return lambda_k, q_k, iter

def tridiagonal_matrix_vector_product(diagonal, off_diagonal, x):
    """
    Compute the product of a symmetric tridiagonal matrix and a vector.
    
    Parameters:
    diagonal (np.array): The main diagonal of the matrix.
    off_diagonal (np.array): The lower and upper diagonal of the matrix (they are the same).
    x (np.array): The vector to be multiplied.

    Returns:
    np.array: The result of the matrix-vector multiplication.
    """
    n = len(diagonal)
    y = np.zeros_like(x)

    # Compute the matrix-vector product
    for i in range(n):
        y[i] = diagonal[i] * x[i]
        if i > 0:
            y[i] += off_diagonal[i-1] * x[i-1]
        if i < n - 1:
            y[i] += off_diagonal[i] * x[i+1]
    
    return y

if __name__ == "__main__":
    # # Test case for power iteration function with power_iteration_msr.txt
    # msr_matrix = "power_test_msr.txt"
    # eigenvalue, eigenvector, errors, iteration_indices, runtimes, overall_runtime = power_iteration_msr(msr_matrix)
    # print(f"Matrix: -------------{msr_matrix}-------------")
    # print(f"Largest eigenvalue with msr format power iteration method: {eigenvalue}")
    # print(f"Number of iterations: {iteration_indices[-1]}")
    # print(f"Computation Time: {overall_runtime} seconds")

    # jm, vm = get_columns(msr_matrix)
    # A = msr_to_dense(jm[1:], vm[1:], symmetric=True)
    # eigenvalues, eigenvectors = np.linalg.eig(A)
    # print(f"Largest eigenvalue with numpy method: {max(eigenvalues)}")
    
    # # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.semilogy(iteration_indices, errors, linestyle='-', color='b')
    # plt.xlabel('Iteration', fontsize=14)
    # plt.ylabel('Error', fontsize=14)
    # plt.grid(True, which="both", ls="--")
    # plt.tight_layout()
    # plt.savefig('error_vs_iteration_power.png', dpi=300)
    # plt.show()

    # # Test case for power iteration function with cg_test_msr.txt
    # msr_matrix = "cg_test_msr.txt"
    # eigenvalue, eigenvector, errors, iteration_indices, runtimes, overall_runtime = power_iteration_msr(msr_matrix, maxiter=3000)
    # print(f"Matrix: -------------{msr_matrix}-------------")
    # print(f"Largest eigenvalue with msr format power iteration method: {eigenvalue}")
    # print(f"Number of iterations: {iteration_indices[-1]}")
    # print(f"Computation Time: {overall_runtime} seconds")

    # jm, vm = get_columns(msr_matrix)
    # A = msr_to_dense(jm[1:], vm[1:], symmetric=True)
    # eigenvalues, eigenvectors = np.linalg.eig(A)
    # print(f"Largest eigenvalue with numpy method: {max(eigenvalues)}")
    
    # # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.semilogy(iteration_indices, errors, linestyle='-', color='b')
    # plt.xlabel('Iteration', fontsize=14)
    # plt.ylabel('Error', fontsize=14)
    # plt.grid(True, which="both", ls="--")
    # plt.tight_layout()
    # plt.savefig('error_vs_iteration_cg.png', dpi=300)
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.semilogy(iteration_indices, runtimes, linestyle='-', color='r')
    # plt.xlabel('Iteration', fontsize=14)
    # plt.ylabel('Runtime (seconds)', fontsize=14)
    # plt.grid(True, which="both", ls="--")
    # plt.tight_layout()
    # plt.savefig('error_vs_runtime_cg_2.png', dpi=300)
    # plt.show()

    # Test case for tridiagonal matrix-vector product
    diagonal = np.array([4, 4, 4, 4])
    off_diagonal = np.array([1, 1, 1])
    matrix = np.array([[4, 1, 0, 0],
                      [1, 4, 1, 0],
                      [0, 1, 4, 1],
                      [0, 0, 1, 4]])
    x = np.array([1, 2, 3, 4])

    y_tridiagonal = tridiagonal_matrix_vector_product(diagonal, off_diagonal, x)
    y_numpy = np.dot(matrix, x)
    print(f"Tridiagonal matrix-vector product result implementation: {y_tridiagonal}")
    print(f"Numpy matrix-vector product result: {y_numpy}")

    # Test case for tridiagonal power iteration
    eigenvalue, eigenvector, iter = power_iteration_tridiagonal(diagonal, off_diagonal)
    print(f"Largest eigenvalue for tridiagonal matrix power iteration method : {eigenvalue}")
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    print(f"Largest eigenvalue with numpy method: {max(eigenvalues)}")


