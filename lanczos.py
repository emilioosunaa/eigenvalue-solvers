import numpy as np
from msr_utils import get_columns, msr_matrix_vector_product
from power_iteration import power_iteration_tridiagonal
import time
import matplotlib.pyplot as plt

def lanczos_algorithm(msr_matrix, m=100, tol=1e-10):
    """
    Lanczos method to find the largest eigenvalue of a symmetric matrix A.
    
    Parameters:
    msr_matrix (numpy.ndarray): MSR format symmetric matrix.
    m (int): Dimension of the Krylov space.
    tol (float): Tolerance for convergence.

    Returns:
    tuple: Approximation of the largest eigenvalue, corresponding eigenvector,
           number of iterations, and overall runtime.
    """
    # Extract information from MSR format file
    jm, vm = get_columns(msr_matrix)
    n = int(jm[0])

    # Initialization
    diagonal, off_diagonal = [], []
    v_k = np.ones(n) / np.sqrt(n)
    v_km1, beta = 0.0, 0.0
    
    start_time = time.time()

    for i in range(m):
        # Iteration steps
        w_prime = msr_matrix_vector_product(jm[1:], vm[1:], v_k, symmetric=True)
        conj = np.matrix.conjugate(w_prime)
        alpha = np.dot(conj, v_k)
        w = w_prime - alpha * v_k - beta * v_km1
        beta = np.linalg.norm(w)

        diagonal.append(np.linalg.norm(alpha))
        if i < (n-1):
            off_diagonal.append(beta)

        # Update
        v_km1 = v_k
        v_k = w / beta
    
    # Power iteration method to find the largest eigenvalue
    eigenvalue, eigenvector, k_iter = power_iteration_tridiagonal(diagonal, off_diagonal, tol)

    overall_runtime = time.time() - start_time
    
    return eigenvalue, eigenvector, k_iter, overall_runtime


if __name__ == "__main__":
    msr_matrix = "cg_test_msr.txt"
    m_list = [30, 50, 75, 100]
    tol_list = [1e-2, 1e-4, 1e-6, 1e-10]
    k_iterations = []
    final_errors = []
    overall_runtimes = []

    for i in range(len(m_list)):
        eigenvalue, eigenvector, k_iter, overall_runtime = lanczos_algorithm(msr_matrix, m_list[i], tol_list[i])
        final_error = abs(eigenvalue - 9.5986080894852857E+03)
        k_iterations.append(k_iter)
        final_errors.append(final_error)
        overall_runtimes.append(overall_runtime)
        print(f"m = {m_list[i]}, tol = {tol_list[i]}")
        print(f"Largest eigenvalue with power iteration method: {eigenvalue}")
        print(f"Number of k-iterations: {k_iter}")
        print(f"Overall runtime: {overall_runtime:.4f} seconds")
        print(f"Final error: {final_error}")

    plt.figure(figsize=(10, 6))
    plt.semilogy(overall_runtimes, final_errors, marker='x', linestyle='-', color='b')
    plt.xlabel('Runtime (seconds)', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig('error_vs_runtime_lanczos.png', dpi=300)
    plt.show()

