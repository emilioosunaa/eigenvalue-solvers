import numpy as np
from msr_utils import get_columns, msr_matrix_vector_product
from power_iteration import power_iteration_tridiagonal
import time
import matplotlib.pyplot as plt

def lanczos_algorithm(msr_matrix, m=100, tol=1e-10):
    """
    Lanczos method to find the largest eigenvalue of a symmetric matrix A.
    
    Parameters:
    A (numpy.ndarray): Symmetric matrix.
    m (int): dimension of the Krylov space.

    Returns:
    float: Approximation of the largest eigenvalue.
    """
    # Extract information from MSR format file
    jm, vm = get_columns(msr_matrix)
    n = int(jm[0])

    # Initialization
    v_jm1 = np.zeros(n)
    v_j = np.ones(n) / np.sqrt(n)
    alpha = np.zeros(m)
    beta = np.zeros(m)
    
    start_time = time.time()

    # Lanczos algorithm
    for j in range(m):
        w = msr_matrix_vector_product(jm[1:], vm[1:], v_j, symmetric=True) - beta[j - 1] * v_jm1
        alpha[j] = np.dot(v_j.conj().T, w)
        w = w - alpha[j] * v_j
        beta[j] = np.linalg.norm(w)

        v_jm1 = v_j
        v_j = w / beta[j]
    
    beta = beta[:m - 1]  # Note: beta[m] is not needed
    
    # Power iteration method to find the largest eigenvalue
    eigenvalue, eigenvector, k_iter = power_iteration_tridiagonal(alpha, beta, tol)

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
    plt.semilogy(overall_runtimes, final_errors, linestyle='-', color='b')
    plt.xlabel('Runtime (seconds)', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig('error_vs_iteration_power.png', dpi=300)
    plt.show()

