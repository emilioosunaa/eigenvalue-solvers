import numpy as np
import matplotlib.pyplot as plt

def msr_to_dense(jm, vm, symmetric=False):
    """
    Convert a matrix in MSR format to a dense matrix.

    Args:
        jm (numpy.ndarray): Index array for MSR format.
        vm (numpy.ndarray): Value array for MSR format.
        symmetric (bool): Whether the matrix is symmetric.

    Returns:
        numpy.ndarray: Dense matrix representation.
    """
    # Adjust the indexes to 0-based
    jm = jm - 1

    # Determine the number of rows/columns (n)
    n = get_dimension(vm)

    # Initialize an n x n matrix with zeros
    A = np.zeros((n, n))

    # Fill in the diagonal elements
    for i in range(n):
        A[i, i] = vm[i]

    # Fill in the off-diagonal elements
    if symmetric:
        for i in range(n):
            start = jm[i]
            end = jm[i + 1]

            for j in range(start, end):
                col_index = jm[j]
                A[i, col_index] = vm[j]
                A[col_index, i] = vm[j]
    else:
        for i in range(n):
            start = jm[i]
            end = jm[i + 1]

            for j in range(start, end):
                col_index = jm[j]
                A[i, col_index] = vm[j]

    return A

def get_dimension(arr):
    """
    Determine the dimension of the matrix based on the value array.

    Args:
        arr (numpy.ndarray): Value array.

    Returns:
        int: Dimension of the matrix.
    """
    if 0 in arr:
        index = np.where(arr == 0)[0][0]
        return index
    else:
        return None

def get_columns(file_path):
    """
    Read the MSR format columns from a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        tuple: Two numpy arrays, jm and vm respectively.
    """
    col1 = []
    col2 = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 2:  # Ensure the line has exactly two columns
                col1.append(int(parts[0]))
                col2.append(float(parts[1]))

    return np.array(col1), np.array(col2)

def csr_matrix_vector_product(ia, ja, va, x):
    """
    Perform matrix-vector multiplication for a matrix stored in CSR format.

    Args:
        ia (numpy.ndarray): Array of row indices.
        ja (numpy.ndarray): Array of column indices.
        va (numpy.ndarray): Array of non-zero values.
        x (numpy.ndarray): Input vector for multiplication.

    Returns:
        numpy.ndarray: Resulting vector after multiplication.
    """
    y = np.zeros(len(x))

    for i in range(len(ia) - 1):
        start, end = ia[i], ia[i + 1]
        for j in range(start, end):
            y[i] += va[j] * x[ja[j]]

    return y

def csc_matrix_vector_product(ia, ja, va, x):
    """
    Perform matrix-vector multiplication for a matrix stored in CSC format.

    Args:
        ia (numpy.ndarray): Array of column indices.
        ja (numpy.ndarray): Array of row indices.
        va (numpy.ndarray): Array of non-zero values.
        x (numpy.ndarray): Input vector for multiplication.

    Returns:
        numpy.ndarray: Resulting vector after multiplication.
    """
    y = np.zeros(len(x))

    for i in range(len(ia) - 1):
        start, end = ia[i], ia[i + 1]
        for j in range(start, end):
            y[ja[j]] += va[j] * x[i]

    return y

def msr_matrix_vector_product(jm, vm, x, symmetric=False):
    """
    Perform matrix-vector multiplication for a matrix stored in MSR format.

    Args:
        jm (numpy.ndarray): Index array for MSR format.
        vm (numpy.ndarray): Value array for MSR format.
        x (numpy.ndarray): Input vector for multiplication.
        symmetric (bool): Whether the matrix is symmetric.

    Returns:
        numpy.ndarray: Resulting vector after multiplication.
    """
    n = len(x)
    jm = jm - 1  # Convert to 0-based indexing
    diagonal = vm[:n]
    off_diag_values = vm[n+1:]
    ia = jm[:n + 1] - (n + 1)
    ja = jm[n + 1:]

    y_offdiagonal = csr_matrix_vector_product(ia, ja, off_diag_values, x)
    if symmetric:
        y_offdiagonal += csc_matrix_vector_product(ia, ja, off_diag_values, x)

    y_diagonal = diagonal * x

    return y_offdiagonal + y_diagonal

def save_matrix_to_csv(matrix, file_path):
    """
    Save a matrix to a CSV file.

    Args:
        matrix (numpy.ndarray): Matrix to save.
        file_path (str): Path to the output file.
    """
    np.savetxt(file_path, matrix, delimiter=',')

def plot_matrix(matrix):
    """
    Plot a matrix with zero values shown in white.

    Args:
        matrix (numpy.ndarray): Matrix to plot.
    """
    # Create a custom colormap
    cmap = plt.cm.viridis
    cmap.set_bad(color='white')  # Set the color for masked values to white

    # Mask the zero values
    masked_matrix = np.ma.masked_where(matrix == 0, matrix)

    plt.figure(figsize=(5, 5))
    plt.imshow(masked_matrix, cmap=cmap, aspect='equal', interpolation='none')
    plt.colorbar()
    plt.title('Matrix Visualization (Zero Values in White)')
    plt.show()

if __name__ == "__main__":
    # Test 1
    jm = np.array([6, 7, 8, 8, 9, 3, 3, 3])
    vm = np.array([1, 2, 1, 2, 0, 4, 2, 4])
    x = np.ones(4)
    A = msr_to_dense(jm, vm)

    y_msr = msr_matrix_vector_product(jm, vm, x)
    y_dense = np.dot(A, x)

    print("Test 1, 4x4 matrix")
    print("Matrix:\n", A)
    print("MSR result:", y_msr)
    print("Dense result:", y_dense)
    print("Difference:", np.array(y_msr) - y_dense)

    # Test 2, symmetric matrix
    jm = np.array([6, 6, 7, 9, 11, 1, 1, 2, 2, 3])
    vm = np.array([10, 20, 30, 40, 0, 2, 3, 4, 5, 6])
    x = np.ones(4)
    A = msr_to_dense(jm, vm, symmetric=True)

    y_msr = msr_matrix_vector_product(jm, vm, x, symmetric=True)
    y_dense = np.dot(A, x)

    print("Test 2, symmetric 4x4 matrix")
    print("Matrix:\n", A)
    print("MSR result:", y_msr)
    print("Dense result:", y_dense)
    print("Difference:", np.array(y_msr) - y_dense)

    # Test 3, with big symmetric matrix
    file_path = 'cg_test_msr.txt'
    jm, vm = get_columns(file_path)
    n = get_dimension(vm[1:])
    x = np.ones(n)
    A = msr_to_dense(jm[1:], vm[1:], symmetric=True)

    y_msr = msr_matrix_vector_product(jm[1:], vm[1:], x, symmetric=True)
    y_dense = np.dot(A, x)

    difference = y_msr - y_dense
    plot_matrix(A)

    print("Test 3, symmetric matrix: cg_test_msr.txt")
    print("Sum of difference:", np.sum(np.abs(difference)))

    # Test 4, with big non-symmetric matrix
    file_path = 'gmres_test_msr.txt'
    jm, vm = get_columns(file_path)
    n = get_dimension(vm[1:])
    x = np.ones(n)
    A = msr_to_dense(jm[1:], vm[1:])

    y_msr = msr_matrix_vector_product(jm[1:], vm[1:], x)
    y_dense = np.dot(A, x)

    difference = y_msr - y_dense
    plot_matrix(A)

    print("Test 4, symmetric matrix: gmres_test_msr.txt")
    print("Sum of difference:", np.sum(np.abs(difference)))