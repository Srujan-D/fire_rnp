import numpy as np

def generate_samples(matrix, N, M):
    """
    Generate N * M samples for each timestamp from a 3D matrix.

    Parameters:
    - matrix: 3D NumPy array of shape (7, 30, 30)
    - N: Number of samples per agent
    - M: Number of agents

    Returns:
    - samples: NumPy array of shape (N, M, 7, 30, 30)
    """

    # Get the shape of the matrix
    T, I, J = matrix.shape

    # Initialize the array to store samples
    samples = np.zeros((N, M, T, I, J))

    # Generate samples
    for n in range(N):
        for m in range(M):
            # Choose all timestamps
            time_indices = np.arange(T)
            
            # Randomly choose indices for i and j
            i_indices = np.random.randint(0, I, size=(T,))
            j_indices = np.random.randint(0, J, size=(T,))

            # Use the chosen indices to extract values from the matrix
            samples[n, m, :, :, :] = matrix[time_indices, i_indices, j_indices]

    return samples

# Example usage:
# Assuming you have a 3D matrix called 'your_matrix' of shape (7, 30, 30)
N = 5  # Number of samples per agent
M = 3  # Number of agents

result_samples = generate_samples(your_matrix, N, M)
print(result_samples.shape)  # Should print (N, M, 7, 30, 30)
