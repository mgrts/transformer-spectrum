import numpy as np


def slice_array_to_chunks(array, chunk_size=300):
    n = len(array)
    n_slices = (n + chunk_size - 1) // chunk_size  # ceil(n / chunk_size)
    chunks = []

    for i in range(n_slices):
        start = i * chunk_size
        end = start + chunk_size
        if end > n:
            end = n
            start = max(0, n - chunk_size)
        chunks.append(array[start:end])

    return np.array(chunks)