"""
I/O utilities for reading and writing fvecs/ivecs format files
"""

import numpy as np
import struct
from typing import Tuple, Optional


def read_fvecs(filename: str) -> np.ndarray:
    """
    Read vectors from fvecs format file.
    
    Args:
        filename: Path to the fvecs file
        
    Returns:
        numpy array of shape (n_vectors, dimension)
    """
    with open(filename, 'rb') as f:
        # Read the first 4 bytes to get dimension
        dim_bytes = f.read(4)
        if len(dim_bytes) != 4:
            raise ValueError(f"Invalid fvecs file: {filename}")
        
        dim = struct.unpack('<i', dim_bytes)[0]
        
        # Reset file pointer to beginning
        f.seek(0)
        
        # Read all data
        data = f.read()
        
        # Calculate number of vectors
        vector_size = 4 + dim * 4  # 4 bytes for dim + dim * 4 bytes for floats
        n_vectors = len(data) // vector_size
        
        if len(data) % vector_size != 0:
            raise ValueError(f"Invalid fvecs file format: {filename}")
        
        # Parse vectors
        vectors = []
        offset = 0
        
        for i in range(n_vectors):
            # Read dimension (should be same for all vectors)
            vec_dim = struct.unpack('<i', data[offset:offset+4])[0]
            if vec_dim != dim:
                raise ValueError(f"Inconsistent dimension in fvecs file: {filename}")
            offset += 4
            
            # Read vector data
            vector_data = struct.unpack(f'<{dim}f', data[offset:offset+dim*4])
            vectors.append(vector_data)
            offset += dim * 4
        
        return np.array(vectors, dtype=np.float32)


def read_ivecs(filename: str) -> np.ndarray:
    """
    Read integer vectors from ivecs format file.
    
    Args:
        filename: Path to the ivecs file
        
    Returns:
        numpy array of shape (n_vectors, dimension)
    """
    with open(filename, 'rb') as f:
        # Read the first 4 bytes to get dimension
        dim_bytes = f.read(4)
        if len(dim_bytes) != 4:
            raise ValueError(f"Invalid ivecs file: {filename}")
        
        dim = struct.unpack('<i', dim_bytes)[0]
        
        # Reset file pointer to beginning
        f.seek(0)
        
        # Read all data
        data = f.read()
        
        # Calculate number of vectors
        vector_size = 4 + dim * 4  # 4 bytes for dim + dim * 4 bytes for ints
        n_vectors = len(data) // vector_size
        
        if len(data) % vector_size != 0:
            raise ValueError(f"Invalid ivecs file format: {filename}")
        
        # Parse vectors
        vectors = []
        offset = 0
        
        for i in range(n_vectors):
            # Read dimension (should be same for all vectors)
            vec_dim = struct.unpack('<i', data[offset:offset+4])[0]
            if vec_dim != dim:
                raise ValueError(f"Inconsistent dimension in ivecs file: {filename}")
            offset += 4
            
            # Read vector data
            vector_data = struct.unpack(f'<{dim}i', data[offset:offset+dim*4])
            vectors.append(vector_data)
            offset += dim * 4
        
        return np.array(vectors, dtype=np.int32)


def write_fvecs(filename: str, vectors: np.ndarray) -> None:
    """
    Write vectors to fvecs format file.
    
    Args:
        filename: Output file path
        vectors: numpy array of shape (n_vectors, dimension)
    """
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    
    n_vectors, dim = vectors.shape
    
    with open(filename, 'wb') as f:
        for i in range(n_vectors):
            # Write dimension
            f.write(struct.pack('<i', dim))
            # Write vector data
            f.write(struct.pack(f'<{dim}f', *vectors[i]))


def write_ivecs(filename: str, vectors: np.ndarray) -> None:
    """
    Write integer vectors to ivecs format file.
    
    Args:
        filename: Output file path
        vectors: numpy array of shape (n_vectors, dimension)
    """
    if vectors.dtype != np.int32:
        vectors = vectors.astype(np.int32)
    
    n_vectors, dim = vectors.shape
    
    with open(filename, 'wb') as f:
        for i in range(n_vectors):
            # Write dimension
            f.write(struct.pack('<i', dim))
            # Write vector data
            f.write(struct.pack(f'<{dim}i', *vectors[i]))


def get_file_info(filename: str) -> Tuple[int, int]:
    """
    Get basic information about fvecs/ivecs file.
    
    Args:
        filename: Path to the file
        
    Returns:
        Tuple of (n_vectors, dimension)
    """
    with open(filename, 'rb') as f:
        # Read dimension from first vector
        dim_bytes = f.read(4)
        if len(dim_bytes) != 4:
            raise ValueError(f"Invalid file: {filename}")
        
        dim = struct.unpack('<i', dim_bytes)[0]
        
        # Get file size
        f.seek(0, 2)  # Seek to end
        file_size = f.tell()
        
        # Calculate number of vectors
        vector_size = 4 + dim * 4
        n_vectors = file_size // vector_size
        
        return n_vectors, dim