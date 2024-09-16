'''
Module: lindependence

This module contains the implementation of the Gram-Schmidt algorithm.
'''


def norm(vector):
    return vector / np.linalg.norm(vector)


def gram_schmidt(vectors):
    ''' 
    The Gram-Schmidt algorithm is used for determining whether a set of vectors are linearly independent. 
    It works by orgthogonalizing each initial vector one by one. After orthogonalizing one vector it checks whether it is equal to zero or not.
    If the ortogonalized vector is zero then the set of vectors is linearly dependent; otherwise, the algorithm return the orthonormal set of the initial vectors.

    Function:
        - `gram_schmidt`: Test for linear independence using Gram-Schmidt algorithm.
        
        Parameters:
        - `vectors` list[np.array]: A list of vectors.

        Returns:
        - `list[np.array]|None`: Returns the orthonormal basis if `vectors` are linearly independent or `None` if they are not.
        
    Example:
        >>> a1 = np.array([1., 2.])
        >>> a2 = np.array([2., -1.])
        >>> result = gram_schmidt([a1, a2])
        >>> print(result)
        [array([0.4472136 , 0.89442719]), array([ 0.89442719, -0.4472136 ])]

    Example 2:
        >>> a1 = np.array([1., 2.])
        >>> a2 = np.array([0., 1.])
        >>> result = gram_schmidt([a1, a2])
        >>> result
        None

    '''
    k = len(vectors)
    if not k:
        raise ValueError('An empty list was provided.')

    try:
        sum(vectors)
    except ValueError:
        raise ValueError('The provided vectors have differing dimensions.')
    n = vectors[0].shape[0]

    q = [norm(vectors[0])]
    for i in range(k)[1:]:
        # 1. Orthogonalize the vector q_i = a_i - (q_1^T * a_i) * q_1 - ... - (q_(i-1)^T * a_i) * q_(i-1)
        v = vectors[i] - sum([(q[j].T @ vectors[i]) * q[j] for j in range(i)])
        v = np.around(v, 10)
        # 2. Test linear independence, if v = 0 then quit
        if not v.any():
            return
        # 3. Normalize the vector, q = q / ||q||
        v = v / np.linalg.norm(v)
        q.append(v)

    return q


if __name__ == '__main__':
    import numpy as np

    # Create two skew-symmetric matrices
    A1 = np.array([[ 0,  1,  1],
                   [-1,  0,  1],
                   [-1, -1,  0]])
    
    A2 = np.array([[ 0,  1, -1],
                   [-1,  0,  1],
                   [ 1, -1,  0]])
    
    # Create base vector
    v1 = np.array([3, 7, 2])
    # Create two other orthogonal vectors using the skew-symmetric matrices
    v2 = A1 @ v1
    v3 = A2 @ v1
    print('Determine if the following vectors are linearly independent using Gram-Schmidt algorithm:', v1, v2, v3, sep='\n')

    # This will return the orthonormal basis vectors of the v vectors.
    print(f'Orthonormal basis of the vectors: {gram_schmidt([v1, v2, v3])=}')

    # This will return None
    print(gram_schmidt([v1, v2, v3, np.array([0., 0., 1.])]))
