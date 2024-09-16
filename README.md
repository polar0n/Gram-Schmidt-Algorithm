# Gram-Schmidt Algorithm

## Description
Implements Gram-Schmidth Algorithm in Python.

## Algorithm
Given a set of vectors $a_1, a_2, ..., a_k$ determine if the vectors are linearly independent.

1. _Orthogonalization._ $\widetilde{q_i}=a_1-(q_1^Ta_i)q_1-...-(q_{i-1}^Ta_i)q_{i-1}$
2. _Test for linear dependence._ If $\widetilde{q_i} = 0$ then quit
3. _Normalization._ $q_i=\frac{\widetilde{q_i}}{||\widetilde{q_i}||}$

## Usage

```python
from lindependence import gram_schmidt
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
print(gram_schmidt([v1, v2, v3]))
