import numpy as np 

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    m = len(X)
    n = len(Y)
    result = np.zeros([m,n])
    for i in range(m):
        for j in range(n):
            temp = np.linalg.norm(X[i]-Y[j])
            result[i][j] = temp
    return result


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    m = len(X)
    n = len(Y)
    result = np.zeros([m,n])
    for i in range(m):
        for j in range(n):
            temp = np.sum(np.abs(X[i]-Y[j]))
            result[i][j] = temp
    return result


def cosine_distances(X, Y):
    """Compute Cosine distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    m = len(X)
    n = len(Y)
    result = np.zeros([m,n])
    for i in range(m):
        for j in range(n):
            temp=np.dot(X[i], Y[j])/(np.linalg.norm(X[i])*np.linalg.norm(Y[j]))
            result[i][j] = 1-temp
    return result


