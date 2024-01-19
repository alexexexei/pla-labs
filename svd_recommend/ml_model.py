import numpy as np

# user-item_rating matrix
M = np.array([[5, 0, 1, 0, 4, 0, 0, 3, 4, 1],
              [0, 2, 5, 0, 0, 0, 0, 2, 0, 0],
              [0, 0, 0, 3, 2, 5, 0, 0, 0, 0],
              [0, 3, 5, 0, 0, 0, 2, 1, 0, 2],
              [1, 1, 5, 0, 0, 4, 4, 0, 5, 1],
              [3, 0, 3, 0, 4, 0, 2, 0, 0, 2],
              [5, 0, 0, 0, 3, 0, 0, 4, 3, 5],
              [0, 4, 0, 5, 0, 5, 0, 5, 0, 0],
              [0, 0, 0, 5, 0, 5, 5, 5, 4, 0]])

def count_not_null(M):
    not_null = []
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if (M[i, j] > 0):
                not_null.append((i, j))
    return not_null

def MSE(M, P, Q):
    mse = 0
    M_new = P * Q
    n = np.count_nonzero(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if (M[i, j] > 0):
                mse += (M[i, j] - M_new[i, j])**2 / n
    return mse

def SVD_recommend(M, l, n_iters, k):
    P = np.matrix(np.random.rand(M.shape[0], k))
    Q = np.matrix(np.random.rand(k, M.shape[1]))
    start_mse = MSE(M, P, Q)
    
    N = np.count_nonzero(M)
    not_null = count_not_null(M)

    for n in range(n_iters):
        choice = np.random.randint(0, len(not_null))
        i, j = not_null[choice]
        
        for a in range(k):
            P[i, a] = P[i, a] + (2 / N) * (M[i, j] - np.dot(P[i, :], Q[:, j])) * Q[a, j] - l * P[i, a]
            Q[a, j] = Q[a, j] + (2 / N) * (M[i, j] - np.dot(P[i, :], Q[:, j])) * P[i, a] - l * Q[a, j]

    mse = MSE(M, P, Q)

    return P, Q, start_mse, mse

k = 6
l = 0.01
n_iters = 5000

P, Q, s, m = SVD_recommend(M, l, n_iters, k)
print(P*Q)