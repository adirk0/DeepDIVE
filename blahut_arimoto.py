import numpy as np

# This file contains code adapted from:
# https://github.com/kobybibas/blahut_arimoto_algorithm
def blahut_arimoto(p_y_x: np.ndarray,  log_base: float = 2, thresh: float = 1e-12, max_iter: int = 1e3) -> tuple:
    '''
    Maximize the capacity between I(X;Y)
    p_y_x: each row represents probability assignment
    log_base: the base of the log when calculating the capacity
    thresh: the threshold of the update, finish the calculation when getting to it.
    max_iter: the maximum iterations of the calculation
    '''

    assert np.abs(p_y_x.sum(axis=1).mean() - 1) < 1e-6
    assert p_y_x.shape[0] > 1

    # The number of inputs: size of |X|
    m = p_y_x.shape[0]
    # The number of outputs: size of |Y|
    n = p_y_x.shape[1]
    # Initialize the prior uniformly
    r = np.ones((1, m)) / m

    # Compute the r(x) that maximizes the capacity
    for iteration in range(int(max_iter)):

        q = r.T * p_y_x
        q = q / (np.sum(q, axis=0) + 1e-16)
        r1 = np.array([np.prod(np.power(q, p_y_x), axis=1)])
        r1 = r1 / np.sum(r1)
        tolerance = np.linalg.norm(r1 - r)
        r = r1
        if tolerance < thresh:
            break

    # Calculate the capacity
    r = r.flatten()
    c = 0
    for i in range(m):
        if r[i] > 0:
            c += np.sum(r[i] * p_y_x[i, :] *
                        np.log(q[i, :] / r[i] + 1e-16))
    c = c / np.log(log_base)
    return c, r
