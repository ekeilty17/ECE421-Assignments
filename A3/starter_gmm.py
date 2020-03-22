import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

# For Validation set
if is_valid:
    valid_batch = int(num_pts / 3.0)
    np.random.seed(45689)
    rnd_idx = np.arange(num_pts)
    np.random.shuffle(rnd_idx)
    val_data = data[rnd_idx[:valid_batch]]
    data = data[rnd_idx[valid_batch:]]

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the squared pairwise distance matrix (NxK)

    # Umm, I don't think we actually need to do this one since log_GaussPDF literally does the same thing... (2.1 uses the
    # log probability density function as its distance function

    return None

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1

    # Outputs:
    # log Gaussian PDF N X K

    K, N, D = len(mu), len(X), len(X[0])

    pair_dist = []
    for x in X:
        row = []
        for k, mu_k in enumerate(mu):
            # Note that (x-mu).T @ sigma[k]*{dxd identity matrix} @ (x-mu) = 1/sigma[k] * ||x-mu||
            # since sigma[k] is a constant

            # Note that the determinant of a constant times an nxn Identity matrix is the constant raised to n
            # because determinant is linear in each column and the determinant of the identity matrix is 1
            prob = np.exp(-0.5 * np.square(x - mu_k).sum() / sigma[k]) / np.sqrt((2 * np.pi) ** K * sigma[k] ** D)
            row.append(np.log(prob))
        pair_dist.append(row)

    # So, the output here is an NxK matrix. The i,j entry represent the log probability of the the i-th example
    # belonging to cluster j.
    # This is a distribution, so the sum of all the probabilities should give 1. To check this, don't forget to
    # take the exponential of each term first (since it's log prob)
    return np.array(pair_dist)


def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # So this essentially just normalizes each row of the log_PDF so that each row is a valid probability distribution
    # idk how tensor stuff works, so I'm going to leave you with a breakdown of what this fcn needs to do Sandra

    # The input log_PDF is an NxK matrix. Each entry needs to have a term subtracted from it. the update is as follows
    # log_PDF[i][j] = log_PDF[i][j] - log_sumOfExponentialsOfAllEntriesInRowi

    # that's literally it
    # Fortunately, the helper function called reduce_logsumexp calculates exactly this.

    # reduce_logsumexp should take in the log_PDF matrix (in the form of a tf tensor) and it will output a tensor
    # of dimension Nx1. Please do some vector magic and subtract the i-th entry of this tensor from every
    # entry in the i-th row of log_PDF
    # Thanks!
    return log_PDF


def neg_log_likelihood_gmm(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_likelihood: scalar

    """
    Not sure if this can be vectorized if need for speed
    """
    neg_log_likelihood = 0
    for n, sample in enumerate(log_PDF):
        prob = 0
        for k, clusters in enumerate(sample):
            prob += np.exp(log_pi[k] + log_PDF[n][k])
        neg_log_likelihood -= np.log(prob)

    return neg_log_likelihood
