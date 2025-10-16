# from utils import calc_multinomial_BA

import torch
import numpy as np
import pickle
import itertools
import operator
from scipy.special import factorial, binom
from blahut_arimoto import blahut_arimoto
from sys import float_info


# ### Files ### #
def save_metrics(my_dict, file_path="training_metrics.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump(my_dict, f)


def load_metrics(file_path="training_metrics.pkl"):
    # Load from a file
    with open(file_path, "rb") as f:
        loaded_results = pickle.load(f)
    return loaded_results


# ### Information ### #
def calc_entropy(p_x):
    log_base = 2
    return -sum([a * np.log(a) for a in p_x if a > 0])/np.log(log_base)


def calc_information(p_y_x, r):
    eps = float_info.epsilon
    log_base = 2
    # The number of inputs: size of |X|
    m = p_y_x.shape[0]
    # The number of outputs: size of |Y|
    n = p_y_x.shape[1]

    q = (np.array([r])).T * p_y_x
    q = q / (np.sum(q, axis=0)+eps)

    c = 0
    for i in range(m):
        if r[i] > 0:
            c += np.sum(r[i] * p_y_x[i, :] *
                        np.log(q[i, :] / r[i] + 1e-16))
    c = c / np.log(log_base)
    return c


def kl_divergence(p, q):
    eps = 1e-16
    return np.sum(np.where(p != 0, p * np.log2(eps + p / (q + eps)), 0))


def point_kl(n, x, p_y_k):
    p = calc_multinomial_channel(np.array([x]), n)
    return kl_divergence(p, p_y_k)


def binomial_p(n, x, y):
    return binom(n, y)*(x**y)*((1-x)**(n-y))


def calc_binomial_channel(x, n):
    y = np.array(range(n+1))
    p_y_x = np.asarray([binomial_p(n, i, y) for i in x])
    return p_y_x


def calc_pyk(x, n):
    p = calc_multinomial_channel(x, n)
    I, r = blahut_arimoto(np.asarray(p))
    p_y_k = np.matmul(np.array([r]), p)
    return p_y_k, I, r


def calc_binomial_information(x, r, n):
    assert len(x) == len(r)
    p_y_x = calc_binomial_channel(x, n)
    C = calc_information(p_y_x, r)
    return C


def calc_multinomial_information(x, r, n):
    assert len(x) == len(r)
    p_y_x = calc_multinomial_channel(x,n)
    C = calc_information(np.asarray(p_y_x), r)
    return C


def calc_binomial_BA(x, n):
    p = calc_binomial_channel(x, n)
    C, r = blahut_arimoto(np.asarray(p))
    return C, r


def calc_BA_input_information(x, n):
    C, _ = calc_binomial_BA(x, n)
    return C


def calc_multinomial_BA_input_information(x, n):
    C, _ = calc_multinomial_BA(x, n)
    return C


def calc_equal_input_information(x, n):
    m = len(x)
    r = np.array([1/m]*m)
    return calc_binomial_information(x, r, n)


def calc_equal_input_multinomial_information(x, n):
    m = len(x)
    r = np.array([1/m]*m)
    return calc_multinomial_information(x, r, n)


# r - number of balls
def combinations_with_replacement_counts(r, n):
    size = n + r - 1
    for indices in itertools.combinations(range(size), r-1):
        starts = [0] + [index+1 for index in indices]
        stops = indices + (size,)
        yield np.array(list(map(operator.sub, stops, starts)))


def multinomial_coeff(c):
    return factorial(c.sum()) / factorial(c).prod()


def multinomial_p(n, x_vec, y_vec):
    assert sum(y_vec) == n
    if np.abs(sum(x_vec)) - 1 > 1e-4 :
        x_vec = x_vec/sum(x_vec)
    assert len(x_vec) == len(y_vec)
    power_vec = x_vec ** y_vec
    ret_val = multinomial_coeff(y_vec)*(power_vec.prod())
    if ret_val < 0:
        ret_val = 0
    return ret_val


def calc_multinomial_channel(x, n):
    dim = len(x[0])
    p_y_x = []

    for symbol in x:
        p_y_symbol = [multinomial_p(n, symbol, permutation) for permutation in combinations_with_replacement_counts(dim, n)]
        p_y_symbol = p_y_symbol/sum(p_y_symbol)
        p_y_x.append(p_y_symbol)
    return p_y_x


def calc_multinomial_BA(x, n):
    p = calc_multinomial_channel(x, n)
    C, r = blahut_arimoto(np.asarray(p))
    return C, r


# ### Deep Binomial ### #
def probs_to_logits(probs):
    # Ensure that the probabilities are valid (no zeroes, as log(0) is undefined)
    probs = torch.clamp(probs, min=1e-16)

    # Compute logits from probs
    logits = torch.log(probs)

    return logits


def get_p_from_explicit_model(model, condition_dim, device, multi=False):
    batch = torch.eye(condition_dim)
    batch = batch.to(device)
    if model.numerical_input:
        # Convert one-hot encoded input to numerical indices
        num_categories = batch.size(-1)  # Number of categories (size of one-hot vector)
        batch = batch.argmax(dim=-1, keepdim=True).float()  # Indices as float
        batch = batch / (num_categories - 1)  # Normalize to [0, 1]
    mean_p = model.encoder(batch)
    if not multi:
        mean_p = mean_p[:,0]
    return mean_p.detach().cpu().numpy()


def ret_triangle(res = 25, edge_res = None):

    n_x = res
    n_y = res
    xd = np.linspace(0, 1, n_x)
    yd = np.linspace(0, 1, n_y)

    if edge_res is not None:
        n_x_log = edge_res
        n_y_log = edge_res
        y = np.logspace(-10, -1, n_y_log)
        yd = np.concatenate([yd, y], 0)
        x = 1 - np.logspace(-10, -1, n_x_log)
        xd = np.concatenate([xd, x], 0)

    # Meshgrid points and triangle selection
    x, y = np.meshgrid(xd, yd)
    x = np.ravel(x)
    y = np.ravel(y)
    xy = list(zip(x, y))
    triangle = list(filter(lambda a: a[0] + a[1] <= 1, xy))
    t_len = len(triangle)
    t_x, t_y = zip(*triangle)
    t_z = [1- t_x[i]-t_y[i] for i in range(t_len)]

    return t_x, t_y, t_z


def run_decoder_2d(model, device="cpu", n_test=101, multi=False):

    model.eval()

    with torch.no_grad():
        # Encode to get latent space

        t_x, t_y, t_z = ret_triangle(n_test)
        probs = np.array([t_x, t_y, t_z]).T
        probs = torch.tensor(probs, dtype=torch.float32).to(device)
        output_data = model.decoder(probs)
        return output_data.cpu().numpy()


def get_multi_encoder(model, condition_dim, device="cpu"):
    from model_classes import ModifiedEncoder
    # Assuming `trained_model` is your already trained model
    modified_model = ModifiedEncoder(model)

    model.eval()

    with torch.no_grad():
        input_data = torch.eye(condition_dim).to(device)
        probs = modified_model(input_data, return_pre_activation=False).cpu()
    return probs


def calc_mean_kl(training_metrics):
    epochs = training_metrics["num_epochs"]
    assert len(training_metrics["running_p"]) == epochs
    assert len(training_metrics["running_weights"]) == epochs
    assert len(training_metrics["running_decoder"]) == epochs

    n_trials = training_metrics["n_trials"]
    res = training_metrics["res"]
    n_test = training_metrics["n_test"]
    assert n_test == n_trials * res + 1

    # convert decoder output to probabilities
    prob_vecs = np.array(training_metrics["running_decoder"])[:, 0:n_test:res, :]
    prob_vecs = np.exp(prob_vecs)
    sums = np.sum(prob_vecs, axis=-1, keepdims=True)
    prob_vecs = prob_vecs / sums

    running_mean_kl = []
    for ind in range(epochs):

        # The goal is to calc p(x|y)
        x = training_metrics["running_p"][ind]
        p_y_x = calc_binomial_channel(x, n=n_trials)  # p(y|x)
        M, N = np.shape(p_y_x) # M = input dimension , N = n_trials + 1

        p_x = training_metrics["running_weights"][ind]  # p(x)
        p_x_tiled = np.tile(p_x, (N, 1)).T
        p_xy = p_x_tiled * p_y_x  # p(x,y) = p(x)*p(y|x)
        p_y = np.sum(p_xy, axis=0) # p(y)
        p_y_tiled = np.tile(p_y, (M, 1))
        p_x_y = p_xy / p_y_tiled  # p(x|y) = p(x,y)/p(y)

        p_d = prob_vecs[ind] # p_d(x|y)

        mean_kl = 0
        for y in range(N):
            mean_kl += p_y[y] * kl_divergence(p_x_y.T[y], p_d[y])

        running_mean_kl.append(mean_kl)
    return running_mean_kl

