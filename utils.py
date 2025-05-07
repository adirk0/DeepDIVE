# from utils import calc_multinomial_BA

import torch
import numpy as np
import pickle
import itertools
import operator
from scipy.special import factorial
from blahut_arimoto import blahut_arimoto

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


def save_metrics(my_dict, file_path="training_metrics.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump(my_dict, f)


def probs_to_logits(probs):
    # Ensure that the probabilities are valid (no zeroes, as log(0) is undefined)
    probs = torch.clamp(probs, min=1e-16)

    # Compute logits from probs
    logits = torch.log(probs)

    return logits