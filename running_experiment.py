import torch
import torch.optim as optim
import torch.nn.functional as F
from model_classes import Multinomial_VAE
from train import train_multinomial, train_multinomial_BA
from utils import save_metrics

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_file = 'running_d/test_model_multi_BA_3d_6clean_next6_'
# dict_file = 'running_d/training_metrics_multi_BA_3d_6clean_next6_'
# running_p_file="running_d/running_p_multi_BA_new_next6.pkl"
# running_w_file="running_d/running_w_multi_BA_new_next6.pkl"

test_number = 2
date = 705
model_file = f'results/test_model_{date}_{test_number}_'
dict_file = f'results/training_metrics_{date}_{test_number}_'
running_p_file = f"results/running_p_{date}_{test_number}.pkl"
running_w_file = f"results/running_w_{date}_{test_number}.pkl"

# Model hyperparameters
hidden_dim = 256
hidden_amount = 1
tau = 0.01

# Channel parameters
latent_dim = 3
n_trials = 10

# Visualization parameters
res = 10
n_test = n_trials*res + 1

# Training parameters
increase_points = 6
batch_size_without_BA = int(2 ** 12)
initial_batch_size = int(2 ** 12)
max_batch_size = int(2 ** 15)
growth_factor = 2
num_epochs = 2


def run_multinomial(iter=0, n_trials=12, use_BA=True, input_dim=5):
    model = Multinomial_VAE(input_dim, hidden_dim, hidden_amount, latent_dim, n_trials, tau=tau, device=device).to(
        device)
    loss_function = F.cross_entropy
    optimizer = optim.Adam(model.parameters())  # default parameters: lr=1e-3, betas=(0.9, 0.999))

    if use_BA:
        training_metrics = train_multinomial_BA(model, optimizer, input_dim, num_epochs, initial_batch_size,
                                                max_batch_size, increase_points, growth_factor, res, n_test, device)
    else:
        training_metrics = train_multinomial(model, optimizer, loss_function, input_dim, num_epochs,
                                             batch_size_without_BA, res, n_test, device)

    torch.save(model.state_dict(), model_file + str(input_dim) + '_' + str(iter) + '.pth')
    save_metrics(training_metrics, dict_file + str(input_dim) + '_' + str(iter) + '.pkl')
    # Save to a file
    print("Model saved.")

    return training_metrics["running_p"], training_metrics["running_weights"]


def run_list():
    n_trials = 10
    num_repetitions = 1
    min_d = 2
    max_d = 9
    running_p_list = []
    running_w_list = []
    for d in range(min_d, max_d + 1):
        running_p_inner_list = []
        running_w_inner_list = []
        for i in range(num_repetitions):
            running_p, running_w = run_multinomial(iter=i, n_trials=n_trials, use_BA=True, input_dim=d)
            running_p_inner_list.append(running_p)
            running_w_inner_list.append(running_w)
        running_p_list.append(running_p_inner_list)
        running_w_list.append(running_w_inner_list)

    save_metrics(running_p_list, file_path=running_p_file)
    save_metrics(running_w_list, file_path=running_w_file)


# run_list()
running_p, running_w = run_multinomial(iter=0, n_trials=n_trials, use_BA=True, input_dim=6)
