import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import probs_to_logits

class ModifiedEncoder(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        self.layers = original_model.encoder[:-1]  # Exclude Sigmoid layer
        self.final_layer = original_model.encoder[-1]  # Final Linear layer

    def forward(self, x, return_pre_activation=False):
        x = self.layers(x)  # Pass through all layers except the last
        if return_pre_activation:
            return x
        return self.final_layer(x)   # Apply last layer manually


class Multinomial_VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_amount, latent_dim, n_trials, tau=0.01, device='cpu'):
        super(Multinomial_VAE, self).__init__()
        self.n_trials = n_trials  # Number of trials for the Binomial distribution
        self.tau = tau
        self.latent_dim = latent_dim
        if latent_dim == 1:
            self.encoder = self._build_layers(input_dim, hidden_dim, hidden_amount, latent_dim, final_activation=nn.Sigmoid)
        elif latent_dim > 1:
            self.encoder = self._build_layers(input_dim, hidden_dim, hidden_amount, latent_dim, final_activation=nn.Softmax)
        self.decoder = self._build_layers(latent_dim, hidden_dim, hidden_amount, input_dim)

        self.numerical_input = False

        # Generate all possible count vectors for multinomial outcomes
        # This step assumes a small number of trials and categories for feasibility
        from itertools import product
        self.count_vectors = torch.tensor(
            [list(count) for count in product(range(self.n_trials + 1), repeat=latent_dim)
             if sum(count) == self.n_trials],
            dtype=torch.float,
            device=device,
        )  # Shape: [num_combinations, num_categories]

        # Compute multinomial coefficients for each count vector
        self.multinomial_coeffs = torch.exp(
            torch.lgamma(torch.tensor(self.n_trials + 1, device=device)) -
            torch.lgamma(self.count_vectors + 1).sum(dim=-1)
        ).to(device)  # Shape: [num_combinations]


    def _build_layers(self, start_size: int, hidden_size: int, hidden_amount: int, end_size: int,
                      final_activation=None, activation=nn.ReLU) -> nn.Sequential:
        layers = [nn.Linear(start_size, hidden_size)]

        # Add activation if provided, otherwise skip
        if activation is not None:
            layers.append(activation())

        for _ in range(hidden_amount - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if activation is not None:
                layers.append(activation())
        layers.append(nn.Linear(hidden_size, end_size))
        if final_activation:
            layers.append(final_activation())
        return nn.Sequential(*layers)

    def get_n(self):
        return self.n_trials

    def update_n(self, new_n: int):
        self.n_trials = new_n

    def get_tau(self):
        return self.tau

    def update_tau(self, new_tau: int):
        self.tau = new_tau

    def multinomial(self, probs):
        # Input: probs of shape [batch_size, latent_dim]
        # Ensure probs sums to 1 along the categories dimension
        assert torch.allclose(probs.sum(dim=-1), torch.tensor(1.0, device=probs.device))

        # Compute probabilities for each count vector
        probs_expanded = probs.unsqueeze(-2)  # Shape: [batch_size, 1, latent_dim]
        count_vectors_expanded = self.count_vectors.unsqueeze(0).unsqueeze(
            0)

        probabilities = self.multinomial_coeffs * torch.prod(
            torch.pow(probs_expanded.to(probs.device), count_vectors_expanded.to(probs.device)), dim=-1
        )  # Shape: [1, batch_size, num_combinations]

        return probabilities

    def encode(self, x):
        # Step 1: Compute category probabilities from the encoder
        probs = self.encoder(x)  # Shape: [batch_size, latent_dim]

        # Step 2: Compute multinomial PMF probabilities
        multinomial_pmf = self.multinomial(probs)  # Shape: [1, batch_size, num_combinations]

        # Step 3: Convert probabilities to logits
        logits = probs_to_logits(multinomial_pmf)  # Shape: [1, batch_size, num_combinations]

        # Step 4: Reshape logits for downstream processing
        logits = logits.view(-1, multinomial_pmf.size(-1))  # Reshape to [batch_size, num_combinations]
        return logits

    def reparameterize(self, logits):
        # Apply Gumbel-Softmax trick
        rand_grid = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(rand_grid))
        y = F.softmax((logits + gumbel_noise) / self.tau, dim=-1)
        return y  # This is the continuous relaxation of the binomial latent variable

    def forward(self, x):
        logits = self.encode(x)
        z = self.reparameterize(logits)
        mean_z = torch.matmul(z, self.count_vectors)/self.n_trials
        return self.decoder(mean_z)
