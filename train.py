import torch
import torch.nn.functional as F
from DataSets import get_DataLoader, get_SampledDataLoader
from tqdm import tqdm
from utils import get_p_from_explicit_model, calc_multinomial_BA, run_decoder_2d, get_multi_encoder


def train_multinomial(model, optimizer, loss_function, input_dim, num_epochs=100, batch_size=2 ** 8, res=None,
                      n_test=101, device="cpu"):
    if res is not None:
        n_test = model.n_trials * res + 1
        if n_test is not None:
            print(f"note: {n_test = }")

    # Initialize metrics as a dictionary
    training_metrics = {
        "input_dim": input_dim,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "n_trials": model.n_trials,
        "res": res,
        "n_test": n_test,
        "lr_list": [],
        "epoch_losses": [],
        "running_p": [],
        "running_decoder": [],
        "running_model": [],
        "running_weights": []
    }

    probabilities = [1 / input_dim] * input_dim  # torch.ones(len(data)) / len(data)  # Default: uniform distribution
    training_metrics["running_weights"].extend([probabilities] * num_epochs)

    # epoch_size = batch_size
    epoch_size = max(batch_size * 32, int(2 ** 10))
    print(f"{epoch_size = }")
    training_metrics["epoch_size"] = epoch_size
    data_loader = get_DataLoader(input_dim, batch_size, epoch_size, device, shuffle=False)

    # Wrap the epoch loop with tqdm
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        model.train()  # Set the model to training mode

        running_loss = 0.0  # To accumulate the loss over the epoch

        # with tqdm(total=len(data_loader), desc=f'Epoch {epoch}') as pbar:
        for batch in data_loader:
            # Save the old parameters (before the update)
            old_params = [param.clone() for param in model.parameters()]

            # batch = batch.to(device)
            optimizer.zero_grad()  # Clear the gradients

            recon_batch = model(batch)
            # Compute the loss
            loss = loss_function(recon_batch, batch)

            # Backward pass: Compute gradients
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Compute the step size (parameter update norm)
            step_size = 0
            for old_param, new_param in zip(old_params, model.parameters()):
                step_size += (new_param - old_param).norm().item()
            training_metrics["lr_list"].append(step_size)

        # Average loss for the epoch
        epoch_loss = running_loss / len(data_loader)
        training_metrics["epoch_losses"].append(epoch_loss)

        training_metrics["running_p"].append(get_p_from_explicit_model(model, input_dim, device, multi=True))

        # Update the progress bar and display the loss
        pbar.set_postfix(Loss=f'{epoch_loss:.4f}')
        pbar.update(1)

    mean_p = get_p_from_explicit_model(model, input_dim, device, multi=True)
    print(f"{mean_p = }")
    return training_metrics


def train_multinomial_BA(model, optimizer, input_dim, num_epochs=100, initial_batch_size=int(2 ** 12),
                         max_batch_size=int(2 ** 15), increase_points=6, growth_factor=2,
                         res=None, n_test=101, device="cpu", use_loss_weights=False, reset_weights=False):
    print(f"{device = }")
    if res is not None:
        n_test = model.n_trials * res + 1
        if n_test is not None:
            print(f"note: {n_test = }")

    # Initialize metrics as a dictionary
    training_metrics = {
        "input_dim": input_dim,
        "increase_points": increase_points,
        "growth_factor": growth_factor,
        "initial_batch_size": initial_batch_size,
        "max_batch_size": max_batch_size,
        "n_trials": model.n_trials,
        "use_loss_weights": use_loss_weights,
        "res": res,
        "n_test": n_test,
        "lr_list": [],
        "epoch_losses": [],
        "running_p": [],
        "running_decoder": [],
        "running_model": [],
        "running_weights": []
    }

    probabilities = [1 / input_dim] * input_dim

    if increase_points > num_epochs:
        increase_points = num_epochs

    current_batch_size = initial_batch_size
    epoch_size = max(current_batch_size * 32, int(2 ** 10))

    training_metrics["epoch_size"] = epoch_size
    data_loader = get_DataLoader(input_dim, current_batch_size, epoch_size, device, shuffle=False)

    # Wrap the epoch loop with tqdm
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        model.train()  # Set the model to training mode

        running_loss = 0.0  # To accumulate the loss over the epoch

        # with tqdm(total=len(data_loader), desc=f'Epoch {epoch}') as pbar:
        for batch in data_loader:
            # Save the old parameters (before the update)
            old_params = [param.clone() for param in model.parameters()]

            # batch = batch.to(device)
            optimizer.zero_grad()  # Clear the gradients

            recon_batch = model(batch)

            if use_loss_weights:
                # Compute the loss with class weights
                loss = F.cross_entropy(recon_batch, batch, weight=torch.tensor(probabilities).to(device))
            else:
                loss = F.cross_entropy(recon_batch, batch)

            # Backward pass: Compute gradients
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Compute the step size (parameter update norm)
            step_size = 0
            for old_param, new_param in zip(old_params, model.parameters()):
                step_size += (new_param - old_param).norm().item()
            training_metrics["lr_list"].append(step_size)

        # Increase batch size exponentially after certain epochs
        if (epoch + 1) % (num_epochs // increase_points) == 0 and current_batch_size < max_batch_size:
            current_batch_size = min(current_batch_size * growth_factor, max_batch_size)
            epoch_size = max(current_batch_size * 32, int(2 ** 10))
            if reset_weights:
                probabilities = [1 / input_dim] * input_dim
            data_loader = get_SampledDataLoader(input_dim, probabilities, current_batch_size, epoch_size, device)

        if (epoch + 1) % 1 == 0:
            mean_p = get_p_from_explicit_model(model, input_dim, device, multi=True)
            C, probabilities = calc_multinomial_BA(mean_p, model.n_trials)
            data_loader = get_SampledDataLoader(input_dim, probabilities, current_batch_size, epoch_size, device)

        training_metrics["running_weights"].append(probabilities)

        # Average loss for the epoch
        epoch_loss = running_loss / len(data_loader)
        training_metrics["epoch_losses"].append(epoch_loss)

        training_metrics["running_p"].append(get_p_from_explicit_model(model, input_dim, device, multi=True))

        training_metrics["running_decoder"].append(run_decoder_2d(model, device, n_test))
        training_metrics["running_model"].append(get_multi_encoder(model, input_dim, device))

        # Update the progress bar and display the loss
        pbar.set_postfix(Loss=f'{epoch_loss:.4f}')
        pbar.update(1)

    mean_p = get_p_from_explicit_model(model, input_dim, device, multi=True)
    print(f"{mean_p = }")
    return training_metrics

