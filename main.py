import torch

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_multinomial(iter=0, n_trials=12, plot=False, use_BA=True, input_dim=5):
    model = Multinomial_VAE(input_dim, hidden_dim, hidden_amount, latent_dim, n_trials, tau=tau, device=device).to(
        device)
    loss_function = F.cross_entropy
    optimizer = optim.Adam(model.parameters())  # , lr=5e-4, betas=(0.5, 0.99))

    if use_BA:
        training_metrics = train_multinomial_BA(model, optimizer, input_dim, num_epochs, batch_size, res,
                                                n_test, device)
    else:
        training_metrics = train_multinomial(model, optimizer, loss_function, input_dim, num_epochs, batch_size,
                                             res, n_test, device)

    torch.save(model.state_dict(), 'running_d/' + model_file + str(input_dim) + '_' + str(iter) + '.pth')
    save_metrics(training_metrics, 'running_d/' + dict_file + str(input_dim) + '_' + str(iter) + '.pkl')
    # Save to a file
    print("Model saved.")

    if plot:
        plot_loss_CE_decomposition(training_metrics)
        plot_running_decoder(training_metrics)
        plot_running_decoder(training_metrics, use_res=False)
        plot_lr(training_metrics["lr_list"])
        plot_loss(training_metrics["epoch_losses"])
        plot_running_p(training_metrics["running_p"])

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
            # running_p, running_w = run_noised(pyx,  n_trials=n, plot=True, use_BA=True)
            running_p, running_w = run_multinomial(iter=i, n_trials=n_trials, plot=False, use_BA=True, input_dim=d)
            running_p_inner_list.append(running_p)
            running_w_inner_list.append(running_w)
        running_p_list.append(running_p_inner_list)
        running_w_list.append(running_w_inner_list)

    save_metrics(running_p_list, file_path="running_d/running_p_multi_BA_new_next6.pkl")
    save_metrics(running_w_list, file_path="running_d/running_w_multi_BA_new_next6.pkl")

run_list()