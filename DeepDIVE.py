import matplotlib.pyplot as plt
from utils import load_metrics
from visualize import plot_loss_CE_decomposition, plot_capacity_with_zoom_20, calc5, plot_5, plot_capacity_4d_6

def deepdive_paper_figures():

    # Figure 2: Loss decomposition during the training process.
    dict_file = 'results/training_metrics_50_0.pkl'
    training_metrics = load_metrics(dict_file)
    plot_loss_CE_decomposition(training_metrics, block=False)

    # Figure 3: DeepDIVE’s geometric and probabilistic shaping results compared to previous methods.
    plot_capacity_with_zoom_20(20, use_load=True, use_latex=True, use_BA=True, plot_random=True, block=False, plot_new=True)

    # Figure 4: DeepDIVE’s only geometric shaping results compared to previous methods.
    plot_capacity_with_zoom_20(20, use_load=True, use_latex=True, use_BA=False, plot_random=True, block=False)

    # Figure 5: Five-symbols constellations on two-dimensional simplex;
    # corners configuration (a) and middle configuration (b).
    calc5(max_n=10, use_latex=True, plot_random=False, plot_simplex=True, block=False)

    # Figure 6: DeepDIVE’s configuration result during the training process,
    # compared to corners and middle configuration.
    plot_5(plot_simplex=False, block=False)

    # Figure 7: Six-letter composite alphabet comparison.
    plot_capacity_4d_6(use_latex=True, block=False)

    plt.show()


deepdive_paper_figures()
