from sys import float_info
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize, Bounds
from utils import (calc_entropy, calc_mean_kl, point_kl, calc_pyk, calc_binomial_information,
                   calc_equal_input_information, calc_BA_input_information, calc_multinomial_information,
                   calc_equal_input_multinomial_information, calc_multinomial_BA_input_information)


def plot_loss_CE_decomposition(training_metrics, block=False):

    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.style"] = 'italic'
    plt.rcParams['font.family'] = 'serif'

    def used_color(i, cmap=plt.cm.BuPu, amount=4, color_shift=2):
        return cmap((i + color_shift) / (amount + color_shift))

    n = training_metrics["n_trials"]
    running_weights_information = [calc_binomial_information(p, w, n) for p, w in
                                   zip(training_metrics["running_p"], training_metrics["running_weights"])]
    running_entropy = [calc_entropy(w) for w in training_metrics["running_weights"]]
    running_mean_kl = calc_mean_kl(training_metrics)
    loss = np.array(running_entropy) - np.array(running_weights_information) + np.array(running_mean_kl)

    epoch_loss = np.array(training_metrics["epoch_losses"])/np.log(2)

    amount = 5
    color_shift = 1
    cmap = plt.cm.hot
    colors = [cmap((i + color_shift) / (amount + color_shift)) for i in range(amount)]


    plt.figure(figsize=(8, 4))
    plt.yscale('log')  # Set y-axis to log scale
    plt.plot(running_entropy, label='H(S)', c=colors[0])
    plt.plot(running_weights_information, label='I(X;Y)', c=colors[1])
    plt.plot(loss, label='H(S)-I(X;Y)+E\{KL\}', c=colors[2])
    plt.plot(epoch_loss, label='Loss', c=colors[3])
    plt.plot(running_mean_kl, label='E\{KL\}', c=colors[4])

    plt.ylim((2e-3, 2.75))
    plt.xlim((0, 300))
    plt.xlabel("$Epoch$", fontsize="18")
    plt.ylabel('$Cross \hspace{2.5mm} Entropy \hspace{2.5mm} Loss$', fontsize="18")
    # plt.title('Loss Decomposition')
    plt.legend(fontsize="14",  bbox_to_anchor=(0.5, 0.475, 0.5, 0.5)) # loc='upper right')
    plt.grid(which='both')
    plt.show(block=block)


def used_color(i, cmap=plt.cm.BuPu,  amount=4, color_shift=2) :
    return cmap((i + color_shift) / (amount + color_shift))


def my_zoom_effect(ax_main, ax_zoom, lw=1.5):
    """
    Create zoom effect lines between the main plot and zoomed subplot.

    Parameters:
        ax_main (matplotlib.axes.Axes): Main plot axis.
        ax_zoom (matplotlib.axes.Axes): Zoomed subplot axis.
        lw (float): Line width of the zoom effect lines.
    """
    # Get the zoomed subplot limits
    xlim_zoom, ylim_zoom = ax_zoom.get_xlim(), ax_zoom.get_ylim()

    # Main plot data limits (rectangle region)
    xlim_main, ylim_main = ax_main.get_xlim(), ax_main.get_ylim()

    # Define corners of the zoomed region in data coordinates
    zoom_corners = [
        (xlim_zoom[0], ylim_zoom[0]),  # Bottom-left
        (xlim_zoom[0], ylim_zoom[1]),  # Top-left
        (xlim_zoom[1], ylim_zoom[0]),  # Bottom-right
        (xlim_zoom[1], ylim_zoom[1]),  # Top-right
    ]

    # Transformation pipelines
    trans_main = ax_main.transData
    trans_zoom = ax_zoom.transData
    trans_fig = ax_main.figure.transFigure.inverted()

    # Convert corners to display coordinates
    main_coords = [trans_fig.transform(trans_main.transform(corner)) for corner in zoom_corners]
    main_coords = [main_coords[1], main_coords[3]]
    zoom_coords = [trans_fig.transform(trans_zoom.transform(corner)) for corner in zoom_corners]
    zoom_coords = [zoom_coords[0], zoom_coords[2]]

    # Draw lines connecting the corners
    for (main, zoom) in zip(main_coords, zoom_coords):
        line = lines.Line2D(
            [main[0], zoom[0]], [main[1], zoom[1]],
            transform=ax_main.figure.transFigure,  # Transform applied to figure
            color="k", linestyle="--", linewidth=lw
        )
        ax_main.figure.add_artist(line)


def load_support_information(use_BA=False, file=None, take_last = 1, use_BA_information=None):

    if file is None:
        if use_BA:
            running_p_array_20 = np.load('results/real_running_p_array_20_BA.npy')
            running_p_array_30 = np.load('results/real_running_p_array_30_BA.npy')
            running_p_array = np.load('results/real_running_p_array_50_BA.npy')
            running_p_array[:20] = running_p_array_20
            running_p_array[20:30] = running_p_array_30[20:30]

        else:
            running_p_array_20 = np.load('results/running_p_array_20.npy')
            running_p_array = np.load('results/running_p_array_50.npy')
            running_p_array[:20] = running_p_array_20
    else:
        running_p_array = np.load(file)

    running_p_array = np.sort(running_p_array, axis=-1)[:,:,-take_last:,:]

    output_shape = running_p_array.shape[:-1]  # Shape of the output after reducing the last axis
    informations = np.zeros(output_shape)

    n_array = [i for i in range(1,51)]
    if use_BA_information is None:
        use_BA_information = use_BA

    # Loop over rows and compute
    for i in range(running_p_array.shape[0]):

        if use_BA_information:
            informations[i] = np.apply_along_axis(calc_BA_input_information, axis=-1, arr=running_p_array[i], n=n_array[i])
        else:
            informations[i] = np.apply_along_axis(calc_equal_input_information, axis=-1, arr=running_p_array[i], n=n_array[i])

    informations = np.max(informations, axis=(1,2))
    return informations


def print_list_shape(running_p_list):
    amount_d = len(running_p_list)
    reps = len(running_p_list[0])
    epochs = len(running_p_list[0][0])
    dimension = len(running_p_list[0][0][0])
    print(f"{amount_d = }, {reps = }, {epochs = }, {dimension = }")


def load_support_information_multinomial(use_BA=False, file=None, take_last=1, use_BA_information=None, verbose=1):

    if file is not None:
        running_p_array = np.load(file)
    else:
        file1 = 'results/running_p_multi_BA_2d_5clean_0303.npy'
        file2 = 'results/running_p_multi_BA_2k_5d_clean_0803.npy'
        file3 = 'results/running_p_multi_BA_2k_5d_16_clean_0803.npy'

        running_p_array = np.concatenate((np.load(file1), np.load(file2)[10:], np.load(file3)[15:]), axis=0)

    if verbose:
        print_list_shape(running_p_array)

    running_p_array = running_p_array[:,:,-take_last:,:,:]

    amount_n = len(running_p_array)
    informations = np.zeros(amount_n)

    if use_BA_information is None:
        use_BA_information = use_BA

    # Loop over rows and compute
    for i in range(amount_n):
        running_p = running_p_array[i][0][0]
        n = i + 1
        if use_BA_information:
            informations[i] = calc_multinomial_BA_input_information(running_p, n)
        else:
            informations[i] = calc_equal_input_multinomial_information(running_p, n)

    return informations


def inv_square_kernel(x):
    d = x - 0.5
    x = np.sign(d)*np.sqrt(np.abs(d)/2)+0.5
    return x


def fixed_support_information(support, n_vec, use_BA=False):
    if use_BA:
        return [calc_BA_input_information(support, n) for n in n_vec]
    return [calc_equal_input_information(support, n) for n in n_vec]


def plot_capacity_with_zoom_20(max_n=20, use_load=False, use_latex=False, use_BA=False, plot_random=True, block=True, plot_new=False):

    plt.rcParams['text.usetex'] = use_latex
    plt.rcParams['font.family'] = 'serif'

    support_size = 5
    cmap = plt.cm.hot

    amount = 8
    color_shift = 3

    n_vec = [a for a in range(1, max_n + 1)]

    # Create the figure and grid layout (1 row for zoomed-in plots, 1 row for main plot)
    fig = plt.figure(figsize=(9, 6))
    gs = GridSpec(2, 3, height_ratios=[1, 2], hspace=0.4)  # 3 small plots (row 0), 1 big plot (row 1)

    # Define zoom ranges for the small plots
    zoom_ranges = [(1, 4), (12, 14), (18, 20)]  # Example zoom-in ranges
    axes_zoom = []

    # Main Plot
    ax_main = fig.add_subplot(gs[1, :])  # Main plot spans the full width (bottom row)

    ax_main.plot(n_vec, [np.log2(support_size)] * max_n, '-.k', label=f'$\log({support_size})$')

    if use_load:
        load_information = load_support_information(use_BA=use_BA)[:max_n]
        if plot_new:
            load_information = load_support_information_multinomial(use_BA=use_BA)[:max_n]
        ax_main.plot(n_vec, load_information, c=used_color(0, cmap, amount=amount, color_shift=color_shift), label=f'DeepDIVE', zorder=3)


    # result for n=13
    dab_vec = np.array([0, 0.18183863, 0.5, 0.81816137, 1])
    dab_information = fixed_support_information(dab_vec, n_vec, use_BA=use_BA)
    ax_main.plot(n_vec, dab_information, linestyle='--', c=used_color(1, cmap, amount=amount, color_shift=color_shift), label=f'M-DAB$(n=13)$')

    linear_support = np.linspace(0, 1, support_size)
    squared_support = inv_square_kernel(linear_support)
    squared_information = fixed_support_information(squared_support, n_vec, use_BA=use_BA)
    ax_main.plot(n_vec, squared_information, linestyle='--', c=used_color(2, cmap, amount=amount, color_shift=color_shift), label=f'Squared')

    chernoff_vec = np.array([1e-12, 0.11426210403608106, 0.5000002414182476, 0.8857384279718752, 0.999999999999])
    chernoff_information = fixed_support_information(chernoff_vec, n_vec, use_BA=use_BA)
    ax_main.plot(n_vec, chernoff_information, linestyle='--', c=used_color(3, cmap, amount=amount, color_shift=color_shift), label=f'Chernoff')

    # Plot the main data
    linear_information = fixed_support_information(linear_support, n_vec, use_BA=use_BA)
    ax_main.plot(n_vec, linear_information, linestyle='--', c=used_color(4, cmap, amount=amount, color_shift=color_shift), label=f'Linear')

    if plot_random:
        supports_amounts = 20
        supports = np.random.uniform(0, 1, (supports_amounts, support_size))
        informations = np.apply_along_axis(fixed_support_information, axis=1, arr=supports, n_vec=n_vec, use_BA=use_BA)
        mean_info = np.mean(informations, axis=0)
        std_info = np.std(informations, axis=0)
        ax_main.plot(n_vec, mean_info, label=f'Random', linestyle='--', color=used_color(5, cmap, amount=amount, color_shift=color_shift))
        ax_main.fill_between(n_vec, mean_info - std_info, mean_info + std_info, color=used_color(5, cmap, amount=amount, color_shift=color_shift), alpha=0.3)

    ax_main.set_xlim([1, max_n])
    ax_main.set_ylabel("$C_{n,k=2,d=5}$", fontsize="18")  # "Capacity"
    ax_main.set_xlabel("$n$", fontsize="18")  # Number of Multinomial Trials
    ax_main.grid(which='both')
    ax_main.legend(fontsize="12", loc="lower right")

    delta_ranges = (0.01, 0.015, 0.002)
    # Add rectangles to mark zoomed-in regions on the main plot
    rects = []
    for zoom_range, delta in zip(zoom_ranges, delta_ranges):
        # Get the data range for mean_info in the zoom_range
        y_min = np.min(chernoff_information[zoom_range[0] - 1:zoom_range[1]]) - delta
        y_max = np.max(squared_information[zoom_range[0] - 1:zoom_range[1]]) + delta

        # Create a rectangle with adjusted height
        rect = patches.Rectangle(
            (zoom_range[0], y_min),  # Bottom-left corner
            zoom_range[1] - zoom_range[0],  # Width
            y_max - y_min,  # Height
            linewidth=1.5,
            edgecolor='k',
            facecolor='none',
            linestyle='--',
        )
        ax_main.add_patch(rect)
        rects.append(rect)

    # Plot the zoomed-in regions in a row
    for i, zoom_range in enumerate(zoom_ranges):
        ax_zoom = fig.add_subplot(gs[0, i])  # Each zoomed plot occupies one column

        if use_load:
            ax_zoom.plot(n_vec, load_information, c=used_color(0, cmap, amount=amount, color_shift=color_shift), label=f'DeepDIVE', zorder=3)

        ax_zoom.plot(n_vec, dab_information, linestyle='--', c=used_color(1, cmap, amount=amount, color_shift=color_shift), label=f'M-DAB$(n=13)$')
        ax_zoom.plot(n_vec, squared_information, linestyle='--', c=used_color(2, cmap, amount=amount, color_shift=color_shift), label=f'Squared')
        ax_zoom.plot(n_vec, chernoff_information, linestyle='--', c=used_color(3, cmap, amount=amount, color_shift=color_shift), label=f'Chernoff')
        ax_zoom.plot(n_vec, linear_information, linestyle='--', c=used_color(4, cmap, amount=amount, color_shift=color_shift), label=f'Linear')
        ax_zoom.plot(n_vec, [np.log2(support_size)] * max_n, '-.k', label=f'$\log({support_size})$')

        # Zoom limits
        ax_zoom.set_xlim(zoom_range)

        y_min = np.min(chernoff_information[zoom_range[0] - 1:zoom_range[1]]) - delta_ranges[i]
        y_max = np.max(squared_information[zoom_range[0] - 1:zoom_range[1]]) + delta_ranges[i]

        ax_zoom.set_ylim(y_min, y_max)
        ax_zoom.set_title(f"Zoom: $n \in [{zoom_range[0]}, {zoom_range[1]}]$", fontsize=10)
        ax_zoom.tick_params(axis='both', labelsize=8)
        ax_zoom.grid(which='both')
        axes_zoom.append(ax_zoom)

        # Apply zoom effect
        my_zoom_effect(ax_main, ax_zoom, lw=1.5)

    plt.tight_layout()
    plt.show(block=block)


def fixed_multinomial_support_information(support, n_vec):
    return [calc_multinomial_BA_input_information(support, n) for n in n_vec]


def solve5(n, verbose=False):
    eps = float_info.epsilon

    def cost_fun(a):
        a = float(a)
        x = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([1 / 2, 1 / 2, 0]),  np.array([a, 0, 1-a]), np.array([0, a, 1-a]), np.array([1 / 2, 1 / 2 - eps, eps])]
        return - calc_multinomial_BA_input_information(x, n)

    bounds = Bounds(0, 1)
    x0 = 0.5

    res = minimize(cost_fun, x0, bounds=bounds)
    if verbose:
        print(res.x)
    return -res.fun


def map_colors(p3dc, func, cmap='viridis', cmin=0.0, cmax=1.0):
    """
    Color a tri-mesh according to a function evaluated in each barycentre.

    p3dc: a Poly3DCollection, as returned e.g. by ax.plot_trisurf
    func: a single-valued function of 3 arrays: x, y, z
    cmap: a colormap NAME, as a string

    Returns a ScalarMappable that can be used to instantiate a colorbar.
    """

    from matplotlib.cm import ScalarMappable, get_cmap
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from numpy import array

    # reconstruct the triangles from internal data
    x, y, z, _ = p3dc._vec
    slices = p3dc._segslices
    triangles = array([array((x[s], y[s], z[s])).T for s in slices])

    # compute the barycentres for each triangle
    xb, yb, zb = triangles.mean(axis=1).T

    # compute the function in the barycentres
    values = func(xb, yb, zb)

    # usual stuff
    norm = Normalize()
    # colors = get_cmap(cmap)(norm(values))

    # Get the colormap and extract a sub-range
    original_cmap = get_cmap(cmap)
    new_cmap = LinearSegmentedColormap.from_list(
        f'{cmap}_sub',
        original_cmap(np.linspace(cmin, cmax, 256))
    )

    # Apply the colormap
    colors = new_cmap(norm(values))

    # set the face colors of the Poly3DCollection
    p3dc.set_fc(colors)

    # if the caller wants a colorbar, they need this
    return ScalarMappable(cmap=cmap, norm=norm)


def test_plot_simplex_inter_result(my_x = [np.array([0,0,1]),np.array([0,1/2,1/2]),np.array([1/3,1/3,1/3])], n=20,
                                   usetex=False, block=True):
    p_y_k, I, r = calc_pyk(my_x, n)

    print("I: ", I)
    print("r: ", r)

    def f_3d_vec(x, y, z):
        return [point_kl(n, [x[i], y[i], z[i]], p_y_k) for i in range(len(x))]

    n_x = 40
    n_y = 40
    xd = np.linspace(0, 1, n_x)
    yd = np.linspace(0, 1, n_y)
    x, y = np.meshgrid(xd, yd)

    x = np.ravel(x)
    y = np.ravel(y)
    xy = list(zip(x, y))
    triangle = list(filter(lambda a: a[0] + a[1] <= 1, xy))
    t_len = len(triangle)
    t_x, t_y = zip(*triangle)
    t_z = [1- t_x[i]-t_y[i] for i in range(t_len)]

    fig = plt.figure()
    plt.rcParams['text.usetex'] = usetex
    plt.rcParams["font.style"] = 'italic'
    plt.rcParams['font.family'] = 'serif'
    ax = plt.axes(projection='3d', computed_zorder=False)
    ax.view_init(azim=45, elev=20)  # 50

    p3dc = ax.plot_trisurf(t_x, t_y, t_z, alpha=0.7, zorder=-2)
    mappable = map_colors(p3dc, f_3d_vec, 'YlOrRd', cmin=0, cmax=1)

    alpha = 50

    # x_0 = x_1
    zline = np.linspace(0, 1, alpha)
    yline = 0.5 - 0.5*zline
    xline = 0.5 - 0.5*zline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_1 = x_2
    xline = np.linspace(0, 1, alpha)
    yline = 0.5 - 0.5*xline
    zline = 0.5 - 0.5*xline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_0 = x_2
    yline = np.linspace(0, 1,  alpha)
    xline = 0.5 - 0.5*yline
    zline = 0.5 - 0.5*yline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_2 = 0
    yline = np.linspace(0, 1,  alpha)
    xline = 1 - yline
    zline = 0*yline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_1 = 0
    xline = np.linspace(0, 1,  alpha)
    zline = 1 - xline
    yline = 0*xline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_0 = 0
    yline = np.linspace(0, 1,  alpha)
    zline = 1 - yline
    xline = 0*yline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    symbols = np.transpose(my_x)
    ax.scatter(symbols[0], symbols[1], symbols[2], c='maroon', s=70*4*r, marker='o', edgecolor='k', alpha=1, label='Input distribution mass point', zorder=3)  # , labal='before')

    ax.xaxis._axinfo['grid'].update(color='gray')  # X-axis grid
    ax.yaxis._axinfo['grid'].update(color='gray')  # Y-axis grid
    ax.zaxis._axinfo['grid'].update(color='gray')  # Z-axis grid

    ax.set_xlabel("$x_{1}$", fontsize="18")
    ax.set_ylabel("$x_{2}$", fontsize="18")
    ax.set_zlabel("$x_{3}$", fontsize="18")

    plt.show(block=block)


def test_plot_simplex_inter_result_ax(my_x = [np.array([0,0,1]),np.array([0,1/2,1/2]),np.array([1/3,1/3,1/3])], n=20, ax=None, usetex=False, block=True):

    p_y_k, I, r = calc_pyk(my_x, n)
    print("I: ", I)
    print("r: ", r)

    def f_3d_vec(x, y, z):
        return [point_kl(n, [x[i], y[i], z[i]], p_y_k) for i in range(len(x))]

    n_x = 40
    n_y = 40
    xd = np.linspace(0, 1, n_x)
    yd = np.linspace(0, 1, n_y)
    x, y = np.meshgrid(xd, yd)

    x = np.ravel(x)
    y = np.ravel(y)
    xy = list(zip(x, y))
    triangle = list(filter(lambda a: a[0] + a[1] <= 1, xy))
    t_len = len(triangle)
    t_x, t_y = zip(*triangle)
    t_z = [1- t_x[i]-t_y[i] for i in range(t_len)]

    if ax is None:
        fig = plt.figure()
        plt.rcParams['text.usetex'] = usetex
        plt.rcParams["font.style"] = 'italic'
        plt.rcParams['font.family'] = 'serif'
        ax = plt.axes(projection='3d', computed_zorder=False)
    # ax = Axes3D(fig, computed_zorder=False)
    ax.view_init(azim=45, elev=20)  # 50

    p3dc = ax.plot_trisurf(t_x, t_y, t_z, alpha=0.7, zorder=-2)
    mappable = map_colors(p3dc, f_3d_vec, 'YlOrRd', cmin=0, cmax=1)

    alpha = 50

    # x_0 = x_1
    zline = np.linspace(0, 1, alpha)
    yline = 0.5 - 0.5*zline
    xline = 0.5 - 0.5*zline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_1 = x_2
    xline = np.linspace(0, 1, alpha)
    yline = 0.5 - 0.5*xline
    zline = 0.5 - 0.5*xline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_0 = x_2
    yline = np.linspace(0, 1,  alpha)
    xline = 0.5 - 0.5*yline
    zline = 0.5 - 0.5*yline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_2 = 0
    yline = np.linspace(0, 1,  alpha)
    xline = 1 - yline
    zline = 0*yline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_1 = 0
    xline = np.linspace(0, 1,  alpha)
    zline = 1 - xline
    yline = 0*xline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    # x_0 = 0
    yline = np.linspace(0, 1,  alpha)
    zline = 1 - yline
    xline = 0*yline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1, zorder=2)

    symbols = np.transpose(my_x)
    ax.scatter(symbols[0], symbols[1], symbols[2], c='maroon', s=70*4*r, marker='o', edgecolor='k', alpha=1, label='Input distribution mass point', zorder=3)  # , labal='before')

    ax.xaxis._axinfo['grid'].update(color='gray')  # X-axis grid
    ax.yaxis._axinfo['grid'].update(color='gray')  # Y-axis grid
    ax.zaxis._axinfo['grid'].update(color='gray')  # Z-axis grid

    ax.set_xlabel("$x_{1}$", fontsize="18")
    ax.set_ylabel("$x_{2}$", fontsize="18")
    ax.set_zlabel("$x_{3}$", fontsize="18")

    fig.colorbar(mappable, ax=ax, shrink=0.7)


def calc5(max_n=20, use_latex=False, block=True, plot_random=False, plot_simplex=False):
    eps = float_info.epsilon
    support_size = 5
    n_vec = [a for a in range(1, max_n + 1)]

    if plot_simplex:
        x = [np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([0, 1 / 2, 1 / 2]), np.array([1 / 2, 0, 1 / 2])]
        test_plot_simplex_inter_result(x, n=10, block=False, usetex=use_latex)
        a = 0.4
        x = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([1 / 2, 1 / 2, 0]), np.array([a, 0, 1 - a]), np.array([0, a, 1 - a])]
        test_plot_simplex_inter_result(x, n=10, block=False, usetex=use_latex)

    # Note I used also "np.array([1 / 2, 1 / 2 - eps, eps])" as Blahut-Arimoto not supported P(y_i) = 0
    corners_support = [np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([0, 1 / 2, 1 / 2]), np.array([1 / 2, 1 / 2, 0]), np.array([1 / 2, 1 / 2 - eps, eps])]
    corners_information = fixed_multinomial_support_information(corners_support, n_vec)

    middle_information = [solve5(n) for n in n_vec]

    plt.rcParams['text.usetex'] = use_latex
    plt.rcParams['font.family'] = 'serif'

    cmap = plt.cm.hot

    fig, ax_main = plt.subplots(figsize=(8, 6))
    ax_main.plot(n_vec, corners_information, c=used_color(0, cmap), label="Corners support")
    ax_main.plot(n_vec, middle_information, c=used_color(2, cmap), label="Middle support")

    if plot_random:
        supports_amounts = 20

        supports = []
        for i in range(supports_amounts):
            support = []
            for j in range(support_size):
                probs = np.random.uniform(0, 1, 2)
                probs = np.append(probs, [0, 1])
                probs.sort()
                probs = probs[1:] - probs[:-1]
                support.append(probs)
            supports.append(support)

        informations = []
        for i in range(supports_amounts):
            informations.append(fixed_multinomial_support_information(supports[i], n_vec))

        informations = np.array(informations)
        mean_info = np.mean(informations, axis=0)
        std_info = np.std(informations, axis=0)
        ax_main.plot(n_vec, mean_info, label=f'Random', color=used_color(3, cmap))
        ax_main.fill_between(n_vec, mean_info - std_info, mean_info + std_info, color=used_color(3, cmap), alpha=0.3)

    ax_main.plot(n_vec, [np.log2(support_size)]*max_n, '--k', label=f'$\log({support_size})$')
    ax_main.set_xlim([1, max_n])
    ax_main.set_ylabel("$C_{n,k=3,d=5}$", fontsize="18")  # "Capacity"
    ax_main.set_xlabel("$n$", fontsize="18")  # Number of Multinomial Trials
    ax_main.grid(which='both')
    ax_main.legend(fontsize="16", loc="lower right" )
    plt.show(block=block)


def plot_5(block=True, usetex=True, plot_simplex=True):

    eps = float_info.epsilon

    # Create the figure
    fig = plt.figure(figsize=(8, 4))
    plt.rcParams['text.usetex'] = usetex
    plt.rcParams["font.style"] = 'italic'
    plt.rcParams['font.family'] = 'serif'

    if plot_simplex:
        # Define the subplots
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')  # Top-left 3D plot
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')  # Top-right 3D plot
        ax3 = fig.add_subplot(2, 1, 2)  # Bottom 2D plot

        x = [np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([0, 1 / 2, 1 / 2]),
             np.array([1 / 2, 0, 1 / 2])]
        test_plot_simplex_inter_result_ax(x, n=10, block=False, usetex=usetex, ax=ax1)
        a = 0.4
        x = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([1 / 2, 1 / 2, 0]), np.array([a, 0, 1 - a]),
             np.array([0, a, 1 - a])]
        test_plot_simplex_inter_result_ax(x, n=10, block=False, usetex=usetex, ax=ax2)
    else:
        ax3 = plt.axes()

    running_p_array = np.load('results/running_p_multi_BA_3d.npy')
    running_w_array = np.load('results/running_w_multi_BA_3d.npy')

    plot = {"n": (10, 10), "take": 300}
    i = 9
    j = 0
    plot_information = []
    for r in range(1, plot["take"] + 1):
        x = running_p_array[i, j, -r, :]
        w = running_w_array[i, j, -r, :]
        n = i + 1
        plot_information += [calc_multinomial_information(x, w, n)]

    iter_amount = plot["take"]
    iter_vec = np.arange(1, iter_amount + 1)
    ax3.plot(iter_vec, iter_amount * [solve5(n)], 'k--', label='Middle capacity')

    amount = 5
    color_shift = 2
    cmap = plt.cm.hot

    ax3.plot(iter_vec, plot_information[::-1], label="DeepDIVE", c=used_color(0, cmap, amount=amount, color_shift=color_shift))

    corners_support = [np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0]),
                       np.array([0, 1 / 2, 1 / 2]), np.array([1 / 2, 1 / 2, 0]),
                       np.array([1 / 2, 1 / 2 - eps, eps])]
    corners_information = fixed_multinomial_support_information(corners_support, [n])
    ax3.plot(iter_vec, iter_amount * corners_information, 'k', linestyle='dashdot',
             label='Corners capacity')
    ax3.set_ylabel("$C_{n=10,k=3,d=5}$", fontsize="18")  # "Capacity"
    ax3.set_ylim((2.3, 2.32))
    ax3.set_ylim((2.3025, 2.3175))
    ax3.set_xlim((0, 300))

    ax3.set_xlabel("$Epoch$", fontsize="18")
    ax3.grid(which='both')
    ax3.legend(fontsize="16", loc="lower right")

    plt.show(block=block)


def solve(n, k, d, verbose=False):
    # Define the cost function
    def cost_fun(flat_x):
        # Reshape flat_x into a list of d arrays of size k
        x = [flat_x[i * k:(i + 1) * k] for i in range(d)]
        return -calc_multinomial_BA_input_information(x, n)

    # Define the bounds: x_i >= 0 for all i and x_i <= 1
    bounds = [(0, 1) for _ in range(d * k)]

    # Define the constraints: each group of k elements sums to 1
    linear_constraints = [
        {
            'type': 'eq',
            'fun': lambda x, i=i: np.sum(x[i * k:(i + 1) * k]) - 1
        }
        for i in range(d)
    ]

    # Generate different random probability vectors for initialization
    def random_probability_vector(size):
        vec = np.random.rand(size)
        return vec / np.sum(vec)

    x0 = np.concatenate([random_probability_vector(k) for _ in range(d)])

    # Minimize the cost function
    res = minimize(cost_fun, x0, constraints=linear_constraints, bounds=bounds)
    if verbose:
        formatted_data = np.array2string(res.x, formatter={'float_kind': lambda x: f"{x:.2f}"})
        print(formatted_data)
    return -res.fun



def load_p_and_w_4d_6(verbose=False, p_file=None, w_file=None):

    if (p_file is not None) and (w_file is not None):
        running_p_array = np.load(p_file)
        running_w_array = np.load(w_file)
    else:
        running_p_array1 = np.load('results/running_p_multi_BA_4d_5_reps.npy')
        running_w_array1 = np.load('results/running_w_multi_BA_4d_5_reps.npy')
        running_p_array2 = np.load('results/running_p_multi_BA_4d_5_reps1.npy')
        running_w_array2 = np.load('results/running_w_multi_BA_4d_5_reps1.npy')
        running_p_array = np.concatenate((running_p_array1, running_p_array2), axis=1)
        running_w_array = np.concatenate((running_w_array1, running_w_array2), axis=1)

    take = 10
    output_shape = running_w_array.shape[:-2]  # Shape of the output after reducing the last axis
    informations = np.zeros((output_shape[0], output_shape[1], take))
    for i in range(output_shape[0]):  # n - trials
        for j in range(output_shape[1]):  # repeats
            for r in range(take):
                x = running_p_array[i, j, -(r+1), :, :]
                w = running_w_array[i, j, -(r+1), :]
                n = i+1
                informations[i, j, r] = calc_multinomial_information(x, w, n)

    if verbose:
        for i in range(output_shape[0]):  # n - trials
            print(f"{i = }")
            arr = informations[i]
            # Get the indices of the maximum value
            index = np.unravel_index(np.argmax(arr), arr.shape)

            print("running_p:")
            formatted_data = np.array2string(running_p_array[i][index[0]][-(index[1]+1)], formatter={'float_kind': lambda x: f"{x:.3f}"})
            print(formatted_data)
            print("running_w:")
            formatted_data = np.array2string(running_w_array[i][index[0]][-(index[1]+1)], formatter={'float_kind': lambda x: f"{x:.3f}"})
            print(formatted_data)

    informations = np.max(informations, axis=(1,2))  # take best from repeats
    return informations


def plot_capacity_4d_6(max_n=10, use_latex=False, use_solver=False, block=True, plot_new=False):

    eps = float_info.epsilon

    plt.rcParams['text.usetex'] = use_latex
    plt.rcParams['font.family'] = 'serif'

    support_size = 6
    n_vec = [a for a in range(1, max_n + 1)]

    amount = 5
    color_shift = 2
    cmap = plt.cm.hot
    fig, ax_main = plt.subplots(figsize=(8, 3))

    ax_main.plot(n_vec, [np.log2(support_size)] * max_n, '-.k', label=f'$\log({support_size})$')

    information = load_p_and_w_4d_6(verbose=True)
    ax_main.plot(n_vec, information, c=used_color(0, cmap, amount=amount, color_shift=color_shift), label=f'DeepDIVE')

    if plot_new:
        p_file = 'results/running_p_multi_BA_4d_6clean_0303.npy'
        w_file = 'results/running_w_multi_BA_4d_6clean_0303.npy'
        load_new = load_p_and_w_4d_6(verbose=True, p_file=p_file, w_file=w_file)
        ax_main.plot(n_vec, load_new, c=used_color(0.5, cmap, amount=amount, color_shift=color_shift), linestyle='--', label=f'new')


    uniform_support = [np.array([1, 0, 0, 0]), np.array([0, 0, 0, 1]), np.array([0, 0, 1, 0]), np.array([0, 1, 0, 0]),
                       np.array([0, 0, 1 / 2, 1 / 2]),  np.array([1 / 2, 1 / 2, 0, 0]),  np.array([1 / 2, 1 / 2-2*eps, +eps, +eps])]

    if use_solver:
        solver_information = [solve(n=n, k=4, d=6) for n in n_vec]
        ax_main.plot(n_vec, solver_information, c=used_color(1, cmap, amount=amount, color_shift=color_shift), label=f'Solver')

    print(f"{len(uniform_support) = }")
    ax_main.plot(n_vec, fixed_multinomial_support_information(uniform_support, n_vec), c=used_color(2, cmap, amount=amount, color_shift=color_shift),
                 label="Uniform composite")

    ax_main.set_xlim([1, max_n])
    ax_main.set_ylabel("$C_{n,k=4,d=6}$", fontsize="18")  # "Capacity"
    ax_main.set_xlabel("$n$", fontsize="18")  # Number of Multinomial Trials
    ax_main.grid(which='both')
    ax_main.legend(fontsize="16", loc="lower right")  # bbox_to_anchor=(0.3, 0., 0.5, 0.5))
    plt.show(block=block)
