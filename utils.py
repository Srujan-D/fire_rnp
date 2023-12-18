import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def generate_grid(h, w):
    rows = torch.linspace(0, 1, h)
    cols = torch.linspace(0, 1, w)
    x, y = torch.meshgrid(rows, cols)
    grid = torch.stack([x.flatten(), y.flatten()]).t()
    # grid = torch.stack([cols.repeat(h, 1).t().contiguous().view(-1), rows.repeat(w)], dim=1)
    grid = grid.unsqueeze(0)
    return grid


def load_nc_data(data_file, variable="air"):
    from netCDF4 import Dataset as dt

    f = dt(data_file)
    if variable == "air":
        air = f.variables["air"]
        air_range = air.valid_range
        data = air[:].data
        # convert to degree celsius
        if air.units == "degK":
            data -= 273
            air_range -= 273
    elif variable == "precip":
        precip = f.variables["precip"]
        data = precip[:].data
    else:
        data_name = f.variables[variable]
        data_range = data_name.valid_range
        data = data_name[:].data
    return data


def fire_save_image(inps, y_target, mu, fn, var=None, data_mean=0, sz1=0, sz2=0):
    n = len(inps)
    # TODO: find a way to determine figsize
    # compute this from the size of image
    fig, ax = plt.subplots(2, n + 1, figsize=(16, 8), sharex=True, sharey=True)
    ax = ax.flatten()
    for ax_ in ax:
        ax_.get_xaxis().set_visible(False)
        ax_.get_yaxis().set_visible(False)

    true = y_target[-1].cpu().numpy().squeeze().reshape(sz1, sz2) + data_mean
    vmin = true.min()
    vmax = true.max()
    cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])

    sns.heatmap(
        inps[0],
        ax=ax[0],
        cmap="ocean",
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        cbar_ax=cbar_ax,
        square=False,
    )
    ax[0].set_title("T=1")
    for i in range(1, n):
        sns.heatmap(
            inps[i],
            ax=ax[i],
            cmap="ocean",
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            square=False,
        )
        ax[i].set_title("T={}".format(i + 1))

        sns.heatmap(
            y_target[i].cpu().numpy().squeeze().reshape(sz1, sz2) + data_mean,
            ax=ax[i + n],
            cmap="ocean",
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            square=False,
        )
        ax[i].set_title("Actual (T={})".format(i + 1))

    sns.heatmap(
        true, ax=ax[n], cmap="ocean", vmin=vmin, vmax=vmax, cbar=False, square=False
    )
    ax[n].set_title("Actual (T={})".format(n + 1))

    sns.heatmap(
        mu, ax=ax[2*n + 1], cmap="ocean", vmin=vmin, vmax=vmax, cbar=False, square=False
    )
    ax[2*n - 1].set_title("Prediction (T={})".format(n + 1))

    fig.subplots_adjust(wspace=0.05, hspace=0.1)

    plt.savefig(fn)
    plt.close(fig)


def save_image(inps, true, mu, fn, var=None):
    n = len(inps)
    # TODO: find a way to determine figsize
    # compute this from the size of image
    fig, ax = plt.subplots(2, n, figsize=(16, 8), sharex=True, sharey=True)
    ax = ax.flatten()
    for ax_ in ax:
        ax_.get_xaxis().set_visible(False)
        ax_.get_yaxis().set_visible(False)

    n = len(inps)
    vmin = true.min()
    vmax = true.max()
    cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])

    sns.heatmap(
        inps[0],
        ax=ax[0],
        cmap="ocean",
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        cbar_ax=cbar_ax,
        square=False,
    )
    ax[0].set_title("T=1")
    for i in range(1, n):
        sns.heatmap(
            inps[i],
            ax=ax[i],
            cmap="ocean",
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            square=False,
        )
        ax[i].set_title("T={}".format(i + 1))

    sns.heatmap(
        true, ax=ax[n], cmap="ocean", vmin=vmin, vmax=vmax, cbar=False, square=False
    )
    ax[n].set_title("Actual (T={})".format(n + 1))

    # sns.heatmap(
    #     mu, ax=ax[n + 1], cmap="ocean", vmin=vmin, vmax=vmax, cbar=False, square=False
    # )
    sns.heatmap(
        mu, ax=ax[n + 1], cmap="ocean", vmin=mu.min(), vmax=mu.max(), cbar=True, square=False
    )
    ax[n + 1].set_title("Prediction (T={})".format(n + 1))

    fig.subplots_adjust(wspace=0.05, hspace=0.1)

    plt.savefig(fn)
    plt.close(fig)

def save_multi_futures(inps, true, mu, fn, var=None):
    n = len(inps)
    f = len(mu)

    # print(len(inps), len(mu), len(true))
    # TODO: find a way to determine figsize
    # compute this from the size of image
    fig, ax = plt.subplots(2, n+f, figsize=(16, 8), sharex=True, sharey=True)
    ax = ax.flatten()
    for ax_ in ax:
        ax_.get_xaxis().set_visible(False)
        ax_.get_yaxis().set_visible(False)

    n = len(inps)
    vmin = true.min()
    vmax = true.max()
    cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])

    sns.heatmap(
        inps[0],
        ax=ax[0],
        cmap="ocean",
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        cbar_ax=cbar_ax,
        square=False,
    )
    ax[0].set_title("T=1")
    for i in range(1, n):
        sns.heatmap(
            inps[i],
            ax=ax[i],
            cmap="ocean",
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            square=False,
        )
        ax[i].set_title("T={}".format(i + 1))

    for i in range(f):  
        sns.heatmap(
            true[i], ax=ax[n+i], cmap="ocean", vmin=vmin, vmax=vmax, cbar=False, square=False
        )
        ax[n+i].set_title("Actual (T={})".format(n + i+ 1))
 
        # sns.heatmap(
        #     mu, ax=ax[n + 1], cmap="ocean", vmin=vmin, vmax=vmax, cbar=False, square=False
        # )
        sns.heatmap(
            mu[i], ax=ax[2*n + i+ 1], cmap="ocean", vmin=mu.min(), vmax=mu.max(), cbar=True, square=False
        )
        ax[2*n + i+ 1].set_title("Prediction (T={})".format(n +i + 1))

    fig.subplots_adjust(wspace=0.05, hspace=0.1)

    plt.savefig(fn)
    plt.close(fig)
