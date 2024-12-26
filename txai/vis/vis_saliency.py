import torch
import numpy as np
import matplotlib.pyplot as plt

def vis_one_saliency(X, exp, ax, fig, col_num):

    Xnp = X.detach().clone().cpu().numpy()
    enp = exp.detach().clone().cpu().numpy()
    T, d = Xnp.shape

    x_range = np.arange(T)
    
    for i in range(d):
        # Assumes heatmap:
        px, py = np.meshgrid(np.linspace(min(x_range), max(x_range), len(x_range) + 1), [min(Xnp[:,i]), max(Xnp[:,i])])
        ax[i,col_num].plot(x_range, Xnp[:,i], color = 'black')
        cmap = ax[i,col_num].pcolormesh(px, py, np.expand_dims(enp[:,i], 0), alpha = 0.5, cmap = 'Greens')
        fig.colorbar(cmap, ax = ax[i][col_num])
        
def vis_one_saliencyAsh(X, exp, ax, fig, col_num):
    """
    Visualizes the saliency map for a single feature (col_num).
    """
    Xnp = X.detach().clone().cpu().numpy()
    enp = exp.detach().clone().cpu().numpy()
    T, d = Xnp.shape

    x_range = np.arange(T)

    # Assumes heatmap:
    px, py = np.meshgrid(np.linspace(min(x_range), max(x_range), len(x_range) + 1), [min(Xnp[:,col_num]), max(Xnp[:,col_num])])
    ax.plot(x_range, Xnp[:,col_num], color = 'black')
    cmap = ax.pcolormesh(px, py, np.expand_dims(enp[:,col_num], 0), alpha = 0.5, cmap = 'Greens')
    fig.colorbar(cmap, ax=ax)

def vis_one_saliency_univariate(X, exp, ax, fig):

    Xnp = X.detach().clone().cpu().numpy()
    enp = exp.detach().clone().cpu().numpy()
    T, d = Xnp.shape

    assert d == 1, 'vis_one_saliency_univariate is only for univariate inputs'

    x_range = np.arange(T)

    print('enp', enp.shape)

    # Assumes heatmap:
    px, py = np.meshgrid(np.linspace(min(x_range), max(x_range), len(x_range) + 1), [min(Xnp[:,0]), max(Xnp[:,0])])
    ax.plot(x_range, Xnp[:,0], color = 'black')
    cmap = ax.pcolormesh(px, py, enp, alpha = 0.5, cmap = 'Greens')
    fig.colorbar(cmap, ax = ax)
