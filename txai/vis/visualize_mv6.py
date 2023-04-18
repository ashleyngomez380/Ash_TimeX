import torch
import numpy as np
import matplotlib.pyplot as plt
from txai.vis.visualize_cbm1 import get_x_mask_borders

def plot_heatmap(mask_logits, smooth_src, ax, fig):
        x_range = torch.arange(smooth_src.shape[0])
        px, py = np.meshgrid(np.linspace(min(x_range), max(x_range), len(x_range) + 1), [min(smooth_src), max(smooth_src)])
        #ax[0,i].imshow(mask_logits[i,...].T, alpha = 0.5, cmap = 'Greens')
        cmap = ax.pcolormesh(px, py, mask_logits.T, alpha = 0.5, cmap = 'Greens')
        fig.colorbar(cmap, ax = ax)

def vis_concepts(model, test_tup, show = True):

    src, times, y = test_tup

    found_masks, found_smooth_src = model.find_closest_for_prototypes(src, times, captum_input = False)

    found_masks_np = [found_masks[i].cpu().numpy() for i in range(len(found_masks))]
    found_smooth_src_np = [found_smooth_src[i].cpu().numpy() for i in range(len(found_smooth_src))]

    fig, ax = plt.subplots(1, model.n_concepts)

    for i in range(model.n_concepts):
        ax[i].plot(found_smooth_src_np[i][...,0])
        plot_heatmap(np.expand_dims(found_masks_np[i], axis = 1), found_smooth_src_np[i][...,0], ax[i], fig)

    if show:
        plt.show()

def logical_or_mask_along_explanations(total_mask):
    tmask = (total_mask.sum(dim=-1) > 0).float() # Effectively ORs along last dimension
    return tmask

def visualize_explanations(model, test_tup, n = 3, class_num = None, show = True, heatmap = True, topk = None):
    '''
    TODO: Rewrite

    - Shows each extracted explanations along with importance scores for n samples
    - TODO in future: aggregate multiple explanation types into one visualization

    NOTE: Only works for regular time series
    '''
    # Quick function to visualize some samples in test_tup
    # FOR NOW, assume only 2 masks, 2 concepts

    X, times, y = test_tup

    choices = np.arange(X.shape[1])
    if class_num is not None:
        choices = choices[(y == class_num).cpu().numpy()]
    inds = torch.from_numpy(np.random.choice(choices, size = (n,), replace = False)).long()
    num_on_x = (1 + model.n_explanations)
    fig, ax = plt.subplots(num_on_x, n, sharex = True)

    sampX, samp_times, samp_y = X[:,inds,:], times[:,inds], y[inds]
    x_range = torch.arange(sampX.shape[0])

    model.eval()
    with torch.no_grad():
        out = model(sampX, samp_times, captum_input = False)
        #pred, pred_mask, mask_in, ste_mask, smoother_stats, smooth_src = model(sampX, samp_times, captum_input = False)

    pred, pred_mask = out['pred'], out['pred_mask']
    masks = (out['ste_mask']).float() # All masks
    print('masks', masks.shape)
    mask_logits = out['mask_logits'].detach().cpu().numpy()
    print('mask_logits', mask_logits.shape)
    aggregate_mask_discrete = logical_or_mask_along_explanations(masks)
    aggregate_mask_continuous = mask_logits.sum(-1)

    smooth_src = torch.stack(out['smooth_src'], dim = 0) # Shape (N_c, T, B, d)

    pred = pred.softmax(dim=1).argmax(dim=1)
    pred_mask = pred_mask.softmax(dim=1).argmax(dim=1)
    print('pred', pred.shape)

    title_format1 = 'y={:1d}, yhat={:1d}'

    for i in range(n): # Iterate over samples

        # fit lots of info into the title
        yi = samp_y[i].item()
        pi = pred[i].item()
        ax[0,i].set_title(title_format1.format(yi, pi))

        # Top plot shows full sample with full mask:
        sX = sampX[:,i,:].cpu().numpy()
        # Stays fixed (for grids on samples):
        px, py = np.meshgrid(np.linspace(min(x_range), max(x_range), len(x_range) + 1), [min(sX), max(sX)])
        ax[0,i].plot(x_range, sX, color = 'black')


        if heatmap: # Plot discrete mask
            if topk is not None:
                tk_inds = np.flip(np.argsort(aggregate_mask_continuous[i,...]))[:topk]
                tk_mask = np.zeros_like(aggregate_mask_continuous[i,...])
                tk_mask[tk_inds] = 1
                block_inds = get_x_mask_borders(mask = torch.from_numpy(tk_mask))
                for k in range(len(block_inds)):
                    ax[0,i].axvspan(*block_inds[k], facecolor = 'green', alpha = 0.4)
            else:
                cmap = ax[0,i].pcolormesh(px, py, np.expand_dims(aggregate_mask_continuous[i,...], -1).T, alpha = 0.5, cmap = 'Greens')
                fig.colorbar(cmap, ax = ax[0,i])
        else:
            block_inds = get_x_mask_borders(mask = aggregate_mask_discrete[i,...]) # GET WHOLE MASK
            for k in range(len(block_inds)):
                ax[0,i].axvspan(*block_inds[k], facecolor = 'green', alpha = 0.4)

        for j in range(num_on_x-1):

            # Add subtitles:
            #ax[(j+1)][i].set_title('a={}, p={}'.format())
            ax[(j+1)][i].plot(x_range, smooth_src[j,:,i,:].cpu().numpy(), color = 'black')

            if heatmap:
                if topk is not None:
                    # Mask in those that are in the top-k
                    tk_inds = np.flip(np.argsort(mask_logits[i,:,j]))[:topk]
                    tk_mask = np.zeros_like(mask_logits[i,:,j])
                    tk_mask[tk_inds] = 1
                    block_inds = get_x_mask_borders(mask = torch.from_numpy(tk_mask))
                    for k in range(len(block_inds)):
                        ax[j+1][i].axvspan(*block_inds[k], facecolor = 'green', alpha = 0.4)
                else:
                    cmap = ax[j+1][i].pcolormesh(px, py, np.expand_dims(mask_logits[i,:,j], -1).T, alpha = 0.5, cmap = 'Greens')
                    fig.colorbar(cmap, ax = ax[j+1][i])

            else:
                block_inds = get_x_mask_borders(mask = masks[i,:,j])
                for k in range(len(block_inds)):
                    ax[j+1][i].axvspan(*block_inds[k], facecolor = 'green', alpha = 0.4)

    if show:
        plt.show()

def visualize_concepts():
    '''
    Visualizes samples by connecting them to concepts
    '''
    pass




