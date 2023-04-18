import torch
import numpy as np
import matplotlib.pyplot as plt
from txai.vis.visualize_cbm1 import get_x_mask_borders

def visualize(model, test_tup, n = 3, per_class = False, class_num = None, show = True, sim = False, heatmap = False):
    # Quick function to visualize some samples in test_tup
    # FOR NOW, assume only 2 masks, 2 concepts

    X, times, y = test_tup

    assert not per_class or (class_num is None), "You can't set per_class=True and class_num not None"

    choices = np.arange(X.shape[1])
    if class_num is not None:
        choices = choices[(y == class_num).cpu().numpy()]
    inds = torch.from_numpy(np.random.choice(choices, size = (n,), replace = False)).long()
    fig, ax = plt.subplots(2, n, sharex = True)

    sampX, samp_times, samp_y = X[:,inds,:], times[:,inds], y[inds]
    x_range = torch.arange(sampX.shape[0])

    model.eval()
    with torch.no_grad():
        out = model(sampX, samp_times, captum_input = False)
        #pred, pred_mask, mask_in, ste_mask, smoother_stats, smooth_src = model(sampX, samp_times, captum_input = False)

    pred, pred_mask = out['all_preds']
    #mask = out['ste_mask']
    mask = (out['mask_logits'] > 0.5).float() # FIX LATER
    mask_logits = out['mask_logits'].unsqueeze(-1).detach().cpu().numpy()
    print('mask_logits', mask_logits.shape)
    smoother_stats = out['smoother_stats']
    smooth_src = out['smooth_src']

    pred = pred.softmax(dim=1).argmax(dim=1)
    pred_mask = pred_mask.softmax(dim=1).argmax(dim=1)
    print('pred', pred.shape)
    print('smoother_stats', smoother_stats)

    title_format1 = 'y={:1d}, yhat={:1d}, yhat_mask={:1d}'

    for i in range(n): # Iterate over samples

        # fit lots of info into the title
        yi = samp_y[i].item()
        pi = pred[i].item()
        pmi = pred_mask[i].item()
        ax[0,i].set_title(title_format1.format(yi, pi, pmi))
        ax[1,i].set_title('Smoothed')

        sX = sampX[:,i,:].cpu().numpy()

        ax[0,i].plot(x_range, sX, color = 'black')
        ax[1,i].plot(x_range, smooth_src[:,i,:].cpu().numpy(), color = 'black')

        if heatmap:
            print('ML', mask_logits[i,...].T.shape)
            px, py = np.meshgrid(np.linspace(min(x_range), max(x_range), len(x_range) + 1), [min(sX), max(sX)])
            #ax[0,i].imshow(mask_logits[i,...].T, alpha = 0.5, cmap = 'Greens')
            cmap = ax[0,i].pcolormesh(px, py, mask_logits[i,...].T, alpha = 0.5, cmap = 'Greens')
            fig.colorbar(cmap, ax = ax[0,i])
            cmap = ax[1,i].pcolormesh(px, py, mask_logits[i,...].T, alpha = 0.5, cmap = 'Greens')
            fig.colorbar(cmap, ax = ax[1,i])

        else:
            block_inds = get_x_mask_borders(mask = mask[i,:])

            for k in range(len(block_inds)):
                #print(block_inds[k])
                ax[0,i].axvspan(*block_inds[k], facecolor = 'green', alpha = 0.4)
                ax[1,i].axvspan(*block_inds[k], facecolor = 'green', alpha = 0.4)

    if show:
        plt.show()




