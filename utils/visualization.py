import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects


import seaborn as sns
import torch
import numpy as np

import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
# from sklearn.manifold.t_sne import (_joint_probabilities,
#                                     _kl_divergence)
import pickle
import matplotlib.patches as mpatches
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def visualize_TSNE(feat, label, num_class, args, split):

    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    label = torch.cat(label).numpy()
    label = (label > num_class-2).astype(np.int).reshape(-1)
    label = label + 1
    ind = np.argsort(label)
    split_idx = np.array(torch.cat(split)).reshape(-1)

    target_idx = np.where(split_idx == 1)[0]
    source_idx = np.where(split_idx == 0)[0]


    feat = torch.cat(feat)
    dim = feat.size()
    feat = feat.view(dim[0]*dim[1], dim[2])
    target_feat = feat[target_idx]
    source_feat = feat[source_idx]



    # dim = target_feat.size()
    # target_feat = target_feat.view(dim[0]*dim[1], dim[2])
    #
    # src_dim = source_feat.size()
    # source_feat = source_feat.view(src_dim[0]*src_dim[1], src_dim[2])
    # source_feat_select = source_feat[np.random.choice(source_feat.size(0), int(source_feat.size(0)/2))]
    src_label = np.full([source_feat.shape[0]], 0)


    X = np.vstack(target_feat[ind])
    X = np.concatenate([source_feat,X])
    y = np.hstack(label[ind])
    y = np.concatenate([src_label, y])

    digits_proj = TSNE(random_state=2020).fit_transform(X)

    # We choose a color palette with seaborn.
    flatui = ["#1d8bff",  "#ff5e1d","#c5c5c5"]
    palette = np.array(sns.color_palette(flatui, 3))
    label = ['Source', 'Target Known','Target Unknown']
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    for i in range(3):
        idx = np.where(y==i)
        sc = ax.scatter(digits_proj[idx, 0], digits_proj[idx, 1], lw=0, s=15,
                        c=palette[i], label=label[i])

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5, markerscale=4)
    txts = []
    # We add the labels for each digit.
    # txts = ["back_pack", "bike", "bike_helmet", "bookcase", "bottle",
    #                        "calculator", "desk_chair","desk_lamp","desktop_computer","file_cabinet","unk"]
    # for i in range(len(txts)):
    #     # Position of each label.
    #     xtext, ytext = np.median(digits_proj[y == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, txts[i], fontsize=12)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)

    return f, ax, sc, txts


def calc_bins(preds, labels_oneh):
      # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(preds, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(preds[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(preds, labels):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE

def one_hot_encode(data):
    shape = (data.size, data.max()+1)
    one_hot = np.zeros(shape)
    rows = np.arange(data.size)
    one_hot[rows, data] = 1
    return one_hot

def draw_reliability_graph(preds, labels, exp_name):
    with open('{}_vis.pkl'.format(exp_name), 'wb') as f:
        pickle.dump({'preds':preds, 'labels':labels}, f)
    shared_idx = np.where(labels!=labels.max())[0]
    preds = preds[shared_idx]
    labels_shared = labels[shared_idx]
    labels = one_hot_encode(labels_shared)
    ECE, MCE = get_metrics(preds, labels)
    bins, _, bin_accs, _, _ = calc_bins(preds, labels)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    # x/y limits
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)

    # x/y labels
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')

    # Create grid
    ax.set_axisbelow(True) 
    ax.grid(color='gray', linestyle='dashed')

    # Error bars
    plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

    # Draw bars and identity line
    plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
    plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

    # Equally spaced axes
    plt.gca().set_aspect('equal', adjustable='box')

    # ECE and MCE legend
    ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
    MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
    plt.legend(handles=[ECE_patch, MCE_patch])

    #plt.show()
    
    plt.savefig('calibrated_{}.png'.format(exp_name), bbox_inches='tight')




