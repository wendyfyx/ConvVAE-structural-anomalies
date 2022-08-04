import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from utils.general_util import map_list_with_dict


def plot_embedding(embeddings, y, palette, prefix=""):
    '''Plot 2D embedding (first 2 dimensions)'''

    fig, ax = plt.subplots(1, figsize=(9, 9))
    df_vis = pd.DataFrame({'dim1': embeddings[:, 0], 'dim2': embeddings[:, 1],
                           'label': y})
    g=sns.scatterplot(x='dim1', y='dim2', hue='label', 
                      palette=palette, data=df_vis, ax=ax, s=12)
    ax.set_title(f"{prefix} Embeddings")
    g.legend(loc='center right', bbox_to_anchor=(1.45, 0.5), ncol=2)


def plot_embedding_subj(subj_data, palette, use_TSNE=False, **kwargs):
    if use_TSNE:
        plot_embedding(subj_data.X_encoded_tsne, 
                        map_list_with_dict(subj_data.y, subj_data.bundle_num),
                        palette, **kwargs)
    else:
        plot_embedding(subj_data.X_encoded, 
                        map_list_with_dict(subj_data.y, subj_data.bundle_num),
                        palette, **kwargs)


def labeled_colormap(cmap_name, labels, plot_cmap=True):
    '''
        Make custom colormap with given labels
        Returns a dictionary of label name with color value 
    '''
    cmap = plt.get_cmap(cmap_name)
    color_list = cmap(np.linspace(0, 1.0, len(labels)+1))
    colormap = dict(zip(labels, color_list[:-1]))

    if plot_cmap:
        for i, (name, c) in enumerate(colormap.items()):
            plt.axhline(-i, linewidth=10, c=c, label=name)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)

    return colormap


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
        Generate custom range colormap from existing plt ones
    '''
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def symmetrical_colormap(cmap_settings, new_name = None ):
    ''' 
        This function take a colormap and create a new one, as the concatenation of itself by a symmetrical fold.
        From https://stackoverflow.com/a/67005578
    '''
    # get the colormap
    cmap = plt.cm.get_cmap(*cmap_settings)
    if not new_name:
        new_name = "sym_"+cmap_settings[0]  # ex: 'sym_Blues'
    
    # this defined the roughness of the colormap, 128 fine
    n= 128 
    
    # get the list of color from colormap
    colors_r = cmap(np.linspace(0, 1, n))    # take the standard colormap # 'right-part'
    colors_l = colors_r[::-1]                # take the first list of color and flip the order # "left-part"

    # combine them and build a new colormap
    colors = np.vstack((colors_l, colors_r))
    return mcolors.LinearSegmentedColormap.from_list(new_name, colors)