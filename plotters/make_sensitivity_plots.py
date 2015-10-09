""" This code will make Figure 2. It uses some of the libraries available here:
https://github.com/kgullikson88/General
"""

import Sensitivity
import sys
import pandas as pd
import seaborn as sns 
import HelperFunctions
import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import plotutils
import logging


def combine_summaries():
    df_list = []
    for inst in ['HRS', 'CHIRON', 'IGRINS', 'TS23']:
        try:
            df_list.append(pd.read_csv('{}_Sensitivity.csv'.format(inst)))
        except IOError:
            continue
    total = pd.concat(df_list)

    # Only keep up to T=6500
    total = total.loc[total.temperature < 6550]
    total.to_csv('Sensitivity_Dataframe.csv', index=False)
    return


LABELS = {'vsini': 'vsini (km/s)',
          'temperature': 'Temperature (K)',
          'contrast': '$\Delta V$',
          'detection rate': 'Detection Rate'}
def analyze_sensitivity(infile='../data/Sensitivity_Dataframe.csv', interactive=True, outdir='../Figures/', 
                        nrows=4, ncols=3, figsize=(7.5, 9.0), title_fs=11,
                        x_var='vsini', y_var='temperature', c_var='detection rate', **heatmap_kws):
    """
    This uses the output of a previous run of check_sensitivity, and makes plots
    :keyword interactive: If True, the user will pick which stars to plot
    :keyword update: If True, always update the Sensitivity_Dataframe.csv file.
                     Otherwise, try to load that file instead of reading the hdf5 file
    :keyword combine: If True, combine the sensitivity matrix for all stars to get an average sensitivity
    :return:
    """
    df = pd.read_csv(infile)
    
    # Group by a bunch of keys that probably don't change, but could
    groups = df.groupby(('star', 'date', '[Fe/H]', 'logg', 'addmode', 'primary SpT'))
    group_keys = sorted(groups.groups.keys(), key=lambda g: g[0])
    
    # Have the user choose keys
    if interactive:
        for i, key in enumerate(group_keys):
            print('[{}]: {}'.format(i + 1, key))
        inp = raw_input('Enter the numbers of the keys you want to plot (, or - delimited): ')
        chosen = Sensitivity.parse_input(inp, sort_output=False, ensure_unique=True)
        #keys = [k for i, k in enumerate(group_keys) if i + 1 in chosen]
        keys = [group_keys[i-1] for i in chosen]
    else:
        keys = group_keys

    # Compile dataframes for each star
    dataframes = defaultdict(lambda: defaultdict(pd.DataFrame))
    for key in keys:
        logging.info(key)
        g = groups.get_group(key)
        detrate = g.groupby(('temperature', 'vsini', 'logL', 'contrast')).apply(
            lambda df: float(sum(df.significance.notnull())) / float(len(df)))
        significance = g.groupby(('temperature', 'vsini', 'logL', 'contrast')).apply(
            lambda df: np.nanmean(df.significance))
        dataframes['detrate'][key] = detrate.reset_index().rename(columns={0: 'detection rate'})
        dataframes['significance'][key] = significance.reset_index().rename(columns={0: 'significance'})

    # Make heatmap plots for each key. Combine into one big figure to conserve white-space.
    n_subplots = nrows * ncols
    HelperFunctions.ensure_dir(outdir)
    for i, key in enumerate(keys):
        if i % n_subplots == 0:
            fignum = i / n_subplots
            if fignum > 0:
                outfilename = '{}{}_{}_{}_Summary_{}.pdf'.format(outdir, x_var.replace(' ', '_'), y_var.replace(' ', '_'), c_var.replace(' ', '_'), fignum)
                print('Saving to {}'.format(outfilename))
                fig.savefig(outfilename)
                if interactive:
                    plt.show()
                plt.close(fig)
            # make figure and label the appropriate axes
            fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=figsize, squeeze=False)
            for j in range(ncols):
                axes[-1][j].set_xlabel(LABELS[x_var])
            for j in range(nrows):
                axes[j][0].set_ylabel(LABELS[y_var])

            axes = axes.flatten()
            ax = axes[0]
            subplot_number = 0
        else:
            ax = axes[i % n_subplots]

        star = key[0]
        date = key[1]
        spt = key[5]
        #plt.figure(i * 3 + 1)
        if len(dataframes['detrate'][key]) == 0:
            dataframes['detrate'].pop(key)
            dataframes['significance'].pop(key)
            continue

        #sns.heatmap(dataframes['detrate'][key].pivot('temperature', 'vsini', 'detection rate'))
        im = Sensitivity.heatmap(dataframes['detrate'][key][[x_var, y_var, c_var]], 
                                 ax=ax, make_cbar=False, make_labels=False, **heatmap_kws)
    
        ax.set_title('{} / {}'.format(star, date), fontsize=title_fs, fontweight='bold', y=0.99)
        ax.set_xlim((0, 50))

    outfilename = '{}{}_{}_{}_Summary_{}.pdf'.format(outdir, x_var.replace(' ', '_'), y_var.replace(' ', '_'), c_var.replace(' ', '_'), fignum+1)
    print('Saving to {}'.format(outfilename))
    fig.savefig(outfilename)
    if interactive:
        plt.show()
    plt.close(fig)

    return dataframes


def make_2d_plots():
    color_map = sns.cubehelix_palette(as_cmap=True)

    analyze_sensitivity(interactive=True, cmap=color_map)


def make_average_plot(interactive=True, outdir='../Figures/'):
    """ Plots the average (actually median) sensitivity as a function of vsini and temperature.
    """
    sensitivity = pd.read_csv('../data/Sensitivity_Dataframe.csv')

    # Group by a bunch of keys that probably don't change, but could
    groups = sensitivity.groupby(('star', 'date', '[Fe/H]', 'logg', 'addmode', 'primary SpT'))
    group_keys = sorted(groups.groups.keys(), key=lambda g: g[0])
    
    # Have the user choose keys
    if interactive:
        for i, key in enumerate(group_keys):
            print('[{}]: {}'.format(i + 1, key))
        inp = raw_input('Enter the numbers of the keys you want to plot (, or - delimited): ')
        chosen = Sensitivity.parse_input(inp, sort_output=False, ensure_unique=True)
        #keys = [k for i, k in enumerate(group_keys) if i + 1 in chosen]
        keys = [group_keys[i-1] for i in chosen]
    else:
        keys = group_keys

    # Compile dataframes for each star
    dataframes = []
    for key in keys:
        logging.info(key)
        g = groups.get_group(key)
        detrate = g.groupby(('temperature', 'vsini', 'logL', 'contrast')).apply(
            lambda df: float(sum(df.significance.notnull())) / float(len(df)))
        dataframes.append(detrate.reset_index().rename(columns={0: 'median detection rate'}))

    summary = pd.concat(dataframes)
    avg_df = summary.groupby(('temperature', 'vsini')).median().reset_index()

    # Plot
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.18)
    im = Sensitivity.heatmap(avg_df[['vsini', 'temperature', 'median detection rate']], ax=ax,
                             make_cbar=True, make_labels=True, cmap=sns.cubehelix_palette(as_cmap=True))
    ax.set_xlabel('vsini (km/s)')
    ax.set_ylabel('Temperature (K)')

    outfilename = '{}vsini_temperature_detection_rate_median.pdf'.format(outdir)
    print('Saving to {}'.format(outfilename))
    fig.savefig(outfilename)
    if interactive:
        plt.show()


if __name__ == '__main__':
    #combine_summaries()
    #make_2d_plots()
    make_average_plot()

# Input to give when it asks (make_2d_plots):
#10,27,8,13,15,17,23-25,28-30,32,33,36,41,43,44,47,48,1-4

# Input to give when it asks (make_average_plot):
#10,27,8,13,15,17,23-25,28-30,32,33,36,41,43,44,47,48,1-7