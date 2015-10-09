"""
This script goes through the marginalized detection rate files,
and gets the average detection rate as a function of temperature.
"""
from __future__ import print_function, division

import pandas as pd
import numpy as np
import sys
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plottools

sns.set_context('paper', font_scale=2.0)
sns.set_style('white')
sns.set_style('ticks')

import get_parameters

# Some constants
SAMPLE_HIP = [1366, 3300, 12719, 13165, 15338, 17563, 22840, 22958, 24902, 
              26063, 26563, 28691, 33372, 44127, 58590, 65477, 76267, 77516, 
              77858, 79199, 79404, 81641, 84606, 85385, 88290, 89156, 91118, 
              92027, 92728, 98055, 100221, 106786, 113788, 116247, 116611]
SAMPLE_STARS = ['HIP {}'.format(hip) for hip in SAMPLE_HIP]
BASE_DIR = '{}/School/Research'.format(os.environ['HOME'])
INSTRUMENT_DIRS = dict(TS23='{}/McDonaldData/'.format(BASE_DIR),
                       HRS='{}/HET_data/'.format(BASE_DIR),
                       CHIRON='{}/CHIRON_data/'.format(BASE_DIR),
                       IGRINS='{}/IGRINS_data/'.format(BASE_DIR))

def get_undetected_stars(star_list=SAMPLE_STARS):
    """
    Get the undetected stars from my sample.
    """
    full_sample = get_parameters.read_full_sample()

    full_sample['Parsed_date'] = full_sample.Date.map(get_parameters.convert_dates)

    undetected = full_sample.loc[full_sample.Temperature.isnull()]
    matches = undetected.loc[undetected.identifier.isin(star_list)][['identifier', 'Instrument', 'Parsed_date']].copy()

    return matches

def decrement_day(date):
    year, month, day = date.split('-')
    t = datetime.datetime(int(year), int(month), int(day)) - datetime.timedelta(1)
    return t.isoformat().split('T')[0]


def get_detection_rate(instrument, starname, date):
    """
    Read in the detection rate as a function of temperature for the given parameters.
    """
    directory = INSTRUMENT_DIRS[instrument]
    fname = '{}{}_{}_simple.csv'.format(directory, starname.replace(' ', ''), date)
    try:
        df = pd.read_csv(fname)
    except IOError:
        try:
            fname = '{}{}_{}_simple.csv'.format(directory, starname.replace(' ', ''), decrement_day(date))
            df = pd.read_csv(fname)
        except IOError:
            print('File {} does not exist! Skipping!'.format(fname))
            df = pd.DataFrame(columns=['temperature',  'detection rate', 'mean vsini'])
    df['star'] = [starname] * len(df)
    return df


def get_stats(df):
    Teff = df.temperature.values[0]
    low, med, high = np.percentile(df['detection rate'], [16, 50, 84])
    return pd.Series(dict(temperature=Teff, middle=med, low_pct=low, high_pct=high))


def make_plot(stats_df):
    """ Make a plot showing the average and spread of the detection rates
    """
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15, bottom=0.18, right=0.95, top=0.85)
    ax.plot(stats_df.temperature, stats_df.middle, 'r--', lw=2, label='Median')
    ax.fill_between(stats_df.index, stats_df.high_pct, stats_df.low_pct, alpha=0.4, color='blue')

    p = plt.Rectangle((0, 0), 0, 0, color='blue', alpha=0.4, label='16th-84th Percentile')
    ax.add_patch(p)

    plottools.add_spt_axis(ax, spt_values=('M5', 'M0', 'K5', 'K0', 'G5', 'G0', 'F5'))

    leg = ax.legend(loc=4, fancybox=True)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Detection Rate')

    ax.set_xlim((3000, 6550))
    ax.set_ylim((0.0, 1.05))


    return ax



if __name__ == '__main__':
    sample = get_undetected_stars()
    print(sample)

    summary = pd.concat([get_detection_rate(inst, star, d) for inst, star, d in zip(sample.Instrument, 
                                                                                    sample.identifier, 
                                                                                    sample.Parsed_date)])

    # Get statistics on the summary, averaging over the stars
    stats = summary[['temperature', 'detection rate']].groupby('temperature').apply(get_stats)

    make_plot(stats)

    plt.savefig('../Figures/DetectionRate.pdf')
    plt.show()