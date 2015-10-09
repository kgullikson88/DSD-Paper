""" This code will generate Figure 7
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_context('paper', font_scale=2.0)
sns.set_style('white')
sns.set_style('ticks')
sns.set_palette(sns.color_palette("Dark2", 2))

def make_plot(data, equiv_line=True, color='black', label=None):
    # Pull information out
    Tmeas = data['measured_T'].values
    Tact = data['known_T'].values
    Tmeas_err = data[['measured_T_lowerr', 'measured_T_uperr']].values.T
    Tact_err = data[['known_T_lowerr', 'known_T_uperr']].values.T

    # Plot
    plt.errorbar(Tact, Tmeas, yerr=Tmeas_err, xerr=Tact_err, fmt='o', color=color, label=label)
    xlim = np.maximum(plt.xlim(), plt.ylim())
    #plt.xlim(xlim)
    #plt.ylim(xlim)
    if equiv_line:
        plt.plot(xlim, xlim, 'r--')

    plt.xlabel('Expected Temperature (K)')
    plt.ylabel('Measured Temperature (K)')

# Read the data
data = pd.read_csv('../data/known_Teff_data.csv', comment='#')
data = data.convert_objects()

# Drop bad measurements
data = data.loc[~data.HIP.isin([24244])]

# Separate the spectroscopic from imaging detections
spectroscopic = data.loc[data['method'] == 'Spectroscopic']
imaging = data.loc[data['method'] == 'Imaging']

# make the main plot
make_plot(spectroscopic, color='black', equiv_line=False, label='Known from spectroscopy')
make_plot(imaging, color='black', label='Known from imaging')

# Draw an arrow for HIP 79199
#plt.arrow(4700, 5500, 0, -700, head_width=50, head_length=50, fc='k', ec='k', lw=2)
#plt.text(4580, 5200, 'Young', rotation=90, fontsize=20)

plt.xlim((4200, 7500))
plt.ylim((4200, 7500))

# Legend
#plt.legend(loc=4, fancybox=True)

plt.tight_layout()

plt.savefig('../Figures/Known_Binaries.pdf')

plt.show()