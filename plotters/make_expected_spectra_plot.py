""" This code will make Figure 8. It uses some of the libraries available here:
https://github.com/kgullikson88/General
"""

import Sensitivity
import HelperFunctions
import os
import matplotlib.pyplot as plt

home = os.environ['HOME']

def make_plot(filename, prim_spt, Tsec, instrument, vsini):
    orders = HelperFunctions.ReadExtensionFits(filename)
    fig, ax, _ = Sensitivity.plot_expected(orders, prim_spt, Tsec, instrument, vsini=vsini)
    fig.subplots_adjust(left=0.2, bottom=0.18, right=0.94, top=0.94)

    plt.show()


if __name__ == '__main__':
    filenames = ['{}/School/Research/CHIRON_data/20140803/HIP_88290.fits'.format(home)]
    prim_spts = ['A2Vn']
    Tsecs = [5800]             
    instruments = ['CHIRON']
    vsinis = [4]

    for fname, p_spt, T2, inst, vsini in zip(filenames, prim_spts, Tsecs, instruments, vsinis):
        make_plot(fname, p_spt, T2, inst, vsini)