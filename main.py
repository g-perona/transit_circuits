from transit_circuits.transit_network import Line, Station, TransitNetwork
from transit_circuits import utils

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np

from experiments import grid_2to7_demand, grid_all_pairs_line_3, grid_all_pairs_line_3_list

matplotlib.rcParams['figure.figsize'] = (6, 6)
# plt.rcParams.update({'font.size': 18})

# Generate plots and save to PDF
from datetime import datetime as dt
with PdfPages(f'figs/transit_network_plots{dt.now():%Y-%m-%d_%H:%M}.pdf') as pdf:
    # Generate side-by-side Frequency and Flow plots for OD_27
    fig, ax= grid_2to7_demand(demand=100)
    pdf.savefig(fig)
    plt.close()

    headways = np.logspace(start=0, stop=20, base=1.5, num=20)
    fig, ax = grid_all_pairs_line_3(line_3_freq=30, tod=None)
    pdf.savefig(fig)
    plt.close()

    fig, ax = grid_all_pairs_line_3(line_3_freq=60/headways[-1], tod=None)
    pdf.savefig(fig)
    plt.close()

    fig, ax = grid_all_pairs_line_3_list(headways=headways, tod=None)
    pdf.savefig(fig)
    plt.close()

print("Plots saved to 'transit_network_plots.pdf'.")