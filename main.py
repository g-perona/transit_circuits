from transit_circuits.transit_network import Line, Station, TransitNetwork
from transit_circuits import utils

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np

matplotlib.rcParams['figure.figsize'] = (6, 6)
# plt.rcParams.update({'font.size': 18})

tn, D, stations, lines = utils.make_grid()

OD_27 = {
    stations[2]: {stations[7]: 100}
}

OD_trips = {o: {d: np.random.rand() * 50 + 50 for d in tn.stations if d != o} for o in tn.stations}

OD_morning = {o: {d:0 for d in tn.stations} for o in tn.stations}
OD_evening = {o: {d:0 for d in tn.stations} for o in tn.stations}

for o in tn.stations:
    for d in tn.stations:
        if o == d:
            continue
        OD_morning[o][d] += 5
        OD_evening[o][d] += 5

        if o.id not in [3, 4, 7, 8] and d.id in [3, 4, 7, 8]:
            OD_morning[o][d] += 15
        if o.id in [3, 4, 7, 8] and d.id not in [3, 4, 7, 8]:
            OD_evening[o][d] += 15

for o in tn.stations:
    for d in tn.stations:
        assert OD_morning[o][d] == OD_evening[d][o]



# Solve the network with the specific OD trip
tn.calculate_flows(OD_27)

# Generate plots and save to PDF
from datetime import datetime as dt
with PdfPages(f'figs/transit_network_plots{dt.now():%Y-%m-%d_%H:%M}.pdf') as pdf:
    # Generate side-by-side Frequency and Flow plots for OD_27
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    tn.plot_frequency(ax=ax[0])
    ax[0].set_title("(a) Frequency Plot")
    tn.plot_flow(ax=ax[1], label_fraction=0.5)
    ax[1].set_title("(b) Flow Plot")
    pdf.savefig(fig)
    plt.close(fig)

    tn.reset()

    tn = TransitNetwork(D, stations, lines)
    tn.calculate_flows(OD_morning)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    tn.plot_frequency(ax=ax[0])
    ax[0].set_title("(a) Frequency Plot")
    tn.plot_flow(ax=ax[1], label_fraction=0.3, round=1)
    ax[1].set_title("(b) Flow Plot")
    pdf.savefig(fig)
    plt.close(fig)

    # Solve with OD_morning and update frequency
    tn.reset()
    tn.lines[3].frequency = 5
    tn = TransitNetwork(D, stations, lines)
    tn.calculate_flows(OD_morning)

    # Generate side-by-side Frequency and Flow plots for OD_morning
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    tn.plot_frequency(ax=ax[0])
    ax[0].set_title("(a) Frequency Plot")
    tn.plot_flow(ax=ax[1], label_fraction=0.3, round=1)
    ax[1].set_title("(b) Flow Plot")
    pdf.savefig(fig)
    plt.close(fig)

print("Plots saved to 'transit_network_plots.pdf'.")