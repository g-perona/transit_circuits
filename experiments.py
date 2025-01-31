from transit_circuits.transit_network import Line, Station, TransitNetwork
from transit_circuits.transit_network_plotter import TransitNetworkPlotter as TNP
from transit_circuits import utils

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

matplotlib.rcParams['axes.labelsize'] = 17
matplotlib.rcParams['axes.titlesize'] = 20

matplotlib.rcParams['lines.linewidth'] = 4


def grid_2to7_demand(demand=100):
    D, stations, lines = utils.make_grid()
    tn = TransitNetwork(D, stations, lines)
    OD_27 = {stations[2]: {stations[7]: demand}}
    tn.calculate_flows(OD_27)

    return utils.plot_freq_and_flows(tn)


def grid_all_pairs_line_3(line_3_freq, tod=None):
    D, stations, lines = utils.make_grid()
    lines[3].frequency_vpm = line_3_freq
    tn = TransitNetwork(D, stations, lines)

    OD = utils.make_grid_OD(tn, tod)
    tn.calculate_flows(OD)

    return utils.plot_freq_and_flows(tn)

def grid_all_pairs_line_3_list(headways, tod=None, pdf=None):
    D, stations, lines = utils.make_grid()
    tn = TransitNetwork(D, stations, lines)
    OD = utils.make_grid_OD(tn, tod)
    
    flows = {0:[], 1:[], 2:[], 3:[]}
    times = []

    for hw in headways:
        print(hw)
        
        tn.update_headway(tn.lines[3], headway_mpv=hw)
        problems = tn.calculate_flows(OD, _save_disaggregated=True)

        flows[0].append(tn.stations[7].lines[lines[0]][-1].tt_resistor.total_current)
        flows[1].append(tn.stations[4].lines[lines[1]][+1].tt_resistor.total_current)
        flows[2].append(tn.stations[4].lines[lines[2]][-1].tt_resistor.total_current)
        flows[3].append(tn.stations[7].lines[lines[3]][+1].tt_resistor.total_current)

        times.append(utils.calculate_total_travel_time(problems=problems))

        # plotter = TNP(tn)
        # for i, o in enumerate(tn.stations):
        #     fig, ax = plt.subplots(3, 4, figsize=(24,18))
        #     for j, d in enumerate(tn.stations):
        #         ax_ij = ax[j//4][j%4]
        #         if i == j:
        #             hws = [l.get_headway() for l in tn.lines]
        #             line_ids = [l.id for l in tn.lines]
        #             ax_ij.bar(line_ids, hws)
        #             continue
        #         plotter.plot_flow_one_to_one(ax=ax_ij, origin=o, destination=d)
        #     pdf.savefig()
        #     plt.close()

        tn.reset()

    tn.lines[3].frequency_vpm = tn.lines[0].frequency_vpm
    D, stations, lines = utils.make_grid()
    tn1 = TransitNetwork(D, stations, lines)
    OD_27 = {stations[2]: {stations[7]: 100}}
    tn1.calculate_flows(OD_27)
    
    # return utils.plot_freq_var_flows_total_time(tn, headways, flows, times)
    # return utils.plot_freq_single_od_var_flows_total_time(tn, tn1, headways, flows, times)
    return utils.plot_freq_single_od_var_flows(tn, tn1, headways, flows)
