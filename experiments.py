from transit_circuits.transit_network import Line, Station, TransitNetwork
from transit_circuits import utils

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

matplotlib.rcParams['axes.labelsize'] = 20
matplotlib.rcParams['axes.titlesize'] = 20

matplotlib.rcParams['lines.linewidth'] = 4


def grid_2to7_demand(demand=100):
    tn, D, stations, lines = utils.make_grid()

    OD_27 = {
        stations[2]: {stations[7]: demand}
    }

    tn.calculate_flows(OD_27)

    return utils.plot_freq_and_flows(tn)


def grid_all_pairs_line_3(line_3_freq, tod=None):
    tn, D, stations, lines = utils.make_grid()
    OD = utils.make_grid_OD(tn, tod)

    tn.lines[3].frequency = line_3_freq

    tn = TransitNetwork(D, stations, lines)
    tn.calculate_flows(OD)
    
    return utils.plot_freq_and_flows(tn)

def grid_all_pairs_line_3_list(headways, tod=None):
    tn, D, stations, lines = utils.make_grid()
    OD = utils.make_grid_OD(tn, tod)
    
    flows = {0:[], 1:[], 2:[], 3:[]}
    times = []

    for hw in headways:
        print(hw)
        
        tn.reset()
        tn.lines[3].frequency = 60/hw
        
        tn = TransitNetwork(D, stations, lines)
        problems = tn.calculate_flows(OD)

        flows[0].append(tn.stations[7].lines[lines[0]][-1].tt_resistor.total_current)
        flows[1].append(tn.stations[4].lines[lines[1]][+1].tt_resistor.total_current)
        flows[2].append(tn.stations[4].lines[lines[2]][-1].tt_resistor.total_current)
        flows[3].append(tn.stations[7].lines[lines[3]][+1].tt_resistor.total_current)

        times.append(utils.calculate_total_travel_time(problems=problems))

    tn.lines[3].frequency = tn.lines[0].frequency
    
    return utils.plot_freq_var_flows_total_time(tn, headways, flows, times)
