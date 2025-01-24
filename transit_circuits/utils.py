from transit_circuits.transit_network import Line, Station, TransitNetwork

import numpy as np
import matplotlib.pyplot as plt
plt.ioff()

def _make_D_cross():
    return np.array([
    [0, 10, -1, -1, -1],
    [10, 0, 8, 5, 4],
    [-1, 8, 0, -1, -1],
    [-1, 5, -1, 0, -1],
    [-1, 4, -1, -1, 0]
])

def _make_stations_cross(D):
    return [Station(i) for i in range(D.shape[0])]

def _make_lines_cross(stations):
    return [
    Line(0, [stations[0], stations[1], stations[2]], avg_speed=10, frequency=1),
    Line(1, [stations[3], stations[1], stations[4]], avg_speed=20, frequency=2)
    ]

def make_cross():
    D = _make_D_cross()
    stations = _make_stations_cross(D)
    lines = _make_lines_cross(stations)

    return D, stations, lines

def _make_D_grid():
    D = np.zeros((12, 12))
    D[3,0] = 1
    D[3,2] = 1
    D[3,4] = 1
    D[3,7] = 1
    D[4,1] = 1
    D[4,5] = 1
    D[4,8] = 1
    D[7,6] = 1
    D[7,8] = 1
    D[7,10]= 1
    D[8,9] = 1
    D[8,11]= 1

    D += D.T
    np.where(D == 0, -1, D)

    return D

def _make_stations_grid(D):
    x = [1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2]
    y = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3]
    stations = []
    for i in range(D.shape[0]):
        stations.append(Station(i, x[i], y[i]))
    return stations

def _make_lines_grid(stations):
    return [
        Line(0, [stations[0], stations[3], stations[7], stations[10]], avg_speed=40, frequency=30),
        Line(1, [stations[1], stations[4], stations[8], stations[11]], avg_speed=40, frequency=30),
        Line(2, [stations[2], stations[3], stations[4], stations[5]],  avg_speed=40, frequency=30),
        Line(3, [stations[6], stations[7], stations[8], stations[9]],  avg_speed=40, frequency=30),
    ]

def make_grid():
    D = _make_D_grid()
    stations = _make_stations_grid(D)
    lines = _make_lines_grid(stations)

    return TransitNetwork(D, stations, lines), D, stations, lines

def make_grid_OD(tn, tod=None):
    if tod is None:
        OD = {o: {d:0 for d in tn.stations} for o in tn.stations}
        for o in tn.stations:
            for d in tn.stations:
                if o == d:
                    continue
                OD[o][d] = 10
        
        return OD
    
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
    
    if tod == 'morning':
        return OD_morning
    if tod == 'evening':
        return OD_evening

def _plot_2x1():
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    return fig, ax

def _plot_3x1():
    fig, ax = plt.subplots(1,3,figsize=(18,6))
    return fig, ax

def calculate_total_travel_time(problems):
    total = 0
    for p in problems:
        for r in p.resistors:
            total += r.voltage.value
    return total

def plot_freq_and_flows(tn):
    fig, ax = _plot_2x1()
    tn.plot_frequency(ax=ax[0])
    ax[0].set_title("(a) Frequency Plot")
    tn.plot_flow(ax=ax[1], label_fraction=0.5)
    ax[1].set_title("(b) Flow Plot")
    return fig, ax

def plot_freq_and_var_flows(tn, hw, flows):
    fig, ax = _plot_2x1()
    tn.plot_frequency(ax=ax[0])
    ax[0].set_title("(a) Frequency Plot")
    
    for flows_l in flows:
        ax[1].plot(hw, flows_l)
    
    ax[1].xlabel(f"Line 3 headway")
    ax[1].ylabel(f"Ridership through selected resistors")
    ax[1].xscale('log')

    return fig, ax

def plot_freq_var_flows_total_time(tn, headways, flows, times):
    fig, ax = _plot_3x1()
    tn.plot_frequency(ax=ax[0], line_styles={tn.lines[3]: 'solid'})
    ax[0].set_title("(a)")
    # ax[0].set_title("(a) Frequency Plot")
    
    ax[1].plot(headways, flows[0], label=r'$3\to 7$')
    ax[1].plot(headways, flows[2], label=r'$3\to 4$', c='tab:green')
    ax[1].plot(headways, flows[3], label=r'$7\to 8$', c='tab:red')
    
    ax[1].set_xlabel(f"Red Line headway")
    ax[1].set_ylabel(f"Flows through selected resistors")
    ax[1].set_title(f"(b)")
    # ax[1].set_title(f"(b) Varying flows through central links")
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].grid(which='both', alpha=0.5)
    ax[1].legend()

    ax[2].plot(headways, times)
    ax[2].set_xlabel(f"Red Line headway")
    ax[2].set_ylabel(f"Total person-hours of travel")
    ax[2].set_title(f"(c)")
    # ax[2].set_title(f"(c) Varying total person-hours of travel")
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].grid(which='both', alpha=0.5)

    return fig, ax



    
