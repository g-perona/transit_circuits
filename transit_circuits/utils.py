from transit_circuits.transit_network import Line, Station, TransitNetwork

import numpy as np

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
        Line(0, [stations[0], stations[3], stations[7], stations[10]], avg_speed=30, frequency=10),
        Line(1, [stations[1], stations[4], stations[8], stations[11]], avg_speed=30, frequency=10),
        Line(2, [stations[2], stations[3], stations[4], stations[5]],  avg_speed=30, frequency=20),
        Line(3, [stations[6], stations[7], stations[8], stations[9]],  avg_speed=30, frequency=5),
    ]

def make_grid():
    D = _make_D_grid()
    stations = _make_stations_grid(D)
    lines = _make_lines_grid(stations)

    return TransitNetwork(D, stations, lines)