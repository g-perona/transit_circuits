import numpy as np
import cvxpy as cp

from transit_circuits.components import Resistor, Diode, CurrentSource, TTResistor, TransferResistor
from transit_circuits.optimization import Problem
from transit_circuits.transit_network import Line, Station, TransitNetwork

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

def _make_cross():
    D = _make_D_cross()
    stations = _make_stations_cross(D)
    lines = _make_lines_cross(stations)

    return D, stations, lines

def _make_D_grid():
    D = np.zeros((12, 12))
    D[3,0] = 10
    D[3,2] = 10
    D[3,4] = 10
    D[3,7] = 10
    D[4,1] = 10
    D[4,5] = 10
    D[4,8] = 10
    D[7,6] = 10
    D[7,8] = 10
    D[7,10]= 10
    D[8,9] = 10
    D[8,11]= 10

    D += D.T
    np.where(D == 0, -1, D)

    return D

def _make_stations_grid(D):
    return [Station(i) for i in range(D.shape[0])]

def _make_lines_grid(stations):
    return [
        Line(0, [stations[0], stations[3], stations[7], stations[10]], avg_speed=10, frequency=10),
        Line(1, [stations[1], stations[4], stations[8], stations[11]], avg_speed=10, frequency=5),
        Line(2, [stations[2], stations[3], stations[4], stations[5]],  avg_speed=10, frequency=10),
        Line(3, [stations[6], stations[7], stations[8], stations[9]],  avg_speed=10, frequency=10),
    ]

def _make_grid():
    D = _make_D_grid()
    stations = _make_stations_grid(D)
    lines = _make_lines_grid(stations)

    return D, stations, lines

def test_station_creation():
    station = Station(id=1)
    assert station.id == 1
    assert station.lines == {}
    assert station._transfer_diodes == {}
    assert station._transfer_resistors == {}

def test_line_creation():
    stations = _make_stations_cross(_make_D_cross())
    id = 1
    line_stations = [stations[0], stations[1], stations[2]]
    avg_speed = 20
    line = Line(id, line_stations, avg_speed)
    assert line.id == 1
    assert line.stations == line_stations
    assert line.avg_speed == avg_speed

def _test_nontransfer_station_components(station, line):
    assert len(station._transfer_diodes[line][+1]) == 0
    assert len(station._transfer_diodes[line][-1]) == 0
    assert len(station._transfer_resistors[line][+1]) == 0
    assert len(station._transfer_resistors[line][-1]) == 0

def _test_seg(seg, seg_next=None, tt=None):
    assert seg.td_diode.source is seg.v_station
    if seg_next is not None:
        assert seg.td_diode.drain is seg.tt_resistor.source
        assert seg.tt_resistor.drain is seg_next.v_station
        assert seg.tt_resistor.C == 1/tt

def test_transit_network_one_line():
    D, stations, lines = _make_cross()
    lines_one = [lines[0]]
    tn = TransitNetwork(D, stations, lines_one)

    assert np.array_equal(tn.D, D)
    assert tn.stations == stations
    assert tn.lines == lines_one

    line = tn.lines[0]
    s_0 = tn.stations[0]
    s_1 = tn.stations[1]
    s_2 = tn.stations[2]
    tt_01 = D[0, 1] / line.avg_speed
    tt_12 = D[1, 2] / line.avg_speed

    assert len(s_0.lines) == 1
    assert line in s_0.lines
    _test_nontransfer_station_components(s_0, line)
    _test_seg(s_0.lines[line][+1], s_1.lines[line][+1], tt_01)
    _test_seg(s_0.lines[line][-1])

    assert len(s_1.lines) == 1
    assert line in s_1.lines
    _test_nontransfer_station_components(s_1, line)
    _test_seg(s_1.lines[line][+1], s_2.lines[line][+1], tt_12)
    _test_seg(s_1.lines[line][-1], s_0.lines[line][-1], tt_01)

    assert len(s_2.lines) == 1
    assert line in s_2.lines
    _test_nontransfer_station_components(s_2, line)
    _test_seg(s_2.lines[line][+1])
    _test_seg(s_2.lines[line][-1], s_1.lines[line][-1], tt_12)

def _test_transfer_component(station, line_a, line_b, d_a, d_b, diode, resistor):
    assert diode.source is station.lines[line_a][d_a].v_station
    assert diode.drain is resistor.source
    assert resistor.drain is station.lines[line_b][d_b].v_diode
    assert str(resistor.C) == str(line_b.frequency * 2)

def _test_transfer_components(station, line_a, line_b):
    diodes = station.get_transfer_diodes(line_a, line_b)
    resistors = station.get_transfer_resistors(line_a, line_b)

    _test_transfer_component(station, line_a, line_b, +1, +1, diodes[0], resistors[0])
    _test_transfer_component(station, line_a, line_b, +1, -1, diodes[1], resistors[1])
    _test_transfer_component(station, line_a, line_b, -1, +1, diodes[2], resistors[2])
    _test_transfer_component(station, line_a, line_b, -1, -1, diodes[3], resistors[3])

def test_transit_network_cross():
    D, stations, lines = _make_cross()

    tn = TransitNetwork(D, stations, lines)

    assert np.array_equal(tn.D, D)
    assert tn.stations == stations
    assert tn.lines == lines

    line_0 = tn.lines[0]
    line_1 = tn.lines[1]
    s_0 = tn.stations[0]
    s_1 = tn.stations[1]
    s_2 = tn.stations[2]
    s_3 = tn.stations[3]
    s_4 = tn.stations[4]
    tt_01 = D[0, 1] / line_0.avg_speed
    tt_12 = D[1, 2] / line_0.avg_speed
    tt_13 = D[1, 3] / line_1.avg_speed
    tt_14 = D[1, 4] / line_1.avg_speed

    assert len(s_0.lines) == 1
    assert line_0 in s_0.lines
    _test_nontransfer_station_components(s_2, line_0)
    _test_seg(s_0.lines[line_0][+1], s_1.lines[line_0][+1], tt_01)
    _test_seg(s_0.lines[line_0][-1])

    assert len(s_1.lines) == 2
    assert line_0 in s_1.lines
    _test_seg(s_1.lines[line_0][+1], s_2.lines[line_0][+1], tt_12)
    _test_seg(s_1.lines[line_0][-1], s_0.lines[line_0][-1], tt_01)
    _test_transfer_components(s_1, line_1, line_0)

    assert len(s_2.lines) == 1
    assert line_0 in s_2.lines
    _test_nontransfer_station_components(s_2, line_0)
    _test_seg(s_2.lines[line_0][+1])
    _test_seg(s_2.lines[line_0][-1], s_1.lines[line_0][-1], tt_12)

    assert len(s_3.lines) == 1
    assert line_1 in s_3.lines
    _test_nontransfer_station_components(s_3, line_1)
    _test_seg(s_3.lines[line_1][+1], s_1.lines[line_1][+1], tt_13)
    _test_seg(s_3.lines[line_1][-1])

    assert line_1 in s_1.lines
    _test_seg(s_1.lines[line_1][+1], s_4.lines[line_1][+1], tt_14)
    _test_seg(s_1.lines[line_1][-1], s_3.lines[line_1][-1], tt_13)
    _test_transfer_components(s_1, line_0, line_1)


    assert len(s_4.lines) == 1
    assert line_1 in s_4.lines
    _test_nontransfer_station_components(s_4, line_1)
    _test_seg(s_4.lines[line_1][+1])
    _test_seg(s_4.lines[line_1][-1], s_1.lines[line_1][-1], tt_14)

def test_build_subcircuit_one_line():
    D, stations, lines = _make_cross()
    lines = [lines[0]]
    tn = TransitNetwork(D, stations, lines)
    p = Problem()
    
    origin = stations[0]
    destination = stations[2]
    flow = 10.0

    tn._build_subcircuit(origin, destination, flow, p)

    assert len(set(p.objective_terms)) == len(p.objective_terms)
    assert len(p.objective_terms) == 7
    assert len(set(p.constraints)) == len(p.constraints)
    assert len(p.constraints)== 9

    p.solve()

    t = tn.trips[origin][destination]
    assert np.isclose(t._v_origin.value - t._v_destination.value, 23, rtol=0, atol=1e-3)

def test_build_subcircuit_cross():
    D, stations, lines = _make_cross()
    tn = TransitNetwork(D, stations, lines)
    p = Problem()

    origin = stations[0]
    destination = stations[4]
    flow = 20.0

    tn._build_subcircuit(origin, destination, flow, p)

    assert len(set(p.objective_terms)) == len(p.objective_terms)
    assert len(p.objective_terms) == 19
    assert len(set(p.constraints)) == len(p.constraints)
    assert len(p.constraints) == 23

    p.solve()

    t = tn.trips[origin][destination]
    assert np.isclose(t._v_origin.value - t._v_destination.value, 39, rtol=0, atol=1e-5)
    
def test_build_subcircuit_grid():
    D, stations, lines = _make_grid()
    tn = TransitNetwork(D, stations, lines)
    p_1 = Problem()

    origin_1 = stations[2]
    destination_1 = stations[8]
    flow_1 = 20.0

    tn._build_subcircuit(origin_1, destination_1, flow_1, p_1)

    p_1.solve()
    
    t_1 = tn.trips[origin_1][destination_1]
    assert np.isclose(t_1._v_origin.value - t_1._v_destination.value, 42, atol=1e-5)

    p_2 = Problem()
    
    origin_2 = stations[2]
    destination_2 = stations[7]
    flow_2 = 20.0

    tn._build_subcircuit(origin_2, destination_2, flow_2, p_2)

    p_2.solve()

    t_2 = tn.trips[origin_2][destination_2]
    v_2 = flow_2 * (1/20 + 1 + ((20/21) + (20/63)) ** -1)
    assert np.isclose(t_2._v_origin.value - t_2._v_destination.value, v_2, atol=1e-5)
    