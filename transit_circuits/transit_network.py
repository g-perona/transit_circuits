from transit_circuits.components import TTResistor, TransferResistor, Diode, CurrentSource
from transit_circuits.optimization import Problem

import matplotlib.pyplot as plt
import networkx as nx
import cvxpy as cp
import numpy as np

from tqdm import tqdm

import json

class Line():
    '''
    A class representing a transit line, that keeps track of order of the stations it serves, its own speed,
    and its own service frequency.
    Attributes:
    - id: the identification number of the line
    - stations: the list of stations in the order that they are served in the positive direction.
    - avg_speed_kpm: the average speed of the line in kilometers per minute.
    - frequency_vpm: the frequency with which the line's vehicles serve the stations in vehicles per minute.
    '''
    def __init__(self, id:int, stations:list = [], avg_speed_kph=10, frequency_vph=None, headway_mpv=None):
        '''
        Constructor for a transit line.
        Parameters:
        - id: The identification number of the line.
        - stations: A list of Station objects.
        - avg_speed_kmh: The average speed of the line in kilometers per hour.
        - frequency_vph: The frequency of the line in vehicles per hour.
        - headway_mpv: The headway of the line in minutes per vehicle. Ignored if frequency is specified
        '''

        self.id = id
        self.stations = stations
        self.avg_speed_kpm = avg_speed_kph / 60 # Speed in km/minute

        if frequency_vph is None and headway_mpv is None:
            raise ValueError("Either frequency or headway must be specified.")
        
        if frequency_vph is not None:
            self.frequency_vpm = frequency_vph / 60

        if frequency_vph is not None and headway_mpv is not None:
            raise Warning("Both frequency and headway are specified. Using frequency.")

        if headway_mpv is not None:
            self.frequency_vpm = 60/headway_mpv

    def _update_frequency(self, frequency_vph):
        self.frequency_vpm = frequency_vph/60
        for s in self.stations:
            s._update_frequency(self, self.frequency_vpm)
    
    def get_headway(self):
        return 60/self.frequency_vpm

class _LineSegment():
    def __init__(self, line:Line, D_km):
        self.v_station = cp.Variable()
        self.v_diode = cp.Variable()

        self.td_diode = Diode(self.v_station, self.v_diode)
        
        self.line = line
        self.travel_time_m = D_km / line.avg_speed_kpm

        self.tt_resistor = None
    
    def _make_resistor(self, v_next):
        self.v_next = v_next
        self.tt_resistor = TTResistor(self.travel_time_m, self.v_diode, self.v_next)

class Station():
    def __init__(self, id, x=None, y=None):
        self.lines={}
        self.id = id
        self._transfer_diodes = {}
        self._transfer_resistors = {}
        self.x = x
        self.y = y

    def _add_transfer_components(self, l1:Line):
        self._transfer_diodes[l1] = {
            +1:{l:{+1:{}, -1:{}} for l in self.lines if l != l1}, 
            -1:{l:{+1:{}, -1:{}} for l in self.lines if l != l1}
        }
        self._transfer_resistors[l1] = {
            +1:{l:{+1:{}, -1:{}} for l in self.lines if l != l1}, 
            -1:{l:{+1:{}, -1:{}} for l in self.lines if l != l1}
        }
        for l2 in self.lines:
            if l1 == l2:
                continue
            for component_dict in (self._transfer_diodes, self._transfer_resistors):
                for direction in (-1,+1):
                    component_dict[l2][direction][l1] = {+1:{}, -1:{}}
            for d_l1 in (-1,+1):
                for d_l2 in (-1,+1):
                    v_diode1 = cp.Variable()
                    v_diode2 = cp.Variable()
                    self._transfer_diodes[l1][d_l1][l2][d_l2] = Diode(self.lines[l1][d_l1].v_station, v_diode1)
                    self._transfer_diodes[l2][d_l2][l1][d_l1] = Diode(self.lines[l2][d_l2].v_station, v_diode2)
                    self._transfer_resistors[l1][d_l1][l2][d_l2] = TransferResistor(l2.frequency_vpm, v_diode1, self.lines[l2][d_l2].v_diode)
                    self._transfer_resistors[l2][d_l2][l1][d_l1] = TransferResistor(l1.frequency_vpm, v_diode2, self.lines[l1][d_l1].v_diode)

    def add_line(self, line, seg_next, seg_prev):
        self.lines[line] = {
            +1: seg_next,
            -1: seg_prev
        }
        
        self._add_transfer_components(line)

    def get_transfer_diodes(self, line1: Line, line2: Line) -> list[Diode]:
        return [
            self._transfer_diodes[line1][+1][line2][+1],
            self._transfer_diodes[line1][+1][line2][-1],
            self._transfer_diodes[line1][-1][line2][+1],
            self._transfer_diodes[line1][-1][line2][-1]
        ]
    
    def get_transfer_resistors(self, line1: Line, line2: Line) -> list[TransferResistor]:
        return [
            self._transfer_resistors[line1][+1][line2][+1],
            self._transfer_resistors[line1][+1][line2][-1],
            self._transfer_resistors[line1][-1][line2][+1],
            self._transfer_resistors[line1][-1][line2][-1]
        ]
    
    def _update_frequency(self, line, frequency_vpm):
        for l in self.lines:
            if l == line: continue
            tr = self.get_transfer_resistors(l, line)
            for r in tr:
                r.update_frequency(line.frequency_vpm)

class Trip():
    def __init__(self, origin: Station, destination: Station, flow):
        self.origin = origin
        self.destination = destination
        self.flow = flow
        self._current_source = CurrentSource(flow, source=cp.Variable(), drain=cp.Variable())
        self._v_origin = self._current_source.drain
        self._v_destination = self._current_source.source
        self._origin_resistors = []
        self._destination_diodes = []
        
        for line in destination.lines:
            self._destination_diodes.append(Diode(destination.lines[line][+1].v_station, self._v_destination))
            self._destination_diodes.append(Diode(destination.lines[line][-1].v_station, self._v_destination))
        for line in self.origin.lines:
            self._origin_resistors.append(TransferResistor(line.frequency_vpm, self._v_origin, self.origin.lines[line][+1].v_diode))
            self._origin_resistors.append(TransferResistor(line.frequency_vpm, self._v_origin, self.origin.lines[line][-1].v_diode))

    def _update_frequency(self):
        self._origin_resistors = []
        self._destination_diodes = []

        for line in self.origin.lines:
            self._origin_resistors.append(TransferResistor(line.frequency_vpm, self._v_origin, self.origin.lines[line][+1].v_diode))
            self._origin_resistors.append(TransferResistor(line.frequency_vpm, self._v_origin, self.origin.lines[line][-1].v_diode))


class TransitNetwork():
    def _parse_station_coords(self):
        for station in self.stations:
            if station.x and station.y:
                self.stations_xy[station.x,station.y] = station
    
    def __init__(self, D:np.array, stations:list[Station], lines:list[Line]):
        self.D = D
        self.stations = stations
        self.stations_xy = {}
        self.lines = lines
        self.trips = {o:{d:None for d in self.stations} for o in self.stations}
        self._parse_station_coords()
        self._disaggregated_currents = {}

        for line in self.lines:
            for i, station in enumerate(line.stations):
                if i+1 < len(line.stations):
                    next_station = line.stations[i+1]
                    D_next = D[station.id, next_station.id]
                else:
                    D_next = np.inf
                if i > 0:
                    prev_station = line.stations[i-1]
                    D_prev =  D[station.id, prev_station.id]
                else:
                    D_prev = np.inf
                
                seg_next = _LineSegment(line, D_next)
                seg_prev = _LineSegment(line, D_prev)
                
                station.add_line(line, seg_next, seg_prev)
                
                if i > 0:
                    prev_station.lines[line][+1]._make_resistor(seg_next.v_station)
                    seg_prev._make_resistor(prev_station.lines[line][-1].v_station)

    def _build_subcircuit(self, origin:Station, destination:Station, flow:float, problem:Problem):
        for s in self.stations:
            for line1 in s.lines:
                seg1 = s.lines[line1]

                problem.add_diode(seg1[+1].td_diode)
                problem.add_diode(seg1[-1].td_diode)

                R_next = seg1[+1].tt_resistor
                R_prev = seg1[-1].tt_resistor

                if R_next is not None:
                    problem.add_resistor(R_next)
                if R_prev is not None:
                    problem.add_resistor(R_prev)

                for line2 in s.lines:
                    if line1 == line2:
                        continue
                    problem.add_diode(*s.get_transfer_diodes(line1, line2))
                    problem.add_resistor(*s.get_transfer_resistors(line1, line2))

        t = Trip(origin, destination, flow)
        self.trips[origin][destination] = t

        problem.add_resistor(*t._origin_resistors)
        problem.add_diode(*t._destination_diodes)
        problem.add_current_source(t._current_source)
        problem._add_constraint(t._v_origin == 0)

    def _save_disaggregated(self, _save_disaggregated, origin, destination):
        if not _save_disaggregated:
            return
        
        self._disaggregated_currents[origin] = self._disaggregated_currents.get(origin, {})
        self._disaggregated_currents[origin][destination] = self._disaggregated_currents[origin].get(destination, {})
        disagg_od = self._disaggregated_currents[origin][destination]

        for s in self.stations:
            for line in s.lines:
                seg = s.lines[line]

                R_next = seg[+1].tt_resistor
                R_prev = seg[-1].tt_resistor

                if R_next is not None:
                    I_next = R_next.history[-1] * R_next.C
                else:
                    I_next = 0
                if R_prev is not None:
                    I_prev = R_prev.history[-1] * R_prev.C
                else:
                    I_prev = 0

                disagg_od[s] = disagg_od.get(s, {})
                disagg_od[s][line] = {+1: I_next, -1: I_prev}

    def calculate_flows(self, OD_trips:np.array, origins = None, destinations = None, _save_disaggregated=False):
        origins = origins or OD_trips.keys()
        problems = []
        for origin in tqdm(origins):
            if not destinations:
                destinations = OD_trips[origin].keys()
            for destination in destinations:
                if origin == destination:
                    continue
                p = Problem()
                self._build_subcircuit(origin, destination, OD_trips[origin][destination], p)
                p.solve()
                self._save_disaggregated(_save_disaggregated, origin, destination)
                problems.append(p)
        return problems

    def reset(self):
        """Resets all component histories in the network."""
        for station in self.stations:
            for line in station.lines:
                for direction in (-1, +1):
                    segment = station.lines[line][direction]

                    # Reset line segment components
                    if segment.tt_resistor:
                        segment.tt_resistor.reset()
                    segment.td_diode.reset()

                # Reset transfer components
                for transfer_dict in [station._transfer_resistors, station._transfer_diodes]:
                    for components in transfer_dict[line][direction].values():
                        for component in components.values():
                            component.reset()

        # Reset trip components
        for origin in self.trips:
            for destination, trip in self.trips[origin].items():
                if trip:
                    for component in trip._origin_resistors + trip._destination_diodes:
                        component.reset()
                    trip._current_source.reset()
        
        self._disaggregated_currents = {}

    def update_headway(self, line: Line, headway_mpv):
        frequency_vph = 60/headway_mpv
        self.update_frequency(line=line, frequency_vph=frequency_vph)
    
    def update_frequency(self, line:Line, frequency_vph):
        if isinstance(line, Line):
            line._update_frequency(frequency_vph)
        else:
            raise ValueError("Pass a Line object, not a line id.")
        
        for o, trips_o in self.trips.items():
            for d, trips_od in self.trips[o].items():
                if o == d or trips_od is None: continue
                trips_od._update_frequency()
    
    def save_state(self, filename="transit_network_state.json"):
        """
        Saves the current state of the transit network to a text file.
        
        Parameters:
        - filename (str): Name of the file to save the state.
        """

        # Prepare station data
        station_data = [{
            "id": station.id,
            "x": station.x,
            "y": station.y
        } for station in self.stations]

        # Prepare line data
        line_data = [{
            "id": line.id,
            "stations": [station.id for station in line.stations],
            "avg_speed_kpm": line.avg_speed_kpm,
            "frequency_vpm": line.frequency_vpm
        } for line in self.lines]

        # Prepare trips data
        trip_data = []
        for origin, destinations in self.trips.items():
            for destination, trip in destinations.items():
                if trip:
                    trip_data.append({
                        "origin": origin.id,
                        "destination": destination.id,
                        "flow": trip.flow
                    })

        # Create a dictionary of the network state
        network_state = {
            "stations": station_data,
            "lines": line_data,
            "trips": trip_data
        }

        # Save to file in JSON format for readability
        with open(filename, "w") as file:
            json.dump(network_state, file, indent=4)

        print(f"Transit network state saved to {filename}.")
        return network_state