import cvxpy as cp
import numpy as np

class Component():
    def __init__(self, source, drain):
        self.source = source
        self.drain = drain
        self.voltage = self.source - self.drain
        self.history = np.array([])
    
    def cache(self):
        self.history = np.append(self.history, self.voltage.value)
    
    def reset(self):
        self.history = np.array([])

class Resistor(Component):
    def __init__(self, C:float|cp.Variable, source:cp.Variable, drain:cp.Variable):
        '''
        Constructor for the Resistor object.
        Parameters:
        - C: The conductance of the resistor.
        - source: The source node for the resistor, a cvxpy Variable object.
        - drain: The drain node for the resistor, a cvxpy Variable object.
        '''
        super().__init__(source, drain)
        self._update_C(C)
        self.total_current = 0
    
    def cache(self):
        super().cache()
        self.total_current = self.C * np.sum(self.history)

    def _update_C(self, C):
        self.C = C
        self.R = 1/C
        self.constraint = None
        self.is_variable = isinstance(C, cp.Variable)
        if self.is_variable:
            self.constraint = C >= 0
        self.energy = 0.5 * self.C * cp.power(self.voltage, 2)
        self.current = self.C * self.voltage

class TTResistor(Resistor):
    def __init__(self, travel_time_m, source, drain):
        '''
        Constructor for a travel time resistor object, modeling the time spent to travel between stations.
        Parameters:
        - travel_time: the travel time in minutes for the segment modeled by the resistor.
        - source: source node, the v_diode value of some direction of a line at a station 
            from which current will flow through the resistor.
        - drain: drain node, the v_station value of some direction of a line at a station 
            to which current may flow through the resistor.
        '''
        C = 1 / travel_time_m
        super().__init__(C, source, drain)

class TransferResistor(Resistor):
    def __init__(self, freq_vpm, source, drain):
        '''
        Constructor for a transfer resistor, modeling the time taken to transfer to a line.
        Parameters:
        - freq: The frequency of the line to be connected to in vehicles per minute.
        - source: The source node, v_station of some direction of a line at a station
            from which current will flow through the resistor.
        - drain: The drain node, v_diode of some direction of a line at a station
            to which current will flow to the resistor, whose frequency determines
            the resistance of the resistor.
        '''
        C = freq_vpm * 2
        super().__init__(C, source, drain)

    def update_frequency(self, freq_vpm):
        C = freq_vpm * 2
        self._update_C(C)


class Diode(Component):
    def __init__(self, source, drain):
        super().__init__(source, drain)
        # self.constraint = self.source - self.drain >= 0
        self.constraint = self.voltage <= 0

class CurrentSource(Component):
    def __init__(self, I, source, drain):
        super().__init__(source, drain)
        self.I = I
        # self.energy = self.I * (self.drain - self.source)
        self.energy = self.I * (self.source - self.drain)