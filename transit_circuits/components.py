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

class Resistor(Component):
    def __init__(self, C, source, drain):
        super().__init__(source, drain)
        self.C = C
        self.R = 1/C
        self.constraint = None
        self.is_variable = isinstance(C, cp.Variable)
        if self.is_variable:
            self.constraint = C >= 0
        self.energy = 0.5 * C * cp.power(self.voltage, 2)
        self.current = self.C * self.voltage
        self.total_current = 0
    
    def cache(self):
        super().cache()
        self.total_current = self.C * np.sum(self.history)

class TTResistor(Resistor):
    def __init__(self, travel_time, source, drain):
        C = 1 / travel_time
        super().__init__(C, source, drain)

class TransferResistor(Resistor):
    def __init__(self, freq, source, drain):
        C = freq * 2
        super().__init__(C, source, drain)

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