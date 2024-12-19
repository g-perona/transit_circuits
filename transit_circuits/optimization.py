from transit_circuits.components import Resistor, Diode, CurrentSource
import cvxpy as cp

class Problem():

    def __init__(self):
        self.objective = 0
        self.objective_terms = []
        self.constraints = []
        
        self.resistors = []
        self.diodes = []
        self.current_sources = []
    
    def _add_objective_term(self, obj_term):
        self.objective_terms.append(obj_term)
        self.objective += obj_term

    def _add_constraint(self, const):
        self.constraints.append(const)
    
    def add_resistor(self, *R:Resistor):
        for r in R:
            self._add_objective_term(r.energy)
            if r.is_variable:
                self._add_constraint(r.constraint)
            self.resistors.append(r)

    def add_diode(self, *D:Diode):
        for d in D:
            self._add_constraint(d.constraint)
            self.diodes.append(d)

    def add_current_source(self, *CS:CurrentSource):
        for cs in CS:
            self._add_objective_term(cs.energy)
            self.current_sources.append(cs)
    
    def _cache_component_voltages(self):
        [r.cache() for r in self.resistors]
        [d.cache() for d in self.diodes]
        [c.cache() for c in self.diodes]
    
    def solve(self, solver=None):
        self.problem = cp.Problem(cp.Minimize(self.objective), self.constraints)
        rv = self.problem.solve(solver)
        self._cache_component_voltages()
        return rv
        