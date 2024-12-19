from transit_circuits.optimization import Problem
from transit_circuits.components import Resistor, Diode, CurrentSource

import cvxpy as cp
import numpy as np

def _make_resistors():
    R = [Resistor(1, 1, 0), Resistor(2, 1, 0), Resistor(3, 1, 0)]
    return R

def _make_diodes():
    D = [Diode(1, 0), Diode(0, 1), Diode(0, 2)]
    return D

def _make_current_sources():
    CS = [
        CurrentSource(1, 1, 0),
        CurrentSource(2, 1, 0),
        CurrentSource(3, 1, 0)
    ]
    return CS

def _make_components():
    return _make_resistors(), _make_diodes(), _make_current_sources()

def _make_simple_loop():
    I_CS = 10
    R = 2
    V_0 = cp.Variable()
    V_1 = cp.Variable()
    V_2 = cp.Variable()

    CS = CurrentSource(I_CS, V_2, V_0)
    D_1 = Diode(V_0, V_1)
    R_1 = Resistor(1/R, V_1, V_2)

    return V_0, V_1, V_2, CS, D_1, R_1



def test_add_resistor():
    R = _make_resistors()
    p = Problem()
    p.add_resistor(R[0])
    assert p.objective.value == R[0].energy.value

def test_add_resistors():
    R = _make_resistors()
    p = Problem()
    p.add_resistor(*R[:2])
    expected_energy = R[0].energy.value + R[1].energy.value
    assert p.objective.value == expected_energy

def test_add_diode():
    D = _make_diodes()
    p = Problem()
    p.add_diode(D[0])
    assert D[0].constraint in p.constraints
    assert len(p.constraints) == 1

def test_add_diodes():
    D = _make_diodes()
    p = Problem()
    p.add_diode(*D[:2])  # Add the first two diodes
    assert D[0].constraint in p.constraints
    assert D[1].constraint in p.constraints
    assert len(p.constraints) == 2

def test_add_current_source():
    CS = _make_current_sources()
    p = Problem()
    p.add_current_source(CS[0])
    assert p.objective == CS[0].energy

def test_add_current_sources():
    CS = _make_current_sources()
    p = Problem()
    p.add_current_source(*CS[:2])  # Add the first two current sources
    expected_energy = CS[0].energy + CS[1].energy
    assert p.objective == expected_energy

def test_add_resistor_diode_current_source():
    R, D, CS = _make_components()
    p = Problem()
    p.add_resistor(R[0])
    p.add_diode(D[0])
    p.add_current_source(CS[0])
    assert p.objective.value == R[0].energy.value + CS[0].energy
    assert D[0].constraint in p.constraints
    assert len(p.constraints) == 1

def test_add_resistors_diodes_current_sources():
    R, D, CS = _make_components()
    p = Problem()
    p.add_resistor(*R)
    p.add_diode(*D)
    p.add_current_source(*CS)
    expected_energy = sum(r.energy.value for r in R) + sum(cs.energy for cs in CS)
    assert p.objective.value == expected_energy
    assert all(d.constraint in p.constraints for d in D)
    assert len(p.constraints) == 3

def test_simple_loop():
    V_0, V_1, V_2, CS, D_1, R_1 = _make_simple_loop()

    p = Problem()
    p.add_current_source(CS)
    p.add_diode(D_1)
    p.add_resistor(R_1)

    p.solve()

    assert V_0.value - V_2.value == 20

def test_parallel_opposing_diode():
    i_CS = 10
    r_1 = 2
    r_2 = 0.5
    V_0 = cp.Variable()
    V_1 = cp.Variable()
    V_2 = cp.Variable()
    V_3 = cp.Variable()

    CS = CurrentSource(i_CS, V_3, V_0)
    D_1 = Diode(V_0, V_1)
    D_2 = Diode(V_2, V_0)
    R_1 = Resistor(1/r_1, V_1, V_3)
    R_2 = Resistor(1/r_2, V_2, V_3)

    p = Problem()
    p.add_current_source(CS)
    p.add_diode(D_1)
    p.add_diode(D_2)
    p.add_resistor(R_1)
    p.add_resistor(R_2)
    p._add_constraint(V_2==0)

    p.solve()
    
    assert np.isclose(V_0.value - V_3.value, 20, rtol=0, atol=1e-10)

def test_parallel():
    i_CS = 10
    c_1 = 2
    c_2 = 3
    V_0 = cp.Variable()
    V_1 = cp.Variable()

    CS = CurrentSource(i_CS, V_1, V_0)
    R_1 = Resistor(c_1, V_0, V_1)
    R_2 = Resistor(c_2, V_0, V_1)

    p = Problem()
    p.add_current_source(CS)
    p.add_resistor(R_1)
    p.add_resistor(R_2)
    p._add_constraint(V_1==0)

    p.solve()
    
    assert np.isclose(V_0.value - V_1.value, 2, rtol=0, atol=1e-10)
