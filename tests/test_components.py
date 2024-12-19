import pytest
import cvxpy as cp
from transit_circuits.components import Resistor, Diode, CurrentSource  # Assuming the code is saved in `components.py`

def test_resistor_energy():
    source = cp.Variable()
    drain = cp.Variable()
    R = 2
    C = R ** -1
    resistor = Resistor(C, source, drain)

    # Assert energy symbolic equivalence
    expected_energy = 0.5 * C * cp.power(source - drain, 2)
    assert str(resistor.energy) == str(expected_energy)

def test_diode_constraint():
    source = cp.Variable()
    drain = cp.Variable()
    diode = Diode(source, drain)

    # Assert constraint symbolic equivalence
    expected_constraint = source - drain <= 0
    assert str(diode.constraint) == str(expected_constraint)

def test_current_source_energy():
    source = cp.Variable()
    drain = cp.Variable()
    i = 3
    current_source = CurrentSource(i, source, drain)

    # Assert energy symbolic equivalence
    expected_energy = i * (drain - source)
    assert str(current_source.energy) == str(expected_energy)

def test_resistor_initialization():
    source = cp.Variable()
    drain = cp.Variable()
    R = 5
    C = R ** -1
    resistor = Resistor(C, source, drain)

    # Assert initialization
    assert str(resistor.source) == str(source)
    assert str(resistor.drain) == str(drain)
    assert resistor.C == C

def test_diode_initialization():
    source = cp.Variable()
    drain = cp.Variable()
    diode = Diode(source, drain)

    # Assert initialization
    assert str(diode.source) == str(source)
    assert str(diode.drain) == str(drain)

def test_current_source_initialization():
    source = cp.Variable()
    drain = cp.Variable()
    i = 1.5
    current_source = CurrentSource(i, source, drain)

    # Assert initialization
    assert str(current_source.source) == str(source)
    assert str(current_source.drain) == str(drain)
    assert current_source.I == i
