from transit_circuits.transit_network import Line, Station, TransitNetwork
from transit_circuits import utils

import numpy as np

tn = utils.make_grid()

OD_trips = {o: {d: np.random.rand() * 50 + 50 for d in tn.stations if d != o} for o in tn.stations}

OD_morning = {o: {d:0 for d in tn.stations} for o in tn.stations}
OD_evening = {o: {d:0 for d in tn.stations} for o in tn.stations}

for o in tn.stations:
    for d in tn.stations:
        if o == d:
            continue
        OD_morning[o][d] += 25
        OD_evening[o][d] += 25

        if o.id not in [3, 4, 7, 8] and d.id in [3, 4, 7, 8]:
            OD_morning[o][d] += 75
        if o.id in [3, 4, 7, 8] and d.id not in [3, 4, 7, 8]:
            OD_evening[o][d] += 75

for o in tn.stations:
    for d in tn.stations:
        assert OD_morning[o][d] == OD_evening[d][o]

tn.calculate_flows(OD_morning)

tn.plot_flow()

tn.calculate_flows(OD_evening)

tn.plot_flow()