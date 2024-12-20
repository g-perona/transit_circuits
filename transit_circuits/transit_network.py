from transit_circuits.components import TTResistor, TransferResistor, Diode, CurrentSource
from transit_circuits.optimization import Problem

import matplotlib.pyplot as plt
import networkx as nx
import cvxpy as cp
import numpy as np

from tqdm import tqdm

class Line():
    def __init__(self, id, stations=[], avg_speed=10, frequency=None):

        self.id = id
        self.stations = stations
        self.avg_speed = avg_speed

        if frequency is None:
            self.frequency = cp.Variable()
        else:
            self.frequency = frequency

class _LineSegment():
    def __init__(self, line, travel_time):
        self.v_station = cp.Variable()
        self.v_diode = cp.Variable()

        self.td_diode = Diode(self.v_station, self.v_diode)
        
        self.line = line
        self.travel_time = travel_time

        self.tt_resistor = None
    
    def _make_resistor(self, v_next):
        self.v_next = v_next
        self.tt_resistor = TTResistor(self.travel_time, self.v_diode, self.v_next)

class Station():
    def __init__(self, id, x=None, y=None):
        self.lines={}
        self.id = id
        self._transfer_diodes = {}
        self._transfer_resistors = {}
        self.x = x
        self.y = y

    def _add_transfer_components(self, l1):
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
                    self._transfer_resistors[l1][d_l1][l2][d_l2] = TransferResistor(l2.frequency, v_diode1, self.lines[l2][d_l2].v_diode)
                    self._transfer_resistors[l2][d_l2][l1][d_l1] = TransferResistor(l1.frequency, v_diode2, self.lines[l1][d_l1].v_diode)

    
    def add_line(self, line, seg_next, seg_prev):
        self.lines[line] = {
            +1: seg_next,
            -1: seg_prev
        }
        
        self._add_transfer_components(line)

    def get_transfer_diodes(self, line1, line2):
        return [
            self._transfer_diodes[line1][+1][line2][+1],
            self._transfer_diodes[line1][+1][line2][-1],
            self._transfer_diodes[line1][-1][line2][+1],
            self._transfer_diodes[line1][-1][line2][-1]
        ]
    
    def get_transfer_resistors(self, line1, line2):
        return [
            self._transfer_resistors[line1][+1][line2][+1],
            self._transfer_resistors[line1][+1][line2][-1],
            self._transfer_resistors[line1][-1][line2][+1],
            self._transfer_resistors[line1][-1][line2][-1]
        ]

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
            self._origin_resistors.append(TransferResistor(line.frequency, self._v_origin, self.origin.lines[line][+1].v_diode))
            self._origin_resistors.append(TransferResistor(line.frequency, self._v_origin, self.origin.lines[line][-1].v_diode))

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

        for line in self.lines:
            for i, station in enumerate(line.stations):
                if i+1 < len(line.stations):
                    next_station = line.stations[i+1]
                    D_next = D[station.id, next_station.id]
                    tt_next = D_next / line.avg_speed
                else:
                    tt_next = np.inf
                if i > 0:
                    prev_station = line.stations[i-1]
                    D_prev =  D[station.id, prev_station.id]
                    tt_prev = D_prev / line.avg_speed
                else:
                    tt_prev = np.inf

                seg_next = _LineSegment(line, tt_next)
                seg_prev = _LineSegment(line, tt_prev)
                
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


    def calculate_flows(self, OD_trips:np.array, origins = None, destinations = None):
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

    def plot(self):
        """
        Plots a transit network.
        """
        # Create a NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes (stations)
        for station in self.stations:
            station_id = station.id
            G.add_node(station_id, pos=(station_id, 0))  # Example positioning; adjust as needed.
        
        # Add edges (connections between stations)
        for i in range(self.D.shape[0]):
            for j in range(i + 1, self.D.shape[1]):
                if self.D[i, j] > 0:  # Add an edge if there's a connection
                    G.add_edge(i, j, weight=self.D[i, j])
                    G.add_edge(j, i, weight=self.D[i, j])

        
        # Define colors for lines
        line_colors = plt.cm.tab10.colors  # Use a colormap for distinct colors
        line_color_map = {line.id: line_colors[i % len(line_colors)] for i, line in enumerate(self.lines)}
        
        # Plot the graph
        pos = nx.spring_layout(G, seed=42)  # Use spring layout for positioning
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color="black")
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="white")
        
        # Draw lines
        for line in self.lines:
            edges = []
            for i in range(len(line.stations) - 1):
                station1 = line.stations[i].id
                station2 = line.stations[i + 1].id
                edges.append((station1, station2))
                edges.append((station2, station1))
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, edge_color=line_color_map[line.id], label=f"Line {line.id}")
        
        # Create legend
        legend_elements = [plt.Line2D([0], [0], color=line_color_map[line.id], lw=2, label=f"Line {line.id}") for line in self.lines]
        plt.legend(handles=legend_elements, loc="best")
        
        # Display plot
        plt.title("Transit Network")
        plt.axis("off")
        plt.show()
    
    def plot_frequency(self, ax=None):
        """
        Plots the transit network with line frequencies visualized as edge thickness.
        Line colors correspond to their respective lines, and a legend explains the width scaling.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))

        G = nx.Graph()  # Use an undirected graph

        # Assign distinct colors to lines
        line_colors = plt.cm.tab10.colors  # Use colormap (tab10 has 10 distinct colors)
        line_color_map = {line.id: line_colors[i % len(line_colors)] for i, line in enumerate(self.lines)}

        # Add nodes with positions
        pos = {}
        for station in self.stations:
            G.add_node(station.id)
            pos[station.id] = (station.x, station.y) if station.x is not None and station.y is not None else (station.id, 0)

        # Add edges with frequencies and colors
        edge_frequencies = {}
        edge_colors = {}
        for line in self.lines:
            for i in range(len(line.stations) - 1):
                s1 = line.stations[i].id
                s2 = line.stations[i + 1].id

                # Use frequency of the line for thickness
                frequency = line.frequency if isinstance(line.frequency, (int, float)) else line.frequency.value
                edge_frequencies[(s1, s2)] = frequency
                edge_colors[(s1, s2)] = line_color_map[line.id]

                # Add edges to the graph
                G.add_edge(s1, s2)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color="black", ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="white", ax=ax)

        # Find max frequency to scale edge widths
        max_frequency = max(edge_frequencies.values()) if edge_frequencies else 1

        # Draw edges with thickness proportional to frequency and color-coded by line
        for u, v, _ in G.edges(data=True):
            frequency = edge_frequencies.get((u, v), 0)
            width = (frequency / max_frequency) * 10  # Scale to a max width of 10
            color = edge_colors.get((u, v), "grey")

            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], width=width, edge_color=color, ax=ax
            )

        # Create a legend for line widths and colors
        legend_elements = [
            plt.Line2D([0], [0], color=color, lw=2, label=f"Line {line.id}")
            for line in self.lines if hasattr(line, 'id') for color in [line_color_map[line.id]]
        ]
        legend_elements.append(
            plt.Line2D([0], [0], color="black", lw=2, label=f"Max Width = {max_frequency}")
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_title("Transit Network Frequencies")
        plt.axis("off")
        plt.show()

    def plot_od_demand(self, OD_trips, ax=None):
        """
        Visualizes the origin-destination demand matrix as directed edges.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))

        G = nx.DiGraph()
        pos = {station.id: (station.x, station.y) if station.x is not None and station.y is not None else (station.id, 0) for station in self.stations}
        G.add_nodes_from(pos.keys())

        # Add edges based on OD trips
        max_demand = max(max(trips.values()) for trips in OD_trips.values())
        for origin, destinations in OD_trips.items():
            for destination, demand in destinations.items():
                if demand > 0:
                    G.add_edge(origin.id, destination.id, weight=demand)

        # Scale edge widths by demand
        for u, v, data in G.edges(data=True):
            demand = data['weight']
            width = (demand / max_demand) * 10
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, edge_color="red", ax=ax)

        nx.draw_networkx_nodes(G, pos, node_size=300, node_color="black", ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="white", ax=ax)

        ax.set_title("Origin-Destination Demand")
        plt.axis("off")
        plt.show()

    def plot_flow(self, ax=None, label_fraction=0.5, round=0):
        """
        Plots the transit network with flow values visualized as edge widths and edges colored by lines.
        Station positions use x and y coordinates if available, edges are curved, and labels are offset to the right.
        """
        # Create a figure and axis if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))

        # NetworkX graph
        G = nx.DiGraph()

        # Assign distinct colors to lines
        line_colors = plt.cm.tab10.colors  # Use colormap (tab10 has 10 distinct colors)
        line_color_map = {line.id: line_colors[i % len(line_colors)] for i, line in enumerate(self.lines)}

        # Add nodes (stations)
        pos = {}
        for station in self.stations:
            G.add_node(station.id)
            # Use (x, y) coordinates if available; otherwise, placeholder
            if station.x is not None and station.y is not None:
                pos[station.id] = (station.x, station.y)
            else:
                pos[station.id] = (station.id, 0)

        # Add edges with flow and line color
        edge_flows = {}
        edge_colors = {}
        for line in self.lines:
            for i, station in enumerate(line.stations[:-1]):
                s1 = station.id
                s2 = line.stations[i + 1].id

                # Fetch the current (flow) for the resistor
                current_forward = station.lines[line][+1].tt_resistor.total_current
                current_reverse = line.stations[i + 1].lines[line][-1].tt_resistor.total_current

                # Add forward and reverse edges with flow
                G.add_edge(s1, s2, weight=current_forward)
                G.add_edge(s2, s1, weight=current_reverse)

                # Store flows and colors
                edge_flows[(s1, s2)] = current_forward
                edge_flows[(s2, s1)] = current_reverse
                edge_colors[(s1, s2)] = line_color_map[line.id]
                edge_colors[(s2, s1)] = line_color_map[line.id]

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color="black", ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="white", ax=ax)

        # Find the maximum flow to scale edge widths
        max_flow = max(edge_flows.values()) if edge_flows else 1  # Avoid division by zero

        # Draw curved edges with widths proportional to flows and colored by lines
        for edge in G.edges(data=True):
            u, v, _ = edge
            flow = edge_flows.get((u, v), 0)
            width = np.round((flow / max_flow) * 7)  # Scale the flow to a max width of 10
            color = edge_colors.get((u, v), "grey")
            # Use a curved connection style for bidirectional edges
            if G.has_edge(v, u):  # Check for reverse edge
                connectionstyle = "arc3,rad=0.2"  # Arch with curvature radius 0.2
            else:
                connectionstyle = "arc3,rad=0"  # Straight line
            if width > 0:
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)], width=width, edge_color=color, ax=ax,
                    connectionstyle=connectionstyle,
                    arrowstyle="simple"  # Fancy arrows
                )
        # Offset edge labels to the right of each edge
        # Position edge labels closer to the start of the edge
        edge_labels = {(u, v): f"{flow:.{round}f}" for (u, v), flow in edge_flows.items() if flow > 1e-4}
        for (u, v), label in edge_labels.items():
            # Calculate edge vector
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            dx, dy = x2 - x1, y2 - y1

            # Scale the vector to be closer to the source node
            label_pos_x = x1 + label_fraction * dx
            label_pos_y = y1 + label_fraction * dy

            # Apply a perpendicular offset for clarity
            length = (dx**2 + dy**2)**0.5
            offset_x = dy / length * 0.1  # Perpendicular offset (scaled)
            offset_y = -dx / length * 0.1  # Perpendicular offset (scaled)

            # Final label position
            label_pos = (label_pos_x + offset_x, label_pos_y + offset_y)
            ax.text(label_pos[0], label_pos[1], label, fontsize=12, ha="center", va="center")

        # Add legend for line colors
        legend_elements = [
            plt.Line2D([0], [0], color=color, lw=2, label=f"Line {line.id}")
            for line in self.lines if hasattr(line, 'id') for color in [line_color_map[line.id]]
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        # Title and cleanup
        ax.set_title("Transit Network Flows")
        plt.axis("off")
        plt.show()