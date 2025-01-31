import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import schemdraw
import schemdraw.elements as elm
import itertools

from transit_circuits.transit_network import Line, Station

class TransitNetworkPlotter:
    LINE_STYLES = None
    ARROW_STYLE=None
    LABEL_EDGES=False
    ROUND_VAL=0
    RAD=0
    MAX_WIDTH=10
    LABEL_FONTSIZE=18
    LABEL_FRACTION=0.3
    NODE_SIZE=400
    NODE_FONT_SIZE=15
    ADD_LEGEND=False

    def __init__(self, transit_network):
        """
        Initializes the TransitNetworkPlotter class.

        Parameters:
        - transit_network: TransitNetwork object containing lines and stations data.
        """
        self.transit_network = transit_network
        self.lines = self.transit_network.lines
        self.stations = self.transit_network.stations

    def _prep_plot(ax):
        """
        Prepares the plot by creating a new matplotlib axis if none is provided.

        Parameters:
        - ax: matplotlib axis or None

        Returns:
        - ax: matplotlib axis instance
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
            return ax
        else:
            return ax

    def _prep_graph(self, directed=None):
        """
        Prepares a NetworkX graph representation of the transit network.

        Parameters:
        - directed (bool): Whether the graph should be directed.

        Returns:
        - G: NetworkX Graph or DiGraph
        - line_color_map: Dictionary mapping line IDs to colors
        - pos: Dictionary mapping station IDs to coordinates
        """
        G = nx.DiGraph() if directed else nx.Graph()

        # Assign distinct colors to lines
        line_colors = plt.cm.tab10.colors  # Use colormap (tab10 has 10 distinct colors)
        line_color_map = {line.id: line_colors[i % len(line_colors)] for i, line in enumerate(self.lines)}

        # Add nodes (stations)
        pos = {}
        for station in self.stations:
            G.add_node(station.id)
            pos[station.id] = (station.x, station.y) if station.x is not None and station.y is not None else (station.id, 0)

        return G, line_color_map, pos

    def _plot_network(self, ax, edge_data, directed, line_styles=LINE_STYLES, arrowstyle=ARROW_STYLE, label_edges=LABEL_EDGES, 
                      round_val=ROUND_VAL, rad=RAD, max_width=MAX_WIDTH, label_fontsize=LABEL_FONTSIZE, label_fraction=LABEL_FRACTION, 
                      node_size=NODE_SIZE, node_font_size=NODE_FONT_SIZE, add_legend=ADD_LEGEND):
        """
        General function to plot the transit network with customizable parameters.

        Parameters:
        - ax: matplotlib axis for plotting
        - edge_data: Function to extract edge data (e.g., flow or frequency)
        - directed (bool): Whether the graph should be directed
        - line_styles (dict): Custom styles for different lines
        - arrowstyle (str): Style of arrows for directed edges
        - label_edges (bool): Whether to display edge labels
        - round_val (int): Decimal places for edge labels
        - rad (float): Radius of curved edges
        - max_width (int): Maximum edge width for scaling
        - label_fontsize (int): Font size for edge labels
        - label_fraction (float): Fraction of the way along the edge to place labels
        - node_size (int): Size of station nodes
        - node_font_size (int): Font size for station labels
        """
        ax = TransitNetworkPlotter._prep_plot(ax)
        line_styles = line_styles or {}

        # Prepare the graph and layout
        G, line_color_map, pos = self._prep_graph(directed=directed)
        edge_attributes = {}
        
        # Process edge data (flows or frequencies)
        max_value = float('-inf')
        for line in self.lines:
            for i, station in enumerate(line.stations[:-1]):
                station_1 = station
                station_2 = line.stations[i + 1]
                s1, s2 = station_1.id, station_2.id
                value_1 = edge_data(line, station_1, +1)
                value_2 = edge_data(line, station_2, -1)
                color = line_color_map.get(line.id, "gray")
                style = line_styles.get(line.id, "solid")

                # Store edge attributes
                G.add_edge(s1, s2, weight=value_1)
                G.add_edge(s2, s1, weight=value_2)
                edge_attributes[(s1, s2)] = (value_1, color, style)
                edge_attributes[(s2, s1)] = (value_2, color, style)

                max_value = max(max_value, value_1, value_2)

        # Ensure max_value is not zero to prevent division errors
        max_value = max(max_value, 1e-4)

        # Draw nodes and labels with customizable sizes
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="black", ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=node_font_size, font_color="white", ax=ax)

        # Draw edges with scaled widths and colors
        for (u, v), (value, color, style) in edge_attributes.items():
            width = (value / max_value) * max_width
            connectionstyle = f"arc3,rad={rad}" if G.has_edge(v, u) else "arc3,rad=0"
            
            if arrowstyle:
                arrowstyle+=f",head_length=2,head_width=2"

            if value > 1e-4:
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, edge_color=color, 
                                       style=style, ax=ax, connectionstyle=connectionstyle, 
                                       arrowstyle=arrowstyle)

            if label_edges and value > 1e-4:
                self._add_edge_label(ax, pos, u, v, value, round_val, label_fontsize, label_fraction)
        if add_legend:
            self._add_legend(ax, line_color_map)
        ax.axis("off")

        return ax

    def plot_flow(self, ax=None, label_fontsize=LABEL_FONTSIZE, label_fraction=LABEL_FRACTION, 
                  node_size=NODE_SIZE, node_font_size=NODE_FONT_SIZE, round=ROUND_VAL, add_legend=ADD_LEGEND):
        """
        Plots the transit network with flow values visualized as edge widths and edges colored by lines.

        Parameters:
        - ax: matplotlib axis to draw on
        - label_fontsize (int): Font size for edge labels
        - label_fraction (float): Position factor for label placement
        - node_size (int): Size of nodes (stations)
        - node_font_size (int): Font size for station labels
        - round (int): Number of decimal places for labels
        """
        def get_flow(line, station, direction):
            return station.lines[line][direction].tt_resistor.total_current

        ax = self._plot_network(ax=ax, directed=True, edge_data=get_flow, line_styles=None, 
                                arrowstyle='simple', label_edges=True, round_val=round, rad=0.2, 
                                label_fontsize=label_fontsize, label_fraction=label_fraction, 
                                node_size=node_size, node_font_size=node_font_size, add_legend=add_legend)

        ax.set_title("Transit Network Flows")

    def plot_flow_one_to_one(self, origin, destination, ax=None, label_fontsize=LABEL_FONTSIZE, 
                             label_fraction=LABEL_FRACTION, node_size=NODE_SIZE, node_font_size=NODE_FONT_SIZE,
                             round=ROUND_VAL, add_legend=ADD_LEGEND):
        
        def get_flow(line, station, direction):
            return self.transit_network._disaggregated_currents[origin][destination][station][line][direction]
        
        ax = self._plot_network(ax=ax, directed=True, edge_data=get_flow, line_styles=None, 
                                arrowstyle='simple', label_edges=True, round_val=round, rad=0.2, 
                                label_fontsize=label_fontsize, label_fraction=label_fraction,
                                node_size=node_size, node_font_size=node_font_size, add_legend=add_legend)
        
        ax.set_title(fr"{origin.id} $\to$ {destination.id}")

    def plot_flow_one_to_all(self, origin, ax=None, label_fontsize=LABEL_FONTSIZE, 
                             label_fraction=LABEL_FRACTION, node_size=NODE_SIZE, node_font_size=NODE_FONT_SIZE,
                             round=ROUND_VAL, add_legend=ADD_LEGEND):
        
        def get_flow(line:Line, station:Station, direction):
            return np.sum([
                self.transit_network._disaggregated_currents[origin][d][station][line][direction]
                for d in self.transit_network._disaggregated_currents[origin].keys()
            ])
        
        ax = self._plot_network(ax=ax, directed=True, edge_data=get_flow, line_styles=None, 
                                arrowstyle='simple', label_edges=True, round_val=round, rad=0.2, 
                                label_fontsize=label_fontsize, label_fraction=label_fraction,
                                node_size=node_size, node_font_size=node_font_size, add_legend=add_legend)
        
        ax.set_title(fr"{origin.id} $\to$ all")

    def plot_flow_all_to_one(self, destination, ax=None, label_fontsize=LABEL_FONTSIZE, 
                             label_fraction=LABEL_FRACTION, node_size=NODE_SIZE, node_font_size=NODE_FONT_SIZE,
                             round=ROUND_VAL, add_legend=ADD_LEGEND):
        
        def get_flow(line:Line, station:Station, direction):
            return np.sum([
                self.transit_network._disaggregated_currents[o][destination][station][line][direction]
                for o in self.transit_network._disaggregated_currents.keys()
            ])
        
        ax = self._plot_network(ax=ax, directed=True, edge_data=get_flow, line_styles=None, 
                                arrowstyle='simple', label_edges=True, round_val=round, rad=0.2, 
                                label_fontsize=label_fontsize, label_fraction=label_fraction,
                                node_size=node_size, node_font_size=node_font_size, add_legend=add_legend)
        
        ax.set_title(fr"all $\to$ {destination.id}")

    def plot_frequency(self, ax=None, line_styles={}, round=ROUND_VAL, max_width=MAX_WIDTH, label_fontsize=LABEL_FONTSIZE, 
                       label_fraction=LABEL_FRACTION, node_size=NODE_SIZE, node_font_size=NODE_FONT_SIZE, add_legend=ADD_LEGEND):
        """
        Plots the transit network with line frequencies visualized as edge thickness.

        Parameters:
        - ax: matplotlib axis to draw on
        - line_styles (dict): Custom styles for different lines
        - round (int): Number of decimal places to round labels
        - label_fontsize (int): Font size for edge labels
        - label_fraction (float): Position factor for label placement
        - node_size (int): Size of nodes (stations)
        - node_font_size (int): Font size for station labels
        """
        def get_freq(line:Line, station:Station, direction):
            return line.frequency_vpm

        ax = self._plot_network(ax=ax, directed=False, edge_data=get_freq, line_styles=line_styles, 
                           label_edges=False, round_val=round, rad=0, max_width=max_width, label_fontsize=label_fontsize, 
                           label_fraction=label_fraction, node_size=node_size, node_font_size=node_font_size)

        ax.set_title("Transit Network Frequencies")

    def _add_edge_label(self, ax, pos, u, v, value, round_val, label_fontsize, label_fraction):
        """
        Adds edge labels to the plot.

        Parameters:
        - ax: matplotlib axis
        - pos: Dictionary of node positions
        - u, v: Nodes defining the edge
        - value: Value to label
        - round_val: Number of decimal places
        - label_fontsize: Font size of the label
        - label_fraction: Fraction to position the label along the edge
        """
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        dx, dy = x2 - x1, y2 - y1
        length = (dx**2 + dy**2)**0.5

        label_pos_x = x1 + 0.5 * dx
        label_pos_y = y1 + label_fraction * dy

        offset_x = dy / length * 0.1 
        offset_y = -dx / length * 0.1  
        
        ax.text(label_pos_x + offset_x, label_pos_y + offset_y, f"{value:.{round_val}f}", fontsize=label_fontsize, ha="center", va="center")

    def _add_legend(self, ax, line_color_map):
        """
        Adds a legend to the plot showing the color representation of transit lines.

        Parameters:
        - ax: matplotlib axis to which the legend should be added.
        - line_color_map (dict): Dictionary mapping line IDs to their respective colors.

        The legend will display each line with its assigned color for better visualization.
        """
        legend_elements = [
            plt.Line2D([0], [0], color=color, lw=2, label=f"Line {line_id}")
            for line_id, color in line_color_map.items()
        ]
        ax.legend(handles=legend_elements, loc="best")
    
    def plot_station_circuit(station):
        """
        Generates a structured circuit diagram using SchemDraw.
        - Lines are drawn in parallel (vertically stacked).
        - Transfer components are placed with horizontal space.
        - Each line extends until all its transfer components are included.

        Parameters:
        - station: A Station object from the transit network.
        """

        d = schemdraw.Drawing()

        # Dictionary to store component references
        line_positions = {}  # Tracks where each line's path is in the diagram
        transfer_positions = {}  # Tracks where transfers are placed

        # Assign vertical positions for each line
        line_spacing = 2  # Vertical spacing between lines
        base_y = 0
        lines = station.lines

        for i, line in enumerate(lines):
            line_positions[line] = base_y - (i * line_spacing)

        # Step 1: Draw each line as a separate horizontal branch
        line_segments = {}
        for line, segments in lines.items():
            seg_next = segments[+1]
            seg_prev = segments[-1]

            y_pos = line_positions[line]

            # Start the line at the station
            d += elm.Dot().at((0, y_pos)).label(f"Station {station.id} (Line {line})", loc="left")

            # Travel direction diodes
            d += (diode1 := elm.Diode().right().label(f'Diode {line}-1'))
            d += (diode2 := elm.Diode().right().label(f'Diode {line}-2'))

            # Travel time resistors
            if seg_next.tt_resistor:
                d += (r1 := elm.Resistor().right().label(f'R(TT={seg_next.travel_time}s)'))
            if seg_prev.tt_resistor:
                d += (r2 := elm.Resistor().right().label(f'R(TT={seg_prev.travel_time}s)'))

            # Store segment positions
            line_segments[line] = [diode1, r1, diode2, r2]

        # Step 2: Draw transfer components (between pairs of lines)
        transfer_spacing = 3  # Horizontal space for transfers
        for (line1, line2) in itertools.combinations(lines, 2):
            transfers_12 = station._transfer_resistors.get(line1, {}).get(+1, {}).get(line2, {})

            for direction1 in (-1, +1):
                for direction2 in (-1, +1):
                    if (line1, line2, direction1, direction2) not in transfer_positions:
                        x_offset = len(transfer_positions) * transfer_spacing

                        resistor = transfers_12.get(direction2)
                        diode = station._transfer_diodes[line1][direction1][line2][direction2]

                        y1, y2 = line_positions[line1], line_positions[line2]

                        # Transfer resistor
                        d += elm.Resistor().at((x_offset, y1)).down(abs(y2 - y1)).label(f'R({line1}->{line2})')

                        # Transfer diode
                        d += elm.Diode().at((x_offset, y2)).right().label(f'Diode {line1}->{line2}')

                        transfer_positions[(line1, line2, direction1, direction2)] = (x_offset, y1, y2)

        # Step 3: End each line after all transfers
        for line in lines:
            end_x = (len(transfer_positions) + 1) * transfer_spacing
            d += elm.Dot().at((end_x, line_positions[line])).label(f"End of Line {line}", loc="right")

        # Draw the circuit
        d.draw()
