import tkinter as tk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def havel_hakimi(degrees):
    """
    Determine whether the given degree sequence is graphical based on the Havel-Hakimi theorem.
    :param degrees: The specified degree sequence, passed as an array of integers. The sequence can be unordered.
    :return: True if the degree sequence is graphical, False otherwise.
    """
    # 1. Sort the degree sequence in descending order
    degrees.sort(reverse=True)
    # 2. Process the degree sequence in a loop until the algorithm termination condition is met
    while degrees:
        # 2.1. If the first element of the degree sequence is 0, all remaining elements must also be 0
        if degrees[0] == 0:
            return True
        # 2.2. Remove and store the first element of the degree sequence
        d = degrees[0]
        degrees = degrees[1:]
        # 2.3. Check if the length of the degree sequence is less than the value of the first element, it is not a graphical sequence
        if len(degrees) < d:
            return False
        # 2.4. Subtract 1 from the first d elements of the remaining degree sequence
        degrees[:d] = [x - 1 for x in degrees[:d]]
        degrees = [x for x in degrees if x > 0]  # Remove degrees at the end that are less than or equal to zero
        # 2.5. Sort the degree sequence
        degrees.sort(reverse=True)
    # 3. All other cases return a graphical sequence
    return True

def rest_degree_sequence(degrees_total, degrees_used):
    """
    Given the total degree sequence and the degrees already used, calculate the remaining degree sequence
    of the nodes, sort the node indices in descending order according to the remaining degree sequence, and return them.
    :param degrees_total: The total degree sequence of each node.
    :param degrees_used: The degrees already used for each node. Each time an edge is added, the degree used by the connected nodes increases by 1.
    :return: The node sequence sorted in descending order according to the remaining degree sequence.
    """
    degrees_rest = []
    for i in range(len(degrees_total)):
        degrees_rest.append(degrees_total[i] - degrees_used[i])
    indices = sorted(range(len(degrees_rest)), key=lambda idx: degrees_rest[idx], reverse=True)
    return indices

def draw_graph(degrees):
    """
    Draw a simple graph based on the given degree sequence.
    :param degrees: The specified degree sequence, passed as an array of integers. The sequence can be unordered.
    :return: If the degree sequence is graphical, return a networkx.Graph; otherwise, return None.
    """
    # 0. Check if the degree sequence is graphical
    if not havel_hakimi(degrees):
        print(f"The degree sequence {degrees} cannot be a valid graph")
        return None
    # 1. Create an empty graph
    graph = nx.Graph()
    # 2. Add nodes and set their degrees
    for i, deg in enumerate(degrees):
        graph.add_node(i, degree=deg)
    # 3. Traverse the nodes and create edges based on the degrees (the core algorithm)
    for node_idx in graph.nodes:
        # 3.1. Sort the remaining degrees and prioritize the node with the highest remaining degree for edge creation
        node_deg = graph.nodes[node_idx]['degree']
        degrees_order = rest_degree_sequence(degrees, graph.degree)
        # 3.2. Iterate over other nodes and try to create edges with the current node
        for other_node_idx in degrees_order:
            other_node_deg = graph.nodes[other_node_idx]['degree']
            # 3.2.1. If the degree of the current node meets the requirements, terminate
            if graph.degree[node_idx] == node_deg:
                break
            # 3.2.2. If the current other_node is the same as node_idx or an edge between them already exists, skip
            if node_idx == other_node_idx or graph.has_edge(node_idx, other_node_idx):
                continue
            # 3.2.3. If the current other_node still has remaining degrees, create an edge between other_node and node_idx
            if graph.degree[other_node_idx] < other_node_deg:
                graph.add_edge(node_idx, other_node_idx)
    # 4. Return the created graph
    return graph

def print_result(param, tool):
    """
    Print the program's result to the GUI component.
    :param param: The parameter passed from the GUI, which is the degree sequence.
    :param tool: The GUI component that includes the root and the display widget used to show error messages or the generated graph.
    :return: None
    """
    degrees = [int(i) for i in param.split()]
    graph = draw_graph(degrees)
    if tool.widget is not None:
        tool.widget.destroy()
    # 1. Graphical sequence check
    if graph is None:
        tool.widget = tk.Label(tool.root, text=f"The degree sequence {degrees} is not a graphical sequence")
        tool.widget.pack()
        return None
    # 2. Graph drawing
    # 2.1. Create a Matplotlib figure
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.axis('off')
    # 2.2. Draw the NetworkX graph onto the Matplotlib figure
    pos = nx.spring_layout(graph)  # Choose the layout algorithm
    nx.draw_networkx(graph, pos, ax=ax, with_labels=True, node_color='lightblue', edge_color='gray')
    # 2.3. Embed the Matplotlib figure into the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=tool.root)
    canvas.draw()
    tool.widget = canvas.get_tk_widget()
    tool.widget.pack()

class TkTool:
    def __init__(self, tk_root, tk_widget):
        self.root = tk_root
        self.widget = tk_widget

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Degree Sequence Graph Check/Visualization")

    # Create label and entry widget
    label1 = tk.Label(root, text="Enter the degree sequence (space-separated integers, e.g., 2 2 2 2) => ")
    label1.pack()
    entry1 = tk.Entry(root)
    entry1.pack()

    # Create label and button
    tk_tool = TkTool(root, None)
    button = tk.Button(root, text="Draw Graph", command=lambda: print_result(entry1.get(), tk_tool))
    button.pack()

    # Start the main loop
    root.mainloop()

