import matplotlib.pyplot as plt
import networkx as nx
from fractions import Fraction
from collections import deque
from decimal import Decimal, getcontext
import csv

getcontext().prec = 50

class Node:
    def __init__(self, num, den, depth=0, flipped=False):
        self.num, self.den = num, den
        self.val = Fraction(num, den)
        self.depth = depth
        self.flipped = flipped
        self.left = None
        self.right = None

    def __str__(self):
        return f"{self.num}/{self.den}"


def get_by_path(start_node, path, ln, ld, rn, rd, flip_state):
    curr_n, curr_d = start_node.num, start_node.den
    c_ln, c_ld, c_rn, c_rd = ln, ld, rn, rd
    d = start_node.depth

    for move in path:
        actual_move = move
        if flip_state:
            actual_move = 'R' if move == 'L' else 'L'

        if actual_move == 'L':
            c_rn, c_rd = curr_n, curr_d
            curr_n, curr_d = c_ln + curr_n, c_ld + curr_d
        else:
            c_ln, c_ld = curr_n, curr_d
            curr_n, curr_d = curr_n + c_rn, curr_d + c_rd
        d += 1
    return Node(curr_n, curr_d, d, flip_state), c_ln, c_ld, c_rn, c_rd


def generate_red_edges(parent_n, parent_d, parent_depth,
                       la_n, la_d, ra_n, ra_d,
                       direction, gen, max_gen):
    if gen > max_gen:
        return []

    flip_d = 'L' if direction == 'R' else 'R'
    cross  = 'LR' if direction == 'R' else 'RL'
    results = []

    for nav, child_dir in [('LL', direction), ('RR', direction), (cross, flip_d)]:
        parent_node = Node(parent_n, parent_d, parent_depth)

        # Navigate to new parent (2 steps)
        np_node, nla_n, nla_d, nra_n, nra_d = get_by_path(
            parent_node, nav, la_n, la_d, ra_n, ra_d, False)

        # Navigate to child of new parent (2 steps + 1)
        child_node, *_ = get_by_path(
            parent_node, nav + child_dir, la_n, la_d, ra_n, ra_d, False)

        results.append((str(Fraction(np_node.num, np_node.den)),
                        str(Fraction(child_node.num, child_node.den))))

        results += generate_red_edges(
            np_node.num, np_node.den, np_node.depth,
            nla_n, nla_d, nra_n, nra_d,
            child_dir, gen + 1, max_gen)

    return results


def save_red_edges(red_edges):
    
    with open('red_edges.csv', 'w') as target:
        wtr= csv.writer( target )
        for u, v in red_edges:
            wtr.writerow([u,v])
 



def edge_length(u, v):
    """Length of a red edge = |p/q - r/s| = 1/(q*s) for Farey neighbors."""
    pu, qu = (int(x) for x in u.split('/'))
    pv, qv = (int(x) for x in v.split('/'))
    return Decimal(1) / Decimal(qu * qv)


def sum_edge_lengths(red_edges, pos, max_depth=None):
    """Sum of lengths of red edges whose parent is at depth <= max_depth."""
    total = Decimal(0)
    for u, v in red_edges:
        if u not in pos:
            continue
        depth = round(-pos[u][1])  # pos y = -depth
        if max_depth is not None and depth > max_depth:
            continue
        total += edge_length(u, v)
    return total


def build_graph_and_special_edges(recursive_gen):
    max_depth = 2 * recursive_gen + 1

    # 1. Build the background skeleton tree
    root = Node(1, 2, 0)
    G = nx.DiGraph()

    # Queue stores: (node, left_num, left_den, right_num, right_den)
    queue = deque([(root, 0, 1, 1, 1)])

    while queue:
        curr, ln, ld, rn, rd = queue.popleft()
        u_lbl = str(curr.val)
        G.add_node(u_lbl, depth=curr.depth)

        if curr.depth < max_depth:
            curr.left = Node(ln + curr.num, ld + curr.den, curr.depth + 1)
            v_l_lbl = str(curr.left.val)
            G.add_edge(u_lbl, v_l_lbl, color='lightgray', weight=1)
            queue.append((curr.left, ln, ld, curr.num, curr.den))

            curr.right = Node(curr.num + rn, curr.den + rd, curr.depth + 1)
            v_r_lbl = str(curr.right.val)
            G.add_edge(u_lbl, v_r_lbl, color='lightgray', weight=1)
            queue.append((curr.right, curr.num, curr.den, rn, rd))

    # Compute positions via in-order traversal so nodes never overlap
    pos = {}
    counter = [0]

    def inorder(node):
        if node is None or node.depth > max_depth:
            return
        inorder(node.left)
        pos[str(node.val)] = (counter[0], -node.depth)
        counter[0] += 1
        inorder(node.right)

    inorder(root)

    # 2. Generate red edges
    # Initial edge: parent=1/2, la=0/1, ra=1/1, direction=R
    red_edges = [('1/2', '2/3')]
    red_edges += generate_red_edges(
        1, 2, 0,       # parent = 1/2 at depth 0
        0, 1, 1, 1,    # la = 0/1, ra = 1/1
        'R',           # initial direction
        1, recursive_gen)

    return G, pos, red_edges


# --- Run and Plot ---
recursive_gen = 8
G, pos, red_edges = build_graph_and_special_edges(recursive_gen)

save_red_edges(red_edges)

max_depth = 2 * recursive_gen + 1

s = sum_edge_lengths(red_edges, pos, max_depth)
print(f"  sum of lengths up to depth {max_depth}: {s}")

plt.figure(figsize=(15, 9))

# Draw Background Tree
#nx.draw_networkx_edges(G, pos, edge_color='#eeeeee', arrows=False)

# Draw Red Special Edges
#for u, v in red_edges:
#    if u in pos and v in pos:
#        plt.annotate("", xy=pos[v], xytext=pos[u],
#                     arrowprops=dict(arrowstyle="-", color="red", lw=2))

# Draw Nodes
#nx.draw_networkx_nodes(G, pos, node_size=600, node_color='white', edgecolors='black')
#nx.draw_networkx_labels(G, pos, font_size=7)

#plt.title("Stern-Brocot Tree with Red Edges")
#plt.axis('off')
#plt.tight_layout()
#plt.show()
