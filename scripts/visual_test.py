import os
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is on sys.path so `from objects...` works even when
# the script is executed from inside the `scripts/` directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from objects.node import Node
from objects.member import Member2D
from scripts.fillet import fillet_between_members


def create_and_plot():
    # Create nodes
    n1 = Node(0, 0, fixity="free")
    n2 = Node(5, 0, fixity="free")
    n3 = Node(5, 5, fixity="free")

    # Create members connecting the nodes
    # set different areas so visual width varies
    m1 = Member2D("truss", n1, n2, E=200e9, A=1.0, I=0.0)
    m2 = Member2D("truss", n2, n3, E=200e9, A=1.0, I=0.0)

    members = [m1, m2]
    nodes = [n1, n2, n3]

    # Apply bilateral fillet to both members near their shared node
    final_nodes, final_members, replacements = fillet_between_members(
        nodes, members, m1, m2, radius=0.2, n_segments=4
    )
    print("Final Nodes:")
    for n in final_nodes:
        print(f"  {n.name}: ({n.x}, {n.y})")
    print("Final Members:")
    for m in final_members:
        print(f"  {m.name}: from {m.node_start.name} to {m.node_end.name}, A={getattr(m, 'A', 'N/A')}")
    # Plot final members
    fig, ax = plt.subplots(figsize=(6, 6))
    areas = [float(getattr(m, "A", 1.0)) for m in final_members]
    minA = min(areas) if areas else 1.0
    maxA = max(areas) if areas else 1.0
    min_lw, max_lw = 0.8, 6.0

    for m in final_members:
        A = float(getattr(m, "A", 1.0))
        x = [m.node_start.x, m.node_end.x]
        y = [m.node_start.y, m.node_end.y]
        if maxA > minA:
            t = (A - minA) / (maxA - minA)
        else:
            t = 1.0
        t = t**0.5
        lw = min_lw + t * (max_lw - min_lw)
        ax.plot(x, y, "-o", linewidth=lw, markersize=6)

    # annotate final nodes
    for n in final_nodes:
        ax.annotate(n.name, (n.x, n.y), textcoords="offset points", xytext=(5, 5))

    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Visual test: bilateral fillet radius 0.2 m")

    out_path = "scripts/fillet_test_plot.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    create_and_plot()
