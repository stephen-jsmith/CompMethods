"""Example runner for the fillet demo and utilities.

Run `python main.py --demo` to execute the visual test (if available).
"""

from __future__ import annotations

# Expose the fillet utilities at the top level for quick experimentation
from scripts import (
    apply_fillet,
    apply_fillet_radius,
    fillet_and_report,
    fillet_between_members,
    apply_fillets_at_nodes,
)
from scripts.preassembly import dof_index_map
from scripts.assembly import assemblyKff
import numpy as np
import numpy.linalg as linalg
import math

# Node/member classes (used to build example geometry if needed)
from objects.node import Node
from objects.member import Member2D

import scripts.visual_test as visual_test
import argparse
import sys
import os

# plotting (use Agg for headless environments)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_sample_structure():
    """Create the square+diagonal sample structure used in `scripts/assembly.py`.

    Returns (nodes_list, members_list)
    """
    # Define nodes (square: (0,0), (1,0), (1,1), (0,1))
    # Make n1 fully fixed (x,y,rz) and n2 pinned (x,y) so the beam frame is stable
    n1 = (
        Node(0, 0, "fixed", ["x", "y", "rz"])
        if hasattr(Node, "__call__")
        else Node(0, 0, "fixed", ["x", "y", "rz"])
    )
    n2 = (
        Node(1, 0, "pinned", ["x", "y"])
        if hasattr(Node, "__call__")
        else Node(1, 0, "pinned", ["x", "y"])
    )
    n3 = Node(1, 1, "free") if hasattr(Node, "__call__") else Node(1, 1, "free")
    n4 = Node(0, 1, "free") if hasattr(Node, "__call__") else Node(0, 1, "free")

    # Define members (edges of square + diagonal through center)
    members = [
        Member2D(
            "beam",
            n1,
            n2,
            E=200e9,
            A=0.01,
            I=8.3e-6,
        ),
        Member2D(
            "beam",
            n2,
            n3,
            E=200e9,
            A=0.01,
            I=8.3e-6,
        ),
        Member2D(
            "beam",
            n3,
            n4,
            E=200e9,
            A=0.01,
            I=8.3e-6,
        ),
        Member2D(
            "beam",
            n4,
            n1,
            E=200e9,
            A=0.01,
            I=8.3e-6,
        ),
        Member2D(
            "beam",
            n1,
            n3,
            E=200e9,
            A=0.01,
            I=8.3e-6,
        ),
    ]
    nodes = [n1, n2, n3, n4]
    return nodes, members


def plot_frame(nodes, members, out_path: str = "scripts/sample_frame_plot.png"):
    """Plot nodes and members and save to `out_path`."""
    fig, ax = plt.subplots(figsize=(6, 6))
    areas = [float(getattr(m, "A", 1.0)) for m in members]
    minA = min(areas) if areas else 1.0
    maxA = max(areas) if areas else 1.0
    min_lw, max_lw = 0.8, 6.0

    for m in members:
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

    for n in nodes:
        ax.annotate(n.name, (n.x, n.y), textcoords="offset points", xytext=(5, 5))

    ax.set_aspect("equal")
    ax.grid(True)
    # auto scale limits with padding
    xs = [float(n.x) for n in nodes]
    ys = [float(n.y) for n in nodes]
    if xs and ys:
        xmin, xmax = min(xs) - 0.5, max(xs) + 0.5
        ymin, ymax = min(ys) - 0.5, max(ys) + 0.5
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(os.path.basename(out_path))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run fillet demo or import utilities")
    parser.add_argument(
        "--demo", action="store_true", help="Run the visual fillet demo and save a plot"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Build sample nodes/members and run a fillet report",
    )
    parser.add_argument(
        "--fillet-both",
        action="store_true",
        help="Apply a fillet at both nodes of the bottom member and save plot",
    )
    parser.add_argument(
        "--fillet-corners",
        action="store_true",
        help="Apply fillets at all corner nodes and save plot",
    )
    args = parser.parse_args(argv)

    if args.demo:
        if hasattr(visual_test, "create_and_plot"):
            visual_test.create_and_plot()
            print("Demo finished (plot saved by visual_test).")
            return 0
        else:
            print(
                "visual_test.create_and_plot() not found; inspect scripts/visual_test.py"
            )
            return 2

    if args.example:
        nodes, members = create_sample_structure()
        print("Sample nodes:")
        for n in nodes:
            print(f"  {n.name}: ({n.x}, {n.y})")
        print("Sample members:")
        for m in members:
            a = getattr(m.node_start, "name", "?")
            b = getattr(m.node_end, "name", "?")
            print(f"  {m.name}: {a} -> {b} | A={getattr(m,'A',None)}")

        # Run a fillet report: fillet the bottom member at its right node (n2)
        m_bottom = members[0]
        m_right = members[1]
        # Use fillet_between_members which selects only segments on the target member
        new_nodes, new_members, replacements = fillet_between_members(
            nodes, members, m_bottom, m_right, radius=0.25, n_segments=4
        )
        # also save a plot of the frame after fillet
        plot_frame(new_nodes, new_members, out_path="scripts/sample_frame_plot.png")
        # Assemble global stiffness for the filleted frame and solve for displacements
        dof_map = dof_index_map(new_members)
        print("DOF map (Free):", dof_map["Free"])
        print("DOF map (Fixed):", dof_map["Fixed"])
        total_free_dofs = sum(len(v) for v in dof_map["Free"].values())
        Kff_sym, dof_indices = assemblyKff(
            new_members, dof_map, total_dofs=total_free_dofs
        )
        print("Assembled Kff (sym) shape:", Kff_sym.shape)

        # Convert sympy Matrix to numeric numpy array
        try:
            K_numeric = np.array(Kff_sym.evalf(), dtype=np.float64)
        except Exception:
            K_numeric = np.array(
                [[float(x) for x in row] for row in Kff_sym.tolist()], dtype=np.float64
            )

        # Build force vector: apply a unit load of 10 at the top-right node (Node_1_1) in the Y direction
        F = np.zeros((total_free_dofs,))
        target_node_name = "Node_1_1"
        if target_node_name in dof_indices and "y" in dof_indices[target_node_name]:
            idx = dof_indices[target_node_name]["y"]
            F[idx] = 10.0
        else:
            # pick any available DOF as fallback (first free DOF)
            if dof_indices:
                first_node = next(iter(dof_indices))
                first_dof = next(iter(dof_indices[first_node].values()))
                F[first_dof] = 10.0

        # Solve linear system
        try:
            disp = linalg.solve(K_numeric, F)
            print("Nodal displacements (free DOFs):")
            for node_name, dofs in dof_indices.items():
                for dof_name, idx in dofs.items():
                    print(f"  {node_name} DOF {dof_name}: {disp[idx]}")
            # Print displacements only at the original 4 corner nodes (x and y)
            corners = ["Node_0_0", "Node_1_0", "Node_1_1", "Node_0_1"]
            print("\nCorner displacements (u_x, u_y):")
            for cn in corners:
                ux = None
                uy = None
                if cn in dof_indices and isinstance(dof_indices[cn], dict):
                    ux_idx = dof_indices[cn].get("x")
                    uy_idx = dof_indices[cn].get("y")
                    if ux_idx is not None:
                        ux = float(disp[ux_idx])
                    if uy_idx is not None:
                        uy = float(disp[uy_idx])
                # If DOF not present, it's restrained (zero displacement)
                ux_str = f"{ux:.6e}" if ux is not None else "restrained"
                uy_str = f"{uy:.6e}" if uy is not None else "restrained"
                print(f"  {cn}: u_x = {ux_str}, u_y = {uy_str}")
        except Exception as e:
            print("Could not solve linear system:", e)
        print("Fillet replacements:")
        for orig, segs in replacements.items():
            print(f"  {orig} -> {len(segs)} segments")
            for s in segs:
                print(
                    f"    {s.name}: {s.node_start.name} -> {s.node_end.name} | A={getattr(s,'A',None)}"
                )
        return 0

    if args.fillet_both:
        # Build structure and fillet both ends of the bottom member
        nodes, members = create_sample_structure()
        m_bottom = members[0]
        # right neighbor
        m_right = members[1]
        # left neighbor (the member that shares n1 besides bottom)
        m_left = members[3]

        # First fillet at right end
        nodes_a, members_a, repl1 = fillet_between_members(
            nodes, members, m_bottom, m_right, radius=0.25, n_segments=4
        )

        # Find a segment in members_a that lies on the original bottom line and contains node n1
        n1 = nodes[0]
        # original bottom direction
        dx = float(m_bottom.node_end.x) - float(m_bottom.node_start.x)
        dy = float(m_bottom.node_end.y) - float(m_bottom.node_start.y)
        L = math.hypot(dx, dy) or 1.0

        def _on_line_point(x: float, y: float) -> bool:
            cross = (x - float(n1.x)) * dy - (y - float(n1.y)) * dx
            return abs(cross) <= 1e-6 * L

        target_seg = None
        for mm in members_a:
            # choose member that has n1 as one node and lies on the original line
            if mm.node_start.name == n1.name or mm.node_end.name == n1.name:
                sx = float(mm.node_start.x)
                sy = float(mm.node_start.y)
                ex = float(mm.node_end.x)
                ey = float(mm.node_end.y)
                if _on_line_point(sx, sy) and _on_line_point(ex, ey):
                    target_seg = mm
                    break

        if target_seg is None:
            print("Could not find target segment on bottom member for left fillet")
            return 2

        # Second fillet at left end using the found target segment and left neighbor
        nodes_b, members_b, repl2 = fillet_between_members(
            nodes_a, members_a, target_seg, m_left, radius=0.25, n_segments=4
        )

        # Save plot and print replacements
        plot_frame(nodes_b, members_b, out_path="scripts/sample_frame_fillet_both.png")
        print("Fillet both replacements (right):")
        for k, v in repl1.items():
            print(k, "->", len(v))
        print("Fillet both replacements (left):")
        for k, v in repl2.items():
            print(k, "->", len(v))

        return 0

    if args.fillet_corners:
        # Build structure and fillet all four corner nodes
        nodes, members = create_sample_structure()
        # corner node names for the square
        corners = ["Node_0_0", "Node_1_0", "Node_1_1", "Node_0_1"]
        # apply fillets at all corner nodes
        try:
            new_nodes, new_members, repls = apply_fillets_at_nodes(
                nodes, members, corners, radius=0.25, n_segments=4
            )
        except Exception as e:
            print("Error applying corner fillets:", e)
            return 2

        plot_frame(
            new_nodes, new_members, out_path="scripts/sample_frame_fillet_corners.png"
        )

        # Assemble global stiffness and solve with a 10 N x-load at Node_1_1
        dof_map = dof_index_map(new_members)
        total_free_dofs = sum(len(v) for v in dof_map["Free"].values())
        Kff_sym, dof_indices = assemblyKff(
            new_members, dof_map, total_dofs=total_free_dofs
        )

        # Convert to numeric
        try:
            K_numeric = np.array(Kff_sym.evalf(), dtype=np.float64)
        except Exception:
            K_numeric = np.array(
                [[float(x) for x in row] for row in Kff_sym.tolist()], dtype=np.float64
            )

        # Force vector: 10 N in x-direction at Node_1_1
        F = np.zeros((total_free_dofs,))
        target_node = "Node_1_1"
        if target_node in dof_indices and "x" in dof_indices[target_node]:
            idx = dof_indices[target_node]["x"]
            F[idx] = 10.0
        else:
            # fallback: first free DOF
            if dof_indices:
                first_node = next(iter(dof_indices))
                first_dof = next(iter(dof_indices[first_node].values()))
                F[first_dof] = 10.0

        try:
            disp = linalg.solve(K_numeric, F)
            print("Nodal displacements (free DOFs):")
            for node_name, dofs in dof_indices.items():
                for dof_name, idx in dofs.items():
                    print(f"  {node_name} DOF {dof_name}: {disp[idx]:.6e}")

            print("\nCorner displacements (u_x, u_y):")
            for cn in corners:
                ux = None
                uy = None
                if cn in dof_indices and isinstance(dof_indices[cn], dict):
                    ux_idx = dof_indices[cn].get("x")
                    uy_idx = dof_indices[cn].get("y")
                    if ux_idx is not None:
                        ux = float(disp[ux_idx])
                    if uy_idx is not None:
                        uy = float(disp[uy_idx])
                ux_str = f"{ux:.6e}" if ux is not None else "restrained"
                uy_str = f"{uy:.6e}" if uy is not None else "restrained"
                print(f"  {cn}: u_x = {ux_str}, u_y = {uy_str}")
        except Exception as e:
            print("Could not solve linear system:", e)

        print("Fillet corners replacements:")
        for k, v in repls.items():
            print(f"  {k} -> {len(v)} segments")
        return 0

    print("No action specified. Use --demo to run the visual fillet example.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
