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

# ------------------------------------------------------------------------------
# This script is a verification that follows this video:
# https://www.youtube.com/watch?v=K56tmCfi1SA
#
# The video creates a weld between two members at a 90 degree angle
# and verifies that the fillet theoretical stresses match an FEM model.
#
# Since I don't have access to the FEM software used in the video,
# I will create a simple recreation using my code in this repo to verify
# that my model works as expected.
# ------------------------------------------------------------------------------

# Define node and member setup
# Units: mm, N
node_A = Node(0.0, 0.0, "fixed", ["x", "y", "rz"])
node_B = Node(50.0, 0.0, "free", [])
node_C = Node(100.0, 0.0, "fixed", ["x", "y", "rz"])
node_D = Node(50.0, 100.0, "free", [])
# Bottom member
member_AB = Member2D("Beam", node_A, node_B, E=210e3, A=600.0, I=100 * 6**3 / 12)
member_BC = Member2D("Beam", node_B, node_C, E=210e3, A=600.0, I=100 * 6**3 / 12)

# Create fillet at node B (6mm depth, triangular fillet both sides)
node_B_1 = Node(50.0, 1.0, "free", [])
node_B_2 = Node(50.0, 2.0, "free", [])
node_B_3 = Node(50.0, 3.0, "free", [])
node_B_4 = Node(50.0, 4.0, "free", [])
node_B_5 = Node(50.0, 5.0, "free", [])
node_B_6 = Node(50.0, 6.0, "free", [])

nodes = [
    node_A,
    node_B,
    node_C,
    node_D,
    node_B_1,
    node_B_2,
    node_B_3,
    node_B_4,
    node_B_5,
    node_B_6,
]

# Create fillet members
fillet_1 = Member2D("Beam", node_B, node_B_1, E=210e3, A=1200.0, I=100 * 12**3 / 12)
fillet_2 = Member2D("Beam", node_B_1, node_B_2, E=210e3, A=1100.0, I=100 * 11**3 / 12)
fillet_3 = Member2D("Beam", node_B_2, node_B_3, E=210e3, A=1000.0, I=100 * 10**3 / 12)
fillet_4 = Member2D("Beam", node_B_3, node_B_4, E=210e3, A=900.0, I=100 * 9**3 / 12)
fillet_5 = Member2D("Beam", node_B_4, node_B_5, E=210e3, A=800.0, I=100 * 8**3 / 12)
fillet_6 = Member2D("Beam", node_B_5, node_B_6, E=210e3, A=700.0, I=100 * 7**3 / 12)
member_BD = Member2D(
    "Beam", node_B_6, node_D, E=210e3, A=600.0, I=100 * 6**3 / 12
)  # Not technically BD, but you get the gist
# Assemble all members
members = [
    member_AB,
    member_BC,
    fillet_1,
    fillet_2,
    fillet_3,
    fillet_4,
    fillet_5,
    fillet_6,
    member_BD,
]

# Apply load at node D (200 N in x and y directions)
loads = np.array(
    [
        0,  # Node B DOF X
        0,  # Node B DOF Y
        0,  # Node B DOF RZ
        0,  # Node fillet_1 DOF X
        0,  # Node fillet_1 DOF Y
        0,  # Node fillet_1 DOF RZ
        0,  # Node fillet_2 DOF X
        0,  # Node fillet_2 DOF Y
        0,  # Node fillet_2 DOF RZ
        0,  # Node fillet_3 DOF X
        0,  # Node fillet_3 DOF Y
        0,  # Node fillet_3 DOF RZ
        0,  # Node fillet_4 DOF X
        0,  # Node fillet_4 DOF Y
        0,  # Node fillet_4 DOF RZ
        200.0,  # Node D DOF X
        200.0,  # Node D DOF Y
        0.0,  # Node D DOF RZ
        0,  # Node fillet_5 DOF X
        0,  # Node fillet_5 DOF Y
        0,  # Node fillet_5 DOF RZ
        0,  # Node fillet_6 DOF X
        0,  # Node fillet_6 DOF Y
        0,  # Node fillet_6 DOF RZ
    ]
)

# Assemble global stiffness matrix
dof_map = dof_index_map(members)
print("Free DOF Map:")
for item in dof_map["Free"]:
    print(f"Node {item} Free DOFs: {dof_map['Free'][item]}")
# Compute total number of free DOFs (sum of DOFs for each node)
total_free_dofs = sum(len(v) for v in dof_map["Free"].values())
k, dof_indices = assemblyKff(members, dof_map, total_dofs=total_free_dofs)
print("Size of Assembled Kff Matrix:", k.shape)

# Build a numeric load vector that matches the assembled K size using the
# `dof_indices` mapping returned by `assemblyKff`. This ensures the force
# vector length matches the stiffness matrix and avoids shape mismatches.
loads_vec = np.zeros(k.shape[0], dtype=float)
# Apply the 200 N loads at node D for DOFs 'x' and 'y' (if present)
node_name = node_D.name
if node_name in dof_indices:
    if "x" in dof_indices[node_name]:
        loads_vec[dof_indices[node_name]["x"]] = 200.0
    if "y" in dof_indices[node_name]:
        loads_vec[dof_indices[node_name]["y"]] = 200.0
    # rotation left as zero (explicit for clarity)
    if "rotation" in dof_indices[node_name]:
        loads_vec[dof_indices[node_name]["rotation"]] = 0.0
else:
    print(f"Warning: node {node_name} not found in dof_indices; no loads applied")

# Ensure stiffness matrix is numeric (convert from SymPy Matrix if necessary)
try:
    k_numeric = np.array(k.evalf(), dtype=np.float64)
except Exception:
    k_numeric = np.array(
        [[float(entry) for entry in row] for row in k.tolist()], dtype=np.float64
    )

displacements = linalg.solve(k_numeric, loads_vec)
disp = displacements.tolist()
nodal_displacement_map = {}
for node in nodes: # Useful for the next step, finding internal forces/stresses
    nodal_displacement_map[node] = {}
    if node.name in dof_indices:
        for dof in dof_indices[node.name]:
            index = dof_indices[node.name][dof]
            nodal_displacement_map[node][dof] = disp[index]
    else:
        nodal_displacement_map[node] = {"x": 0.0, "y": 0.0, "rotation": 0.0}
print("\nNodal Displacement Map:")
for node in nodal_displacement_map:
    print(f"{node.name}: {nodal_displacement_map[node]}")
# Compute internal forces at the fillet members
print("\nInternal Forces at Fillet Members:")
for member in members: 
    if member in [
        fillet_1,
        fillet_2,
        fillet_3,
        fillet_4,
        fillet_5,
        fillet_6,
    ]:
        k = member.global_stiffness_matrix()
        displacements = np.array(
            [
                nodal_displacement_map[member.node_start]["x"],
                nodal_displacement_map[member.node_start]["y"],
                nodal_displacement_map[member.node_start]["rotation"],
                nodal_displacement_map[member.node_end]["x"],
                nodal_displacement_map[member.node_end]["y"],
                nodal_displacement_map[member.node_end]["rotation"],
            ]
        )
        internal_forces = k @ displacements
        print(
            f"Member {member.node_start.name} to {member.node_end.name} Internal Forces:"
        )
        dof_identifiers = ["Fx_start", "Fy_start", "M_start", "Fx_end", "Fy_end", "M_end"]
        for i, force in enumerate(internal_forces):
            print(f"  {dof_identifiers[i]}: {force} N")
        F_axial = internal_forces[3]  # Axial force at the end of the fillet member
        F_shear = internal_forces[4]  # Shear force at the end of the fillet member
        A = member.A  # Cross-sectional area of the fillet member
        stress_axial = F_axial / A if A != 0 else 0
        stress_shear = F_shear / A if A != 0 else 0
        print(f"  Axial Stress: {stress_axial} N/mm2")
        print(f"  Shear Stress: {stress_shear} N/mm2")