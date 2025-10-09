from scripts import *
from objects import *
import numpy as np
from sympy import symbols, Matrix, zeros

print("ASSUMPTION: The beam will not axially deform.")
print("No axial loads, no axial deformation. If this")
print("was a p delta analysis, then axial deformation")
print("would need to be considered.")

a = Node(0, 2, "pinned", ["x", "y"])  # Node a at (0,2) pinned
b = Node(5, 2, "roller", ["x"])  # Node b at (5,2) is treated as a roller by the code as x translation is constrained.
c = Node(10, 2, "pinned", ["x", "y"])  # Node c at (10,2) pinned
d = Node(5, 0, "free")  # Node d at (5,0) free

# Define members from problem statement
E = 200000  # Mpa
Abeam = 8 * 10e3  # mm^2
Atruss12 = 3 * 10e3  # mm^2
Atruss3 = 2 * 10e3  # mm^2
Ibeam = 200 * 10e6  # mm^4
Itruss = None  # Not used for truss elements

beam1 = Member("beam", a, b, E=E, A=Abeam, I=Ibeam)  # Member ab
beam2 = Member("beam", b, c, E=E, A=Abeam, I=Ibeam)  # Member bc
truss1 = Member("truss", a, d, E=E, A=Atruss12, I=Itruss)  # Member ad
truss2 = Member("truss", d, c, E=E, A=Atruss12, I=Itruss)  # Member dc
truss3 = Member("truss", b, d, E=E, A=Atruss3, I=Itruss)  # Member bd

members = [beam1, beam2, truss1, truss2, truss3]

for member in members:
    print(f"Member {member}:")
    print(f"  Length: {member.length}")
    print(f"  Angle (radians): {member.angle}")
    k = member.global_stiffness_matrix()
    for row in np.array(k):
        print(row)
    print("\n-------------------\n")


K, note = preassemblyTrusses(members)
print("Global stiffness matrix K:")
for row in np.array(K):
    print(row)
print("\nNote:")
print(note)
print("\nDetermining if K is singular... (Matrix(K).det() == 0)")
if Matrix(K).det() == 0:
    print("------------------\n  K is singular.\n------------------\n")
else:
    print("------------------\n  K is not singular.\n------------------\n")

print("Partitioning K...")
# Partition K into submatrices based on DOF types
# partition_from_members returns 6 values: (Kff, Kfc, Kcf, Kcc, free_idx, constrained_idx)
kff, kfs, ksf, kss, free_idx, constrained_idx, free_labels, constrained_labels = (
    partition_from_members(K, members=members)
)
print("Kff:")
for row in np.array(kff):
    print(row)
print("\nFree DOF labels:")
print(free_labels)
print("Constrained DOF labels:")
print(constrained_labels)
print("With free DOF indices:", free_idx)

print("\n Node 2y is Node b y deflection, which we are solving for.")
print("Force vector:")
F = Matrix([[100], [0], [0]])
print(F)
print("\n Solving for displacements at free DOFs...")
kff_inv = kff.inv()
d_free = kff_inv * F
for line in d_free:
    print(f'[{line}]')

print("-----------------------------")
print("Node b y deflection: ", d_free[0])
print("-----------------------------")

print("\nI'm not so sure about that answer...")