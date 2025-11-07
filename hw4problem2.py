from scripts import *
from objects import *
from objects import Member2D as Member
import numpy as np
from sympy import symbols, Matrix, zeros

print("ASSUMPTION: The beam will not axially deform.")
print("No axial loads, no axial deformation. If this")
print("was a p delta analysis, then axial deformation")
print("would need to be considered.")

a = Node(0, 2, "pinned", ["x", "y"])  # Node a at (0,2) pinned
b = Node(
    5, 2, "roller", ["x"]
)  # Node b at (5,2) is treated as a roller by the code as x translation is constrained.
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
    print(f"Member {member.name}:")
    print(f"  Length: {member.length}")
    print(f"  Angle (radians): {member.angle}")
    k = member.global_stiffness_matrix()
    print(member.dof_order)
    for row in np.array(k):
        print(row)
    print("\n-------------------\n")


k, order = preassemblyGen(members)

print("Global stiffness matrix K:")
for row in np.array(k):
    print(row)

print("\nDOF Order:")
print(order)