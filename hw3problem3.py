# Import statements
import numpy as np
from math import sin, cos, pi, atan

'''
Defining the system

Beam member from a-b, b-c
Truss members from a-d, b-d, d-c

Applied temp gradient from a-c of deltaT = 120C
'''

dT = 120 # degC
alpha = 1.2e-5 # mm/mm degC
dabc = 300 # mm
E = 200000 # mPa
I = 200e6 # mm4
Abeam = 8e3 # mm2
Atruss1 = 3e3 # mm2
Atruss2 = 2e3 # mm2

# Find moments using fixed end approach
Ma = E * I * alpha * dT / dabc
Mb1 = - E * I * alpha * dT / dabc
Mb2 = E * I * alpha * dT / dabc
Mc = - E * I * alpha * dT / dabc

Mb = Mb1 + Mb2 # 0 sum, done more so for the sake of completion.
# Really, you could model the whole top beam as a single object,
# which would give the same result.

Pb = E * Atruss2 * alpha * dT
Pd = -Pb

force_vector = np.matrix([
    Ma,
    0,
    Pb,
    Mb,
    0,
    Mc,
    0,
    Pd
])


'''
Unknown DOFs:
    - ThetaA
    - Bx
    - By
    - ThetaB
    - Cx
    - ThetaC
    - Dx
    - Dy

Unknown Forces
    - Ax
    - Ay
    - Cy
'''

# Kff matrix placeholder. Fill in later
Kff = np.matrix([
    # ThetaA, Bx,   By, ThetaB, Cx, ThetaC, Dx, Dy
    0,          0,  0,  0,      0,   0,     0,  0, # ThetaA
    0,          0,  0,  0,      0,   0,     0,  0, # Bx
    0,          0,  0,  0,      0,   0,     0,  0, # By
    0,          0,  0,  0,      0,   0,     0,  0, # ThetaB
    0,          0,  0,  0,      0,   0,     0,  0, # Cx
    0,          0,  0,  0,      0,   0,     0,  0, # ThetaC
    0,          0,  0,  0,      0,   0,     0,  0, # Dx
    0,          0,  0,  0,      0,   0,     0,  0, # Dy
])

p = E * I / 5**3 # for readability
# L = 5 m for beam members
Kab = np.matrix([
    # Ax,       Ay,    ThetaA,      Bx,         By,     ThetaB
    E*Abeam/5,  0,     0,       -E*Abeam/5, 0,      0,
    0,          12*p,  6*5*p,   0,          -12*p,  6*5*p,
    0,          6*5*p, 4*5*5*p, 0,          -6*5*p, 4*5*5*p,
    -E*Abeam/5, 0,     0,       E*Abeam/5,  0,      0,
    0,          -12*p, -6*5*p,  0,          12*p,   -6*5*p,
    0,          6*5*p, 4*5*5*p, 0,          -6*5*p, 4*5*5*p
])

# Kbc is identical to Kab, just different DOFs
Kbc = np.matrix([
    # Bx,       By,    ThetaB,      Cx,         Cy,     ThetaC
    E*Abeam/5,  0,     0,       -E*Abeam/5, 0,      0,
    0,          12*p,  6*5*p,   0,          -12*p,  6*5*p,
    0,          6*5*p, 4*5*5*p, 0,          -6*5*p, 4*5*5*p,
    -E*Abeam/5, 0,     0,       E*Abeam/5,  0,      0,
    0,          -12*p, -6*5*p,  0,          12*p,   -6*5*p,
    0,          6*5*p, 4*5*5*p, 0,          -6*5*p, 4*5*5*p
])


# Vertical Truss element bd

Kbd = np.matrix([
    # Bx,       By,    Dx,      Dy
    E*Atruss2/5,  0,     -E*Atruss2/5, 0,
    0,            0,     0,            0,
    -E*Atruss2/5, 0,     E*Atruss2/5,  0,
    0,            0,     0,            0,
])

c = cos(pi/2)
s = sin(pi/2)
Tbd = np.matrix([
    # Bx, By, Dx, Dy
    c,  s, 0,  0,
    -s, c, 0,  0,
    0,  0, c,  s,
    0,  0, -s, c,
])

Kbd_global = Tbd.T @ Kbd @ Tbd

# Non-vertical truss elements
Kad = np.matrix([
    # Ax,       Ay,    Dx,      Dy
    E*Atruss2/5,  0,     -E*Atruss2/5, 0,
    0,            0,     0,            0,
    -E*Atruss2/5, 0,     E*Atruss2/5,  0,
    0,            0,     0,            0,
])

a = atan(5/2)
c = cos(a)
s = sin(a)
Tad = np.matrix([
    # Bx, By, Dx, Dy
    c,  s, 0,  0,
    -s, c, 0,  0,
    0,  0, c,  s,
    0,  0, -s, c,
])

Kad_global = Tad.T @ Kad @ Tad

Kcd = np.matrix([
    # Ax,       Ay,    Dx,      Dy
    E*Atruss2/5,  0,     -E*Atruss2/5, 0,
    0,            0,     0,            0,
    -E*Atruss2/5, 0,     E*Atruss2/5,  0,
    0,            0,     0,            0,
])

a = atan(-5/2)
c = cos(a)
s = sin(a)
Tcd = np.matrix([
    # Bx, By, Dx, Dy
    c,  s, 0,  0,
    -s, c, 0,  0,
    0,  0, c,  s,
    0,  0, -s, c,
])

Kcd_global = Tcd.T @ Kcd @ Tcd

# Assemble Kff
# Unknown DOFs order:
# 0 ThetaA, 1 Bx, 2 By, 3 ThetaB, 4 Cx, 5 ThetaC, 6 Dx, 7 Dy

# helper to safely read matrix entries
def _m(mat, i, j):
    return np.asarray(mat, dtype=float)[i, j]

Kff_arr = np.zeros((8, 8), dtype=float)

# Kab (Ax, Ay, ThetaA, Bx, By, ThetaB) -> map to global unknowns
map_kab = [-1, -1, 0, 1, 2, 3]
for i_local in range(6):
    gi = map_kab[i_local]
    if gi == -1:
        continue
    for j_local in range(6):
        gj = map_kab[j_local]
        if gj == -1:
            continue
        Kff_arr[gi, gj] += _m(Kab, i_local, j_local)

# Kbc (Bx, By, ThetaB, Cx, Cy, ThetaC)
map_kbc = [1, 2, 3, 4, -1, 5]
for i_local in range(6):
    gi = map_kbc[i_local]
    if gi == -1:
        continue
    for j_local in range(6):
        gj = map_kbc[j_local]
        if gj == -1:
            continue
        Kff_arr[gi, gj] += _m(Kbc, i_local, j_local)

# Kbd_global (Bx, By, Dx, Dy)
map_kbd = [1, 2, 6, 7]
for i_local in range(4):
    gi = map_kbd[i_local]
    for j_local in range(4):
        gj = map_kbd[j_local]
        Kff_arr[gi, gj] += _m(Kbd_global, i_local, j_local)

# Kad_global (Ax, Ay, Dx, Dy) -> only Dx,Dy are unknown
map_kad = [-1, -1, 6, 7]
for i_local in range(4):
    gi = map_kad[i_local]
    if gi == -1:
        continue
    for j_local in range(4):
        gj = map_kad[j_local]
        if gj == -1:
            continue
        Kff_arr[gi, gj] += _m(Kad_global, i_local, j_local)

# Kcd_global (Cx, Cy, Dx, Dy) -> Cy is not an unknown
map_kcd = [4, -1, 6, 7]
for i_local in range(4):
    gi = map_kcd[i_local]
    if gi == -1:
        continue
    for j_local in range(4):
        gj = map_kcd[j_local]
        if gj == -1:
            continue
        Kff_arr[gi, gj] += _m(Kcd_global, i_local, j_local)

# convert to np.matrix to match previous usage
Kff = np.matrix(Kff_arr)

print("Kff matrix:")
for row in np.array(Kff):
    print(row)

displacements = np.linalg.inv(Kff) @ force_vector.T
print("\nDisplacements at unknown DOFs:")
for line in displacements:
    print(f'[{line}]')
