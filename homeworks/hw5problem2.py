import math
import scipy as sp
import scipy.linalg as la
import numpy as np
import sympy as sym

x, L, E, I, Zeta = sym.symbols("x L E I Zeta")

s1 = [1, 0, 0, 0, 0]  # V1
s2 = [1, L, L**2, L**3, L**4]  # Theta1
s3 = [0, 1, 0, 0, 0]  # V2
s4 = [0, 1, 2 * L, 3 * L**2, 4 * L**3]  # Theta2
s5 = [0, 1, 2 * L, 3 * L**2, 4 * L**3]  # V3

X = sym.Matrix(
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, L, L**2, L**3, L**4],
        [0, 1, 2 * L, 3 * L**2, 4 * L**3],
        [1, L / 2, L**2 / 4, L**3 / 8, L**4 / 16],
    ]
)

M = sym.Matrix([[1, x, x**2, x**3, x**4]])

N = M * X.inv()
N = sym.simplify(N)
N = sym.expand(N)
print("The shape functions are:")
for line in N:
    print(line)


# Derive the second derivatives of the shape functions
d2N = N.diff(x, 2)
d2N = sym.simplify(d2N)
d2N = sym.expand(d2N)
print("\nThe second derivatives of the shape functions are:")
for line in d2N:
    print(line)

# Derive the element stiffness matrix
k_e = E * I * sym.integrate(d2N.T * Zeta * d2N, (x, 0, L))
k_e = sym.simplify(k_e)
k_e = sym.expand(k_e)
print("\nThe element stiffness matrix is:")
for row in k_e.tolist():
    print(row)

print("The element K_14 Foundation entry is:")
print(k_e[0, 3])

print("\n-----------------------------------------\n")

print("For the second part of the problem, discretize the member into two elements.")
print("From symmetry, the stiffness matrices are the same for both elements.\n")

k_ff = sym.Matrix(
    [
        [k_e[2, 2] + k_e[0, 0], k_e[2, 3] + k_e[0, 1]],
        [k_e[3, 2] + k_e[1, 0], k_e[3, 3] + k_e[1, 1]],
    ]
)
k_ff = sym.simplify(k_ff)
k_ff = sym.expand(k_ff)
print("The Kff stiffness matrix is:")
for row in k_ff.tolist():
    print(row)
print(
    "\n The 0's off of the main diagonal make sense, as the two members have values that cancel out"
)

# Substitute numerical values
I_val = 128.5  # in^4
E_val = 29000  # ksi
L_val = 12.5 * 12  # in
Zeta_val = 1.5  # k/in^2

k_ff_num = k_ff.subs({I: I_val, E: E_val, L: L_val, Zeta: Zeta_val})
k_ff_num = sym.simplify(k_ff_num)
print("The numerical Kff stiffness matrix is:")
for row in k_ff_num.tolist():
    print(row)

F = sym.Matrix([[40], [0]])  # K, in-lb

print("\nThe numerical force vector is:")
for row in F.tolist():
    print(row)

d = k_ff_num.inv() * F
print("\nDisplacements at Midspan:")
print(d)

print("\nNo rotation occurs at midspan due to symmetry, as expected.")

# Use the solved midspan displacement and the 5th shape function second
# derivative to compute M = E * I * w''(L/2) * w_mid.
d_mid = d[0]  # vertical displacement at midspan (DOF 0 of d)
d2N5_mid = d2N[4].subs(x, L / 2)
M_mid = E * I * d2N5_mid * d_mid

# Substitute numerical values and print symbolic & numeric results
M_mid_num = sym.N(M_mid.subs({E: E_val, I: I_val, L: L_val}))
print("\nBending moment at midspan (symbolic expression):")
print(sym.simplify(M_mid))
print("\nBending moment at midspan (numeric):")
print(M_mid_num)

print("\nComments: The value is slightly lower than the exact solution from the textbook. I am not sure how I could increase the accuract. Perhaps there is a philosophical error in my approach?")