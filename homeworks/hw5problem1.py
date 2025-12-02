import math
import scipy as sp
import scipy.linalg as la
import numpy as np
import sympy as sym


x, L, E, I = sym.symbols('x L E I')

s1 = [1, 0, 0, 0, 0]
s2 = [1, L, L**2, L**3, L**4]
s3 = [0, 1, 0, 0, 0]
s4 = [0, 1, 2 * L, 3 * L**2, 4 * L**3]
s5 = [0, 1, 2 * L, 3 * L**2, 4 * L**3]

X = sym.Matrix(
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, L, L**2, L**3, L**4],
        [0, 1, 2 * L, 3 * L**2, 4 * L**3],
        [1, L / 2, L**2 / 4, L**3 / 8, L**4 / 16],
    ]
)

M = sym.Matrix([
    [1, x, x**2, x**3, x**4]
])

N = M * X.inv() 
N = sym.simplify(N)
N = sym.expand(N)
print("The shape functions are:")
for line in N:
    print(line)


# Use the following code to convert sympy expressions to numerical functions
N_func = sym.lambdify((x, L), N, modules='numpy')
# Example usage:
x_val = 2.0
L_val = 4.0
N_evaluated = N_func(x_val, L_val)
print("\nSanity check to verify numerical evaluation of shape functions:")
print("Evaluated shape functions at x = {}, L = {}".format(x_val, L_val))
for i, val in enumerate(N_evaluated[0]):
    print("N_{} = {}".format(i, val))

# Derive the second derivatives of the shape functions
d2N = N.diff(x, 2)
d2N = sym.simplify(d2N)
d2N = sym.expand(d2N)
print("\nThe second derivatives of the shape functions are:")
for line in d2N:
    print(line)

# Convert second derivatives to numerical functions
d2N_func = sym.lambdify((x, L), d2N, modules='numpy')
# Example usage:
d2N_evaluated = d2N_func(x_val, L_val)
print("\nSanity check to verify numerical evaluation of second derivatives:")
print("Evaluated second derivatives at x = {}, L = {}".format(x_val, L_val))
for i, val in enumerate(d2N_evaluated[0]):
    print("d2N_{} = {}".format(i, val))

# Derive the element stiffness matrix
k_e = E * I * sym.integrate(d2N.T * d2N, (x, 0, L))
k_e = sym.simplify(k_e)
k_e = sym.expand(k_e)
print("\nThe element stiffness matrix is:")
for row in k_e.tolist():
    print(row)

# Solve for DOF 1, 2, and 5 as if cantilevered from DOF 3 and 4
k_ff = sym.Matrix([
    [k_e[0, 0], k_e[0, 1], k_e[0, 4]],
    [k_e[1, 0], k_e[1, 1], k_e[1, 4]],
    [k_e[4, 0], k_e[4, 1], k_e[4, 4]]
])

print("\nThe kff stiffness matrix is:")
for row in k_ff.tolist():
    print(row)

# Set the symbolic values for E, I, and L
E_val = 200 # Kn/mm^2
L_val = 2000 # mm
I_val = 6e4 # mm^4

F = sym.Matrix([7, 0, 0])  # 7 kN point load at the free end

k_ff_num = k_ff.subs({E: E_val, I: I_val, L: L_val})
k_ff_num = sym.N(k_ff_num)
print("\nThe numerical reduced stiffness matrix is:")
for row in k_ff_num.tolist():
    print(row)
d = k_ff_num.inv() * F
print("\nDisplacements at DOF 1, 2, and 5:")
print(d)
# Analytical solution for verification
print("\nAnalytical solution of vertical displacement for verification: ", 7 * L_val**3 / (3 * E_val * I_val))

# Find internal member forces
phi_col = sym.Matrix([d2N[0], d2N[1], d2N[4]]).T * d
phi_col = phi_col.subs({E: E_val, I: I_val, L: L_val})
phi_col = [sym.simplify(i) for i in phi_col]
Mx = 0
for i in phi_col:
    Mx += i * E_val * I_val
Mx = sym.simplify(Mx)
Mx = sym.expand(Mx)
print("\nThe expression for Bending Moment along the beam is:")
print(Mx)

# Evaluate bending moment at x = 0
Mx_0 = Mx.subs({x: 0})
print("\nThe bending moment at the fixed end (x=0) is:")
print(Mx_0)
print("Functionally, this is equal to 0. This is expected as the moment at the free end of a cantilever beam under point load is zero.")