import math
import numpy as np

kab, kcd = 1, 1 # Stiffness values
kbc = 1e4 # Stiffness value

k = np.matrix([[kab + kbc, kbc, 0], 
                [kbc, kbc + kcd, kcd],
                [0, kcd, kcd]])

# Define the cond and rcond values
cond_value = np.linalg.cond(k)
rcond_value = 1 / cond_value

print("a)")
print("Condition number of k:", cond_value)
print("Reciprocal condition number of k:", rcond_value)

comment = """
b)
The condition number of a matrix is a measure of how sensitive the solution of a system 
of linear equations is to changes in the input data. A high condition number indicates 
that the matrix is close to singular, meaning that small changes in the input can lead 
to large changes in the output. Conversely, a low condition number indicates that the 
matrix is well-conditioned, and the solution will be more stable with respect to input 
changes.

In this case, the condition number of the stiffness matrix 'k' is very high, indicating that 
the system is ill-conditioned. This makes sense considering that there is several orders of
magnitude difference between the stiffness values 'kab'/'kcd' and 'kbc'. The large disparity 
in stiffness values contributes to the ill-conditioning of the system, making it more sensitive
to changes in the input data.
"""

print(comment)

d = np.linalg.solve(k, np.matrix([[1], [0], [1]]))
print("c)")
print("Displacement vector d: \n", d)

# find eigenvalues of k
eigenvalues, _ = np.linalg.eig(k)
print(f'max eigenvalue: {max(eigenvalues)}')
print(f'min eigenvalue: {min(eigenvalues)}')
kappa = max(eigenvalues) / min(eigenvalues)
print(f'kappa = {kappa}')
print(f'log10(kappa) = {math.log10(kappa)}')

# ---------------------------------------------------------------------------

kab, kcd = 1, 1  # Stiffness values
kbc = 1e9  # Stiffness value

k = np.matrix([[kab + kbc, kbc, 0], [kbc, kbc + kcd, kcd], [0, kcd, kcd]])

# Define the cond and rcond values
cond_value = np.linalg.cond(k)
rcond_value = 1 / cond_value

print("a)")
print("Condition number of k:", cond_value)
print("Reciprocal condition number of k:", rcond_value)


d = np.linalg.solve(k, np.matrix([[1], [0], [1]]))
print("c)")
print("Displacement vector d: \n", d)

# find eigenvalues of k
eigenvalues, _ = np.linalg.eig(k)
print(f"max eigenvalue: {max(eigenvalues)}")
print(f"min eigenvalue: {min(eigenvalues)}")
kappa = max(eigenvalues) / min(eigenvalues)
print(f"kappa = {kappa}")
print(f"log10(kappa) = {math.log10(kappa)}")

# ---------------------------------------------------------------------------

kab, kcd = 1, 1  # Stiffness values
kbc = 1e16  # Stiffness value

k = np.matrix([[kab + kbc, kbc, 0], [kbc, kbc + kcd, kcd], [0, kcd, kcd]])

# Define the cond and rcond values
cond_value = np.linalg.cond(k)
rcond_value = 1 / cond_value

print("a)")
print("Condition number of k:", cond_value)
print("Reciprocal condition number of k:", rcond_value)


d = np.linalg.solve(k, np.matrix([[1], [0], [1]]))
print("c)")
print("Displacement vector d: \n", d)

# find eigenvalues of k
eigenvalues, _ = np.linalg.eig(k)
print(f"max eigenvalue: {max(eigenvalues)}")
print(f"min eigenvalue: {min(eigenvalues)}")
kappa = max(eigenvalues) / min(eigenvalues)
print(f"kappa = {kappa}")
print(f"log10(kappa) = {math.log10(kappa)}")
