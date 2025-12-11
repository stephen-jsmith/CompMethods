from objects import *
from sympy import sqrt, symbols, Matrix, zeros, pi
import numpy as np
from scripts.preassembly import preassemblyGen, dof_index_map
from typing import List, Union, Optional, Literal
import numpy.linalg as linalg

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def assemblyKff(members: list, dof_map: dict, total_dofs: int) -> Matrix:
    """Assembles the global kff stiffness matrix from member stiffness matrices.
    Parameters
    ----------
    members : list
        List of Member objects in the structure.
    dof_map : dict
        Dictionary mapping node dofs to their locations in the stiffness matrix.
    total_dofs : int
        Total number of DOFs in the global system.
    Returns
    -------
    Matrix
        The assembled global kff stiffness matrix.
    """
    Kff_global = zeros(total_dofs, total_dofs)
    counter = 0
    dof_indices = {}
    for n in dof_map["Free"].keys():
        dof_indices[n] = {}
        for i in dof_map["Free"][n]:
            dof_indices[n][i] = counter
            counter += 1
    for member in members:
        k_global = member.global_stiffness_matrix()
        # member.dof_order contains labels like 'Node_0.25_0.0_y' or 'Node_0.25_0.0_rotation'
        # Iterate element DOFs explicitly so we support different element sizes (truss=4, beam=6, etc.)
        for i, dof_label_i in enumerate(member.dof_order):
            parts_i = dof_label_i.split("_")
            dof_name_i = parts_i[-1]
            node_name_i = "_".join(parts_i[:-1])
            if (
                node_name_i in dof_map["Free"]
                and dof_name_i in dof_map["Free"][node_name_i]
            ):
                global_i = dof_indices[node_name_i][dof_name_i]
                for j, dof_label_j in enumerate(member.dof_order):
                    parts_j = dof_label_j.split("_")
                    dof_name_j = parts_j[-1]
                    node_name_j = "_".join(parts_j[:-1])
                    if (
                        node_name_j in dof_map["Free"]
                        and dof_name_j in dof_map["Free"][node_name_j]
                    ):
                        global_j = dof_indices[node_name_j][dof_name_j]
                        Kff_global[global_i, global_j] += k_global[i, j]

    return Kff_global, dof_indices


if __name__ == """__main__""":
    E, A, I = symbols("E A I")
    print("\n--- Frame Assembly Component ---\n")
    # Define nodes (square: (0,0), (1,0), (1,1), (0,1))
    n1 = Node(0, 0, "pinned", ["x", "y"])
    n2 = Node(1, 0, "pinned", ["x", "y"])
    n3 = Node(1, 1, "free")
    n4 = Node(0, 1, "free")

    # Define members (edges of square + diagonal through center)
    members = [
        Member2D("beam", n1, n2, E=E, A=A, I=I),  # bottom
        Member2D("beam", n2, n3, E=E, A=A, I=I),  # right
        Member2D("beam", n3, n4, E=E, A=A, I=I),  # top
        Member2D("beam", n4, n1, E=E, A=A, I=I),  # left
        Member2D("beam", n1, n3, E=E, A=A, I=I),  # diagonal
    ]
    dof_map = dof_index_map(members)
    print("DOF Map:", dof_map)
    # Compute total number of free DOFs (sum of DOFs for each node)
    total_free_dofs = sum(len(v) for v in dof_map["Free"].values())
    k, dof_indices = assemblyKff(members, dof_map, total_dofs=total_free_dofs)
    print("Size of Assembled Kff Matrix:", k.shape)

    # Replace with actual numeric values for E, A, I if desired
    E_val = 200e9  # Pa
    A_val = 0.01  # m^2
    I_val = 8.3e-6  # m^4

    # Substitute numeric values into the symbolic global stiffness matrix
    k_substituted = k.subs({E: E_val, A: A_val, I: I_val})

    # Convert substituted sympy Matrix to a NumPy float array
    try:
        # Prefer evalf then array conversion
        k_numeric = np.array(k_substituted.evalf(), dtype=np.float64)
    except Exception:
        # Fallback: convert using float() on each entry
        k_numeric = np.array(
            [[float(entry) for entry in row] for row in k_substituted.tolist()],
            dtype=np.float64,
        )

    print("Assembled Numeric Kff Matrix:")
    for row in k_numeric.tolist():
        print(row)

    # Define force vector (numeric) and convert to 1D NumPy array
    forces_sym = Matrix([1000, 0, 0, 0, 0, 0, 0, 0])
    forces_np = np.array(forces_sym.tolist(), dtype=np.float64).reshape((-1,))

    # Solve linear system using NumPy
    displacements = linalg.solve(k_numeric, forces_np)
    print("Nodal Displacements:")
    disp = displacements.tolist()
    print(dof_indices)
    for node in dof_indices:
        for dof in dof_indices[node]:
            index = dof_indices[node][dof]
            print(f"{node} DOF {dof}: {disp[index]}")

    # Do the same with a truss
    print("\n--- Truss Assembly Component ---\n")
    E, A, I = symbols("E A I")

    # Define nodes (square: (0,0), (1,0), (1,1), (0,1))
    n1 = Node(0, 0, "pinned", ["x", "y"])
    n2 = Node(1, 0, "pinned", ["x", "y"])
    n3 = Node(1, 1, "free")
    n4 = Node(0, 1, "free")

    # Define members (edges of square + diagonal through center)
    members = [
        Member2D("truss", n1, n2, E=E, A=A, I=I),  # bottom
        Member2D("truss", n2, n3, E=E, A=A, I=I),  # right
        Member2D("truss", n3, n4, E=E, A=A, I=I),  # top
        Member2D("truss", n4, n1, E=E, A=A, I=I),  # left
        Member2D("truss", n1, n3, E=E, A=A, I=I),  # diagonal
    ]
    dof_map = dof_index_map(members)
    print("DOF Map:", dof_map)
    # Compute total number of free DOFs (sum of DOFs for each node)
    total_free_dofs = sum(len(v) for v in dof_map["Free"].values())
    k, dof_indices = assemblyKff(members, dof_map, total_dofs=total_free_dofs)
    print("Size of Assembled Kff Matrix:", k.shape)

    # Replace with actual numeric values for E, A, I if desired
    E_val = 200e9  # Pa
    A_val = 0.01  # m^2
    I_val = 8.3e-6  # m^4

    # Substitute numeric values into the symbolic global stiffness matrix
    k_substituted = k.subs({E: E_val, A: A_val, I: I_val})

    # Convert substituted sympy Matrix to a NumPy float array
    try:
        # Prefer evalf then array conversion
        k_numeric = np.array(k_substituted.evalf(), dtype=np.float64)
    except Exception:
        # Fallback: convert using float() on each entry
        k_numeric = np.array(
            [[float(entry) for entry in row] for row in k_substituted.tolist()],
            dtype=np.float64,
        )

    print("Assembled Numeric Kff Matrix:")
    for row in k_numeric.tolist():
        print(row)

    # Define force vector (numeric) and convert to 1D NumPy array
    forces_sym = Matrix([1000, 0, 0, 0, 0, 0, 0, 0])
    forces_np = np.array(forces_sym.tolist(), dtype=np.float64).reshape((-1,))

    # Solve linear system using NumPy
    displacements = linalg.solve(k_numeric, forces_np)
    print("Nodal Displacements:")
    disp = displacements.tolist()
    print(dof_indices)
    for node in dof_indices:
        for dof in dof_indices[node]:
            index = dof_indices[node][dof]
            print(f"{node} DOF {dof}: {disp[index]}")
