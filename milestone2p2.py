from scripts.preassembly import dof_index_map
from scripts.assembly import assemblyKff
import numpy.linalg as linalg
from objects.member import Member2D
from objects.node import Node
from sympy import Matrix
import numpy as np


# Helper: compute equivalent nodal loads for a (possibly partial) UDL on a beam element
def element_udl_equivalent(member, load_x_start, load_x_end, q):
    """Return a 6x1 local equivalent nodal load vector (SymPy Matrix) for a UDL
    of magnitude q applied over [load_x_start, load_x_end] in global coords.

    The member local DOF ordering is [u1, v1, rotation1, u2, v2, rotation2].
    Only the transverse DOFs (v1, rotation1, v2, rotation2) receive loads.
    """
    # element geometry
    x0 = float(member.node_start.x)
    x1 = float(member.node_end.x)
    L = float(member.length)
    # overlap in global coords
    a_global = max(x0, load_x_start)
    b_global = min(x1, load_x_end)
    overlap = b_global - a_global
    if overlap <= 1e-12:
        return Matrix([0, 0, 0, 0, 0, 0])

    # convert overlap to local coordinates x in [0, L]
    a = a_global - x0
    b = b_global - x0
    xi_a = a / L
    xi_b = b / L

    # Integrals of Hermite shape functions over xi in [xi_a, xi_b]
    # I1 = L * ∫(1 - 3ξ^2 + 2ξ^3) dξ
    I1 = L * ((xi_b - xi_b**3 + 0.5 * xi_b**4) - (xi_a - xi_a**3 + 0.5 * xi_a**4))
    # I2 = L^2 * ∫(ξ - 2ξ^2 + ξ^3) dξ
    I2 = L**2 * (
        (0.5 * xi_b**2 - (2.0 / 3.0) * xi_b**3 + 0.25 * xi_b**4)
        - (0.5 * xi_a**2 - (2.0 / 3.0) * xi_a**3 + 0.25 * xi_a**4)
    )
    # I3 = L * ∫(3ξ^2 - 2ξ^3) dξ
    I3 = L * ((xi_b**3 - 0.5 * xi_b**4) - (xi_a**3 - 0.5 * xi_a**4))
    # I4 = L^2 * ∫(-ξ^2 + ξ^3) dξ
    I4 = L**2 * (
        ((-1.0 / 3.0) * xi_b**3 + 0.25 * xi_b**4)
        - ((-1.0 / 3.0) * xi_a**3 + 0.25 * xi_a**4)
    )

    f_local = Matrix([0, q * I1, q * I2, 0, q * I3, q * I4])
    return f_local


# Helper: compute bending moment at a given global x coordinate
def compute_bending_moment_at_x(members, dof_indices, displacements, x_global):
    """Locate the element that contains x_global and compute the bending
    moment M(x) = E * I * d2v/dx2 using Hermite cubic shape functions.

    Returns (member, xi, M) or (None, None, None) if x not in any element.
    """
    tol = 1e-9
    for member in members:
        x0 = float(member.node_start.x)
        x1 = float(member.node_end.x)
        if x_global + tol < min(x0, x1) or x_global - tol > max(x0, x1):
            continue
        L = float(member.length)
        xi = (x_global - x0) / L
        if xi < -tol or xi > 1 + tol:
            continue

        # fetch nodal DOFs for this element
        n1 = member.node_start.name
        n2 = member.node_end.name
        try:
            v1 = float(displacements[dof_indices[n1]["y"]])
            th1 = float(displacements[dof_indices[n1]["rotation"]])
            v2 = float(displacements[dof_indices[n2]["y"]])
            th2 = float(displacements[dof_indices[n2]["rotation"]])
        except Exception:
            return (member, xi, None)

        # second derivatives of Hermite shape functions w.r.t xi
        N1_dd = -6.0 + 12.0 * xi
        N2_dd = L * (-4.0 + 6.0 * xi)
        N3_dd = 6.0 - 12.0 * xi
        N4_dd = L * (-2.0 + 6.0 * xi)

        # d2v/dx2 = (1 / L^2) * (N1_dd*v1 + N2_dd*th1 + N3_dd*v2 + N4_dd*th2)
        d2v_dx2 = (1.0 / (L * L)) * (
            N1_dd * v1 + N2_dd * th1 + N3_dd * v2 + N4_dd * th2
        )

        try:
            E = float(member.E)
            I = float(member.I)
            M = E * I * d2v_dx2
        except Exception:
            M = None

        return (member, xi, M)



# 5 Node Beam
def milestone2p2_5nodes(udl_value: float = -10000.0):
    """Run example 5-node beam with a uniform distributed load applied
    from node2 (x=0.25) to node4 (x=0.75).

    Parameters
    ----------
    udl_value : float
        Magnitude of the distributed load (force per unit length). Negative
        for downward load. Default -10000 N/m.
    """
    L = 10.0  # Length of the beam
    # Define nodes (use Node(x, y, fixity, restrained_dofs))
    # Pin the two ends to avoid rigid-body motion
    node1 = Node(0.0, 0.0, "pinned", ["x", "y"])  # left support
    node2 = Node(0.25 * L, 0.0, "free")
    node3 = Node(0.5 * L, 0.0, "free")
    node4 = Node(0.75 * L, 0.0, "free")
    node5 = Node(1.0 * L, 0.0, "pinned", ["x", "y"])  # right support
    # Material / section
    E = 25e9
    A = 0.003
    I = 0.0010125

    # Create beam members (element_type 'beam')
    m1 = Member2D("beam", node1, node2, E=E, A=A, I=I)
    m2 = Member2D("beam", node2, node3, E=E, A=A, I=I)
    m3 = Member2D("beam", node3, node4, E=E, A=A, I=I)
    m4 = Member2D("beam", node4, node5, E=E, A=A, I=I)
    members = [m1, m2, m3, m4]

    # Build DOF map and total DOFs
    dof_map = dof_index_map(members)
    # print("DOF Map:", dof_map) # Debug purpose
    total_free_dofs = sum(len(v) for v in dof_map["Free"].values())

    # Assemble global stiffness (sympy Matrix) and get free-DOF indices
    K_sym, dof_indices = assemblyKff(members, dof_map, total_dofs=total_free_dofs)

    # Convert sympy matrix to numeric numpy array
    K_numeric = np.array(K_sym.evalf(), dtype=np.float64)

    # `dof_indices` returned by assemblyKff maps node_name -> {dof: index}

    # Initialize global force vector (1D NumPy array)
    F_global = np.zeros((total_free_dofs,), dtype=np.float64)

    # Helper to add a member's equivalent nodal loads to the global vector
    for member in members:
        q = udl_value
        f_local = element_udl_equivalent(member, 0.25 * L, 0.75 * L, q)
        # skip if no load on this element
        if all([float(val) == 0.0 for val in list(f_local)]):
            continue
        # Transform to global coordinates
        T = member.transformation_matrix()
        f_global_sym = T.T * f_local
        # Add to global force vector where DOFs are free
        for i, dof_label in enumerate(member.dof_order):
            # dof_label like 'Node_0.25_0.0_x' -> split off node name and dof
            parts = dof_label.split("_")
            dof_name = parts[-1]
            node_name = "_".join(parts[:-1])
            if node_name in dof_indices and dof_name in dof_indices[node_name]:
                idx = dof_indices[node_name][dof_name]
                F_global[idx] += float(f_global_sym[i])

    # print("Global force vector (from UDL on members between node2 and node4):") # More debug purposes
    # print(F_global)

    # Solve for displacements
    displacements = linalg.solve(K_numeric, F_global)

    # --- Compute midspan displacement for member2 (between node2 and node3)  # m2
    # element length and local coordinate (midspan xi=0.5)
    L2 = float(m2.length)
    xi = 0.5

    # fetch DOF values: vertical DOFs are labeled 'y', rotations labeled 'rotation'
    n1_name = m2.node_start.name
    n2_name = m2.node_end.name
    v1 = displacements[dof_indices[n1_name]["y"]]
    th1 = displacements[dof_indices[n1_name]["rotation"]]
    v2 = displacements[dof_indices[n2_name]["y"]]
    th2 = displacements[dof_indices[n2_name]["rotation"]]

    # Hermite cubic shape functions for transverse displacement
    N1 = 1 - 3 * xi**2 + 2 * xi**3
    N2 = L2 * (xi - 2 * xi**2 + xi**3)
    N3 = 3 * xi**2 - 2 * xi**3
    N4 = L2 * (-(xi**2) + xi**3)

    v_mid = N1 * v1 + N2 * th1 + N3 * v2 + N4 * th2
    print(
        f"Midspan displacement of member2 (vertical) at x={float(m2.node_start.x)+0.5*L2}: {v_mid}"
    )
    
    # compute bending moment at x=3.75 m if present in this mesh
    member, xi, M = compute_bending_moment_at_x(members, dof_indices, displacements, 3.75)
    if M is None:
        print("Bending moment at x=3.75 m: not available for this mesh")
    else:
        print(f"Bending moment at x=3.75 m: {M} N*m (element {member.node_start.name}->{member.node_end.name}, xi={xi})")
    print(" ------------------------------------------------ ")


# 9 Node Beam
def milestone2p2_9nodes(udl_value: float = -10000.0):
    """Run example 5-node beam with a uniform distributed load applied
    from node2 (x=0.25) to node4 (x=0.75).

    Parameters
    ----------
    udl_value : float
        Magnitude of the distributed load (force per unit length). Negative
        for downward load. Default -10000 N/m.
    """
    L = 10.0  # Length of the beam
    # Define nodes (use Node(x, y, fixity, restrained_dofs))
    # Pin the two ends to avoid rigid-body motion
    node1 = Node(0.0 * L, 0.0, "pinned", ["x", "y"])  # left support
    node2 = Node(1 / 8 * L, 0.0, "free")
    node3 = Node(0.25 * L, 0.0, "free")
    node4 = Node(3 / 8 * L, 0.0, "free")
    node5 = Node(0.5 * L, 0.0, "free")
    node6 = Node(5 / 8 * L, 0.0, "free")
    node7 = Node(0.75 * L, 0.0, "free")
    node8 = Node(7 / 8 * L, 0.0, "free")
    node9 = Node(1.0 * L, 0.0, "pinned", ["x", "y"])  # right support

    # Material / section
    E = 25e9
    A = 0.003
    I = 0.0010125

    # Create beam members (element_type 'beam')
    m1 = Member2D("beam", node1, node2, E=E, A=A, I=I)
    m2 = Member2D("beam", node2, node3, E=E, A=A, I=I)
    m3 = Member2D("beam", node3, node4, E=E, A=A, I=I)
    m4 = Member2D("beam", node4, node5, E=E, A=A, I=I)
    m5 = Member2D("beam", node5, node6, E=E, A=A, I=I)
    m6 = Member2D("beam", node6, node7, E=E, A=A, I=I)
    m7 = Member2D("beam", node7, node8, E=E, A=A, I=I)
    m8 = Member2D("beam", node8, node9, E=E, A=A, I=I)
    members = [m1, m2, m3, m4, m5, m6, m7, m8]
    # Build DOF map and total DOFs
    dof_map = dof_index_map(members)
    # print("DOF Map:", dof_map) # Debug purpose
    total_free_dofs = sum(len(v) for v in dof_map["Free"].values())

    # Assemble global stiffness (sympy Matrix) and get free-DOF indices
    K_sym, dof_indices = assemblyKff(members, dof_map, total_dofs=total_free_dofs)

    # Convert sympy matrix to numeric numpy array
    K_numeric = np.array(K_sym.evalf(), dtype=np.float64)

    # `dof_indices` returned by assemblyKff maps node_name -> {dof: index}

    # Initialize global force vector (1D NumPy array)
    F_global = np.zeros((total_free_dofs,), dtype=np.float64)

    # Helper to add a member's equivalent nodal loads to the global vector
    for member in members:
        q = udl_value
        f_local = element_udl_equivalent(member, 0.25 * L, 0.75 * L, q)
        if all([float(val) == 0.0 for val in list(f_local)]):
            continue
        T = member.transformation_matrix()
        f_global_sym = T.T * f_local
        for i, dof_label in enumerate(member.dof_order):
            parts = dof_label.split("_")
            dof_name = parts[-1]
            node_name = "_".join(parts[:-1])
            if node_name in dof_indices and dof_name in dof_indices[node_name]:
                idx = dof_indices[node_name][dof_name]
                F_global[idx] += float(f_global_sym[i])

    # print("Global force vector (from UDL on members between node2 and node4):") # More debug purposes
    # print(F_global)

    # Solve for displacements
    displacements = linalg.solve(K_numeric, F_global)
    target_node = "Node_3.75_0.0"
    print(f"Nodal Displacement at {target_node}:   (8 beams)")
    if target_node in dof_indices:
        for dof_name, idx in dof_indices[target_node].items():
            print(f"{target_node} DOF {dof_name}: {displacements[idx]}")
    else:
        print(f"{target_node} not found in dof_indices")
    
    # bending moment at x=3.75 m
    member, xi, M = compute_bending_moment_at_x(members, dof_indices, displacements, 3.75)
    if M is None:
        print("Bending moment at x=3.75 m: not available for this mesh")
    else:
        print(f"Bending moment at x=3.75 m: {M} N*m (element {member.node_start.name}->{member.node_end.name}, xi={xi})")
    print(" ------------------------------------------------ ")


# 17 Node Beam
def milestone2p2_17nodes(udl_value: float = -10000.0):
    """Run example 5-node beam with a uniform distributed load applied
    from node2 (x=0.25) to node4 (x=0.75).

    Parameters
    ----------
    udl_value : float
        Magnitude of the distributed load (force per unit length). Negative
        for downward load. Default -10000 N/m.
    """
    L = 10.0  # Length of the beam
    # Define nodes (use Node(x, y, fixity, restrained_dofs))
    # Pin the two ends to avoid rigid-body motion
    # 17 evenly spaced nodes from x=0 to x=1 (end nodes pinned)
    node1 = Node(0.0 * L, 0.0, "pinned", ["x", "y"])  # left support
    node2 = Node(1 / 16 * L, 0.0, "free")
    node3 = Node(1 / 8 * L, 0.0, "free")
    node4 = Node(3 / 16 * L, 0.0, "free")
    node5 = Node(1 / 4 * L, 0.0, "free")
    node6 = Node(5 / 16 * L, 0.0, "free")
    node7 = Node(3 / 8 * L, 0.0, "free")
    node8 = Node(7 / 16 * L, 0.0, "free")
    node9 = Node(1 / 2 * L, 0.0, "free")
    node10 = Node(9 / 16 * L, 0.0, "free")
    node11 = Node(5 / 8 * L, 0.0, "free")
    node12 = Node(11 / 16 * L, 0.0, "free")
    node13 = Node(3 / 4 * L, 0.0, "free")
    node14 = Node(13 / 16 * L, 0.0, "free")
    node15 = Node(7 / 8 * L, 0.0, "free")
    node16 = Node(15 / 16 * L, 0.0, "free")
    node17 = Node(1.0 * L, 0.0, "pinned", ["x", "y"])  # right support

    # Material / section
    E = 25e9
    A = 0.003
    I = 0.0010125

    # Create beam members (element_type 'beam')
    m1 = Member2D("beam", node1, node2, E=E, A=A, I=I)
    m2 = Member2D("beam", node2, node3, E=E, A=A, I=I)
    m3 = Member2D("beam", node3, node4, E=E, A=A, I=I)
    m4 = Member2D("beam", node4, node5, E=E, A=A, I=I)
    m5 = Member2D("beam", node5, node6, E=E, A=A, I=I)
    m6 = Member2D("beam", node6, node7, E=E, A=A, I=I)
    m7 = Member2D("beam", node7, node8, E=E, A=A, I=I)
    m8 = Member2D("beam", node8, node9, E=E, A=A, I=I)
    m9 = Member2D("beam", node9, node10, E=E, A=A, I=I)
    m10 = Member2D("beam", node10, node11, E=E, A=A, I=I)
    m11 = Member2D("beam", node11, node12, E=E, A=A, I=I)
    m12 = Member2D("beam", node12, node13, E=E, A=A, I=I)
    m13 = Member2D("beam", node13, node14, E=E, A=A, I=I)
    m14 = Member2D("beam", node14, node15, E=E, A=A, I=I)
    m15 = Member2D("beam", node15, node16, E=E, A=A, I=I)
    m16 = Member2D("beam", node16, node17, E=E, A=A, I=I)
    members = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16]
    # Build DOF map and total DOFs
    dof_map = dof_index_map(members)
    # print("DOF Map:", dof_map) # Debug purpose
    total_free_dofs = sum(len(v) for v in dof_map["Free"].values())

    # Assemble global stiffness (sympy Matrix) and get free-DOF indices
    K_sym, dof_indices = assemblyKff(members, dof_map, total_dofs=total_free_dofs)

    # Convert sympy matrix to numeric numpy array
    K_numeric = np.array(K_sym.evalf(), dtype=np.float64)

    # `dof_indices` returned by assemblyKff maps node_name -> {dof: index}

    # Initialize global force vector (1D NumPy array)
    F_global = np.zeros((total_free_dofs,), dtype=np.float64)
    # Helper to add a member's equivalent nodal loads to the global vector
    for member in members:
        # check if the member lies entirely within the loaded region [0.25,0.75] (fraction of beam length)
        x_start = float(member.node_start.x)
        x_end = float(member.node_end.x)
        # compare against 0.25*L and 0.75*L (beam length L is defined at top of each function)
        if x_start >= 0.25 * L - 1e-12 and x_end <= 0.75 * L + 1e-12:
            # Apply full-element UDL of magnitude udl_value on this element
            Le = float(member.length)
            q = udl_value
            # Equivalent nodal loads in local element coordinates [u1,v1,th1,u2,v2,th2]
            f_local = Matrix(
                [
                    0,
                    q * Le / 2.0,
                    q * Le * Le / 12.0,
                    0,
                    q * Le / 2.0,
                    -q * Le * Le / 12.0,
                ]
            )
            # Transform to global coordinates
            T = member.transformation_matrix()
            f_global_sym = T.T * f_local
            # Add to global force vector where DOFs are free
            for i, dof_label in enumerate(member.dof_order):
                # dof_label like 'Node_0.25_0.0_x' -> split off node name and dof
                parts = dof_label.split("_")
                dof_name = parts[-1]
                node_name = "_".join(parts[:-1])
                if node_name in dof_indices and dof_name in dof_indices[node_name]:
                    idx = dof_indices[node_name][dof_name]
                    F_global[idx] += float(f_global_sym[i])

    # print(
    #    "Global force vector (from UDL on members between node2 and node4):"
    # )  # More debug purposes
    # print(F_global)

    # Solve for displacements
    displacements = linalg.solve(K_numeric, F_global)
    target_node = "Node_3.75_0.0"
    print(f"Nodal Displacement at {target_node}:    (16 beams)")
    if target_node in dof_indices:
        for dof_name, idx in dof_indices[target_node].items():
            print(f"{target_node} DOF {dof_name}: {displacements[idx]}")
    else:
        print(f"{target_node} not found in dof_indices")
    
    # bending moment at x=3.75 m
    member, xi, M = compute_bending_moment_at_x(members, dof_indices, displacements, 3.75)
    if M is None:
        print("Bending moment at x=3.75 m: not available for this mesh")
    else:
        print(f"Bending moment at x=3.75 m: {M} N*m (element {member.node_start.name}->{member.node_end.name}, xi={xi})")
    print(" ------------------------------------------------ ")


# 33 Node Beam
def milestone2p2_33nodes(udl_value: float = -10000.0):
    """Run example 5-node beam with a uniform distributed load applied
    from node2 (x=0.25) to node4 (x=0.75).

    Parameters
    ----------
    udl_value : float
        Magnitude of the distributed load (force per unit length). Negative
        for downward load. Default -10000 N/m.
    """
    L = 10.0  # Length of the beam
    # Define nodes (use Node(x, y, fixity, restrained_dofs))
    # Pin the two ends to avoid rigid-body motion
    # 33 evenly spaced nodes from x=0 to x=1 (end nodes pinned)
    node1 = Node(0.0 * L, 0.0, "pinned", ["x", "y"])  # left support
    node2 = Node(1 / 32 * L, 0.0, "free")
    node3 = Node(1 / 16 * L, 0.0, "free")
    node4 = Node(3 / 32 * L, 0.0, "free")
    node5 = Node(1 / 8 * L, 0.0, "free")
    node6 = Node(5 / 32 * L, 0.0, "free")
    node7 = Node(3 / 16 * L, 0.0, "free")
    node8 = Node(7 / 32 * L, 0.0, "free")
    node9 = Node(1 / 4 * L, 0.0, "free")
    node10 = Node(9 / 32 * L, 0.0, "free")
    node11 = Node(5 / 16 * L, 0.0, "free")
    node12 = Node(11 / 32 * L, 0.0, "free")
    node13 = Node(3 / 8 * L, 0.0, "free")
    node14 = Node(13 / 32 * L, 0.0, "free")
    node15 = Node(7 / 16 * L, 0.0, "free")
    node16 = Node(15 / 32 * L, 0.0, "free")
    node17 = Node(1 / 2 * L, 0.0, "free")
    node18 = Node(17 / 32 * L, 0.0, "free")
    node19 = Node(9 / 16 * L, 0.0, "free")
    node20 = Node(19 / 32 * L, 0.0, "free")
    node21 = Node(5 / 8 * L, 0.0, "free")
    node22 = Node(21 / 32 * L, 0.0, "free")
    node23 = Node(11 / 16 * L, 0.0, "free")
    node24 = Node(23 / 32 * L, 0.0, "free")
    node25 = Node(3 / 4 * L, 0.0, "free")
    node26 = Node(25 / 32 * L, 0.0, "free")
    node27 = Node(13 / 16 * L, 0.0, "free")
    node28 = Node(27 / 32 * L, 0.0, "free")
    node29 = Node(7 / 8 * L, 0.0, "free")
    node30 = Node(29 / 32 * L, 0.0, "free")
    node31 = Node(15 / 16 * L, 0.0, "free")
    node32 = Node(31 / 32 * L, 0.0, "free")
    node33 = Node(1.0 * L, 0.0, "pinned", ["x", "y"])  # right support

    # Material / section
    E = 25e9
    A = 0.003
    I = 0.0010125

    # Create beam members (element_type 'beam')
    m1 = Member2D("beam", node1, node2, E=E, A=A, I=I)
    m2 = Member2D("beam", node2, node3, E=E, A=A, I=I)
    m3 = Member2D("beam", node3, node4, E=E, A=A, I=I)
    m4 = Member2D("beam", node4, node5, E=E, A=A, I=I)
    m5 = Member2D("beam", node5, node6, E=E, A=A, I=I)
    m6 = Member2D("beam", node6, node7, E=E, A=A, I=I)
    m7 = Member2D("beam", node7, node8, E=E, A=A, I=I)
    m8 = Member2D("beam", node8, node9, E=E, A=A, I=I)
    m9 = Member2D("beam", node9, node10, E=E, A=A, I=I)
    m10 = Member2D("beam", node10, node11, E=E, A=A, I=I)
    m11 = Member2D("beam", node11, node12, E=E, A=A, I=I)
    m12 = Member2D("beam", node12, node13, E=E, A=A, I=I)
    m13 = Member2D("beam", node13, node14, E=E, A=A, I=I)
    m14 = Member2D("beam", node14, node15, E=E, A=A, I=I)
    m15 = Member2D("beam", node15, node16, E=E, A=A, I=I)
    m16 = Member2D("beam", node16, node17, E=E, A=A, I=I)
    m17 = Member2D("beam", node17, node18, E=E, A=A, I=I)
    m18 = Member2D("beam", node18, node19, E=E, A=A, I=I)
    m19 = Member2D("beam", node19, node20, E=E, A=A, I=I)
    m20 = Member2D("beam", node20, node21, E=E, A=A, I=I)
    m21 = Member2D("beam", node21, node22, E=E, A=A, I=I)
    m22 = Member2D("beam", node22, node23, E=E, A=A, I=I)
    m23 = Member2D("beam", node23, node24, E=E, A=A, I=I)
    m24 = Member2D("beam", node24, node25, E=E, A=A, I=I)
    m25 = Member2D("beam", node25, node26, E=E, A=A, I=I)
    m26 = Member2D("beam", node26, node27, E=E, A=A, I=I)
    m27 = Member2D("beam", node27, node28, E=E, A=A, I=I)
    m28 = Member2D("beam", node28, node29, E=E, A=A, I=I)
    m29 = Member2D("beam", node29, node30, E=E, A=A, I=I)
    m30 = Member2D("beam", node30, node31, E=E, A=A, I=I)
    m31 = Member2D("beam", node31, node32, E=E, A=A, I=I)
    m32 = Member2D("beam", node32, node33, E=E, A=A, I=I)
    members = [
        m1,
        m2,
        m3,
        m4,
        m5,
        m6,
        m7,
        m8,
        m9,
        m10,
        m11,
        m12,
        m13,
        m14,
        m15,
        m16,
        m17,
        m18,
        m19,
        m20,
        m21,
        m22,
        m23,
        m24,
        m25,
        m26,
        m27,
        m28,
        m29,
        m30,
        m31,
        m32,
    ]
    # Build DOF map and total DOFs
    dof_map = dof_index_map(members)
    # print("DOF Map:", dof_map) # Debug purpose
    total_free_dofs = sum(len(v) for v in dof_map["Free"].values())

    # Assemble global stiffness (sympy Matrix) and get free-DOF indices
    K_sym, dof_indices = assemblyKff(members, dof_map, total_dofs=total_free_dofs)

    # Convert sympy matrix to numeric numpy array
    K_numeric = np.array(K_sym.evalf(), dtype=np.float64)

    # `dof_indices` returned by assemblyKff maps node_name -> {dof: index}

    # Initialize global force vector (1D NumPy array)
    F_global = np.zeros((total_free_dofs,), dtype=np.float64)

    # Helper to add a member's equivalent nodal loads to the global vector
    for member in members:
        q = udl_value
        f_local = element_udl_equivalent(member, 0.25 * L, 0.75 * L, q)
        if all([float(val) == 0.0 for val in list(f_local)]):
            continue
        T = member.transformation_matrix()
        f_global_sym = T.T * f_local
        for i, dof_label in enumerate(member.dof_order):
            parts = dof_label.split("_")
            dof_name = parts[-1]
            node_name = "_".join(parts[:-1])
            if node_name in dof_indices and dof_name in dof_indices[node_name]:
                idx = dof_indices[node_name][dof_name]
                F_global[idx] += float(f_global_sym[i])

    # print("Global force vector (from UDL on members between node2 and node4):") # More debug purposes
    # print(F_global)

    # Solve for displacements
    displacements = linalg.solve(K_numeric, F_global)
    target_node = "Node_3.75_0.0"
    print(f"Nodal Displacement at {target_node}:    (32 beams)")
    if target_node in dof_indices:
        for dof_name, idx in dof_indices[target_node].items():
            print(f"{target_node} DOF {dof_name}: {displacements[idx]}")
    else:
        print(f"{target_node} not found in dof_indices")

    
    # bending moment at x=3.75 m
    member, xi, M = compute_bending_moment_at_x(members, dof_indices, displacements, 3.75)
    if M is None:
        print("Bending moment at x=3.75 m: not available for this mesh")
    else:
        print(f"Bending moment at x=3.75 m: {M} N*m (element {member.node_start.name}->{member.node_end.name}, xi={xi})")
    print(" ------------------------------------------------ ")


# 65 Node Beam
def milestone2p2_65nodes(udl_value: float = -10000.0):
    """Run example 5-node beam with a uniform distributed load applied
    from node2 (x=0.25) to node4 (x=0.75).

    Parameters
    ----------
    udl_value : float
        Magnitude of the distributed load (force per unit length). Negative
        for downward load. Default -10000 N/m.
    """
    L = 10.0  # Length of the beam
    # Define nodes (use Node(x, y, fixity, restrained_dofs))
    # Pin the two ends to avoid rigid-body motion
    # 65 evenly spaced nodes from x=0 to x=1 (end nodes pinned)
    node1 = Node(0.0 * L, 0.0, "pinned", ["x", "y"])  # left support
    node2 = Node(1 / 64 * L, 0.0, "free")
    node3 = Node(1 / 32 * L, 0.0, "free")
    node4 = Node(3 / 64 * L, 0.0, "free")
    node5 = Node(1 / 16 * L, 0.0, "free")
    node6 = Node(5 / 64 * L, 0.0, "free")
    node7 = Node(3 / 32 * L, 0.0, "free")
    node8 = Node(7 / 64 * L, 0.0, "free")
    node9 = Node(1 / 4 * L, 0.0, "free")
    node10 = Node(9 / 64 * L, 0.0, "free")
    node11 = Node(5 / 32 * L, 0.0, "free")
    node12 = Node(11 / 64 * L, 0.0, "free")
    node13 = Node(3 / 16 * L, 0.0, "free")
    node14 = Node(13 / 64 * L, 0.0, "free")
    node15 = Node(7 / 32 * L, 0.0, "free")
    node16 = Node(15 / 64 * L, 0.0, "free")
    node17 = Node(1 / 4 * L, 0.0, "free")
    node18 = Node(17 / 64 * L, 0.0, "free")
    node19 = Node(9 / 32 * L, 0.0, "free")
    node20 = Node(19 / 64 * L, 0.0, "free")
    node21 = Node(5 / 32 * L, 0.0, "free")
    node22 = Node(21 / 64 * L, 0.0, "free")
    node23 = Node(11 / 32 * L, 0.0, "free")
    node24 = Node(23 / 64 * L, 0.0, "free")
    node25 = Node(3 / 8 * L, 0.0, "free")
    node26 = Node(25 / 64 * L, 0.0, "free")
    node27 = Node(13 / 32 * L, 0.0, "free")
    node28 = Node(27 / 64 * L, 0.0, "free")
    node29 = Node(7 / 16 * L, 0.0, "free")
    node30 = Node(29 / 64 * L, 0.0, "free")
    node31 = Node(15 / 32 * L, 0.0, "free")
    node32 = Node(31 / 64 * L, 0.0, "free")
    node33 = Node(1 / 2 * L, 0.0, "free")
    node34 = Node(33 / 64 * L, 0.0, "free")
    node35 = Node(17 / 32 * L, 0.0, "free")
    node36 = Node(35 / 64 * L, 0.0, "free")
    node37 = Node(9 / 16 * L, 0.0, "free")
    node38 = Node(37 / 64 * L, 0.0, "free")
    node39 = Node(19 / 32 * L, 0.0, "free")
    node40 = Node(39 / 64 * L, 0.0, "free")
    node41 = Node(5 / 8 * L, 0.0, "free")
    node42 = Node(41 / 64 * L, 0.0, "free")
    node43 = Node(21 / 32 * L, 0.0, "free")
    node44 = Node(43 / 64 * L, 0.0, "free")
    node45 = Node(11 / 16 * L, 0.0, "free")
    node46 = Node(45 / 64 * L, 0.0, "free")
    node47 = Node(23 / 32 * L, 0.0, "free")
    node48 = Node(47 / 64 * L, 0.0, "free")
    node49 = Node(3 / 4 * L, 0.0, "free")
    node50 = Node(49 / 64 * L, 0.0, "free")
    node51 = Node(25 / 32 * L, 0.0, "free")
    node52 = Node(51 / 64 * L, 0.0, "free")
    node53 = Node(13 / 16 * L, 0.0, "free")
    node54 = Node(53 / 64 * L, 0.0, "free")
    node55 = Node(27 / 32 * L, 0.0, "free")
    node56 = Node(55 / 64 * L, 0.0, "free")
    node57 = Node(7 / 8 * L, 0.0, "free")
    node58 = Node(57 / 64 * L, 0.0, "free")
    node59 = Node(29 / 32 * L, 0.0, "free")
    node60 = Node(59 / 64 * L, 0.0, "free")
    node61 = Node(15 / 16 * L, 0.0, "free")
    node62 = Node(61 / 64 * L, 0.0, "free")
    node63 = Node(31 / 32 * L, 0.0, "free")
    node64 = Node(63 / 64 * L, 0.0, "free")
    node65 = Node(1.0 * L, 0.0, "pinned", ["x", "y"])  # right support
    # Material / section
    E = 25e9
    A = 10
    I = 0.0010125

    # Create beam members (element_type 'beam')
    m1 = Member2D("beam", node1, node2, E=E, A=A, I=I)
    m2 = Member2D("beam", node2, node3, E=E, A=A, I=I)
    m3 = Member2D("beam", node3, node4, E=E, A=A, I=I)
    m4 = Member2D("beam", node4, node5, E=E, A=A, I=I)
    m5 = Member2D("beam", node5, node6, E=E, A=A, I=I)
    m6 = Member2D("beam", node6, node7, E=E, A=A, I=I)
    m7 = Member2D("beam", node7, node8, E=E, A=A, I=I)
    m8 = Member2D("beam", node8, node9, E=E, A=A, I=I)
    m9 = Member2D("beam", node9, node10, E=E, A=A, I=I)
    m10 = Member2D("beam", node10, node11, E=E, A=A, I=I)
    m11 = Member2D("beam", node11, node12, E=E, A=A, I=I)
    m12 = Member2D("beam", node12, node13, E=E, A=A, I=I)
    m13 = Member2D("beam", node13, node14, E=E, A=A, I=I)
    m14 = Member2D("beam", node14, node15, E=E, A=A, I=I)
    m15 = Member2D("beam", node15, node16, E=E, A=A, I=I)
    m16 = Member2D("beam", node16, node17, E=E, A=A, I=I)
    m17 = Member2D("beam", node17, node18, E=E, A=A, I=I)
    m18 = Member2D("beam", node18, node19, E=E, A=A, I=I)
    m19 = Member2D("beam", node19, node20, E=E, A=A, I=I)
    m20 = Member2D("beam", node20, node21, E=E, A=A, I=I)
    m21 = Member2D("beam", node21, node22, E=E, A=A, I=I)
    m22 = Member2D("beam", node22, node23, E=E, A=A, I=I)
    m23 = Member2D("beam", node23, node24, E=E, A=A, I=I)
    m24 = Member2D("beam", node24, node25, E=E, A=A, I=I)
    m25 = Member2D("beam", node25, node26, E=E, A=A, I=I)
    m26 = Member2D("beam", node26, node27, E=E, A=A, I=I)
    m27 = Member2D("beam", node27, node28, E=E, A=A, I=I)
    m28 = Member2D("beam", node28, node29, E=E, A=A, I=I)
    m29 = Member2D("beam", node29, node30, E=E, A=A, I=I)
    m30 = Member2D("beam", node30, node31, E=E, A=A, I=I)
    m31 = Member2D("beam", node31, node32, E=E, A=A, I=I)
    m32 = Member2D("beam", node32, node33, E=E, A=A, I=I)
    m33 = Member2D("beam", node33, node34, E=E, A=A, I=I)
    m34 = Member2D("beam", node34, node35, E=E, A=A, I=I)
    m35 = Member2D("beam", node35, node36, E=E, A=A, I=I)
    m36 = Member2D("beam", node36, node37, E=E, A=A, I=I)
    m37 = Member2D("beam", node37, node38, E=E, A=A, I=I)
    m38 = Member2D("beam", node38, node39, E=E, A=A, I=I)
    m39 = Member2D("beam", node39, node40, E=E, A=A, I=I)
    m40 = Member2D("beam", node40, node41, E=E, A=A, I=I)
    m41 = Member2D("beam", node41, node42, E=E, A=A, I=I)
    m42 = Member2D("beam", node42, node43, E=E, A=A, I=I)
    m43 = Member2D("beam", node43, node44, E=E, A=A, I=I)
    m44 = Member2D("beam", node44, node45, E=E, A=A, I=I)
    m45 = Member2D("beam", node45, node46, E=E, A=A, I=I)
    m46 = Member2D("beam", node46, node47, E=E, A=A, I=I)
    m47 = Member2D("beam", node47, node48, E=E, A=A, I=I)
    m48 = Member2D("beam", node48, node49, E=E, A=A, I=I)
    m49 = Member2D("beam", node49, node50, E=E, A=A, I=I)
    m50 = Member2D("beam", node50, node51, E=E, A=A, I=I)
    m51 = Member2D("beam", node51, node52, E=E, A=A, I=I)
    m52 = Member2D("beam", node52, node53, E=E, A=A, I=I)
    m53 = Member2D("beam", node53, node54, E=E, A=A, I=I)
    m54 = Member2D("beam", node54, node55, E=E, A=A, I=I)
    m55 = Member2D("beam", node55, node56, E=E, A=A, I=I)
    m56 = Member2D("beam", node56, node57, E=E, A=A, I=I)
    m57 = Member2D("beam", node57, node58, E=E, A=A, I=I)
    m58 = Member2D("beam", node58, node59, E=E, A=A, I=I)
    m59 = Member2D("beam", node59, node60, E=E, A=A, I=I)
    m60 = Member2D("beam", node60, node61, E=E, A=A, I=I)
    m61 = Member2D("beam", node61, node62, E=E, A=A, I=I)
    m62 = Member2D("beam", node62, node63, E=E, A=A, I=I)
    m63 = Member2D("beam", node63, node64, E=E, A=A, I=I)
    m64 = Member2D("beam", node64, node65, E=E, A=A, I=I)
    members = [
        m1,
        m2,
        m3,
        m4,
        m5,
        m6,
        m7,
        m8,
        m9,
        m10,
        m11,
        m12,
        m13,
        m14,
        m15,
        m16,
        m17,
        m18,
        m19,
        m20,
        m21,
        m22,
        m23,
        m24,
        m25,
        m26,
        m27,
        m28,
        m29,
        m30,
        m31,
        m32,
        m33,
        m34,
        m35,
        m36,
        m37,
        m38,
        m39,
        m40,
        m41,
        m42,
        m43,
        m44,
        m45,
        m46,
        m47,
        m48,
        m49,
        m50,
        m51,
        m52,
        m53,
        m54,
        m55,
        m56,
        m57,
        m58,
        m59,
        m60,
        m61,
        m62,
        m63,
        m64,
    ]
    # Build DOF map and total DOFs
    dof_map = dof_index_map(members)
    # print("DOF Map:", dof_map) # Debug purpose
    total_free_dofs = sum(len(v) for v in dof_map["Free"].values())

    # Assemble global stiffness (sympy Matrix) and get free-DOF indices
    K_sym, dof_indices = assemblyKff(members, dof_map, total_dofs=total_free_dofs)

    # Convert sympy matrix to numeric numpy array
    K_numeric = np.array(K_sym.evalf(), dtype=np.float64)

    # `dof_indices` returned by assemblyKff maps node_name -> {dof: index}

    # Initialize global force vector (1D NumPy array)
    F_global = np.zeros((total_free_dofs,), dtype=np.float64)

    # Helper to add a member's equivalent nodal loads to the global vector
    for member in members:
        q = udl_value
        f_local = element_udl_equivalent(member, 0.25 * L, 0.75 * L, q)
        if all([float(val) == 0.0 for val in list(f_local)]):
            continue
        T = member.transformation_matrix()
        f_global_sym = T.T * f_local
        for i, dof_label in enumerate(member.dof_order):
            parts = dof_label.split("_")
            dof_name = parts[-1]
            node_name = "_".join(parts[:-1])
            if node_name in dof_indices and dof_name in dof_indices[node_name]:
                idx = dof_indices[node_name][dof_name]
                F_global[idx] += float(f_global_sym[i])

    # print("Global force vector (from UDL on members between node2 and node4):") # More debug purposes
    # print(F_global)

    # Solve for displacements
    displacements = linalg.solve(K_numeric, F_global)
    target_node = "Node_3.75_0.0"
    print(f"Nodal Displacement at {target_node}:    (64 beams)")
    if target_node in dof_indices:
        for dof_name, idx in dof_indices[target_node].items():
            print(f"{target_node} DOF {dof_name}: {displacements[idx]}")
    else:
        print(f"{target_node} not found in dof_indices")

    
    # bending moment at x=3.75 m
    member, xi, M = compute_bending_moment_at_x(members, dof_indices, displacements, 3.75)
    if M is None:
        print("Bending moment at x=3.75 m: not available for this mesh")
    else:
        print(f"Bending moment at x=3.75 m: {M} N*m (element {member.node_start.name}->{member.node_end.name}, xi={xi})")
    print(" ------------------------------------------------ ")


if __name__ == "__main__":
    milestone2p2_5nodes(udl_value=-2000.0)
    milestone2p2_9nodes(udl_value=-2000.0)
    milestone2p2_17nodes(udl_value=-2000.0)
    milestone2p2_33nodes(udl_value=-2000.0)
    milestone2p2_65nodes(udl_value=-2000.0)
    print("Analytical solution for comparison:")
    L = 10.0
    E = 25e9
    I = 0.0010125
    q = -2000.0
    x = 3.75
    print(f"Deflection at x={x} m: { (q * x**4) / (E * I) * (841/98304) } m")
    print(f"Bending moment at x={x} m: { (-11/128) * q * L**2 } N*m")
