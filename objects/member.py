from sympy import Matrix, cos, sin, symbols, simplify, sqrt, atan2
import numpy as np


class Member2D:
    """Represents a structural member (element) connecting two nodes in 2D space.

    A Member stores material and geometric properties and computes derived
    quantities such as length and orientation. It can produce local and global
    stiffness matrices for truss and beam formulations.

    Attributes
    ----------
    element_type : {'truss', 'beam'}
        Element formulation used for stiffness calculations.
    node_start : Node
        Start node; must provide ``x``, ``y``, and ``fixity``.
    node_end : Node
        End node; must provide ``x``, ``y``, and ``fixity``.
    E : float
        Young's modulus of the member material.
    A : float
        Cross-sectional area.
    I : float
        Second moment of area (moment of inertia) of the cross-section.
    length : float
        Euclidean length computed from the node coordinates.
    angle : float
        Angle (radians) of the member measured from the global x-axis.

    Examples
    --------
    Create a pinned truss member between two nodes::

        m = Member('truss', node1, node2, E=210e9, A=0.01, I=0.0)

    Create a beam element::

        b = Member('beam', node3, node4, E=200e9, A=0.02, I=8.3e-6)

    Notes
    -----
    - For truss elements the code currently validates that node fixities are
        compatible with a pinned/roller/free connection at each end.
    - The class docstrings and method docstrings should be kept consistent
        so that Sphinx/autodoc can generate usable API documentation.
    """

    def __init__(self, element_type, node_start, node_end, E, A, I):
        # Initialize member with innate properties.
        # Normalize element_type to a lowercase string so callers may pass
        # values like 'Beam' or 'beam' interchangeably.
        if isinstance(element_type, str):
            self.element_type = element_type.lower()
        else:
            self.element_type = element_type
        self.node_start = node_start  # Node start and end are objects of class node.
        self.node_end = (
            node_end  # Node objects are placeholders for coordinates and fixity
        )
        self.E = E
        self.A = A
        self.I = I
        self.nodes = [node_start.name, node_end.name]
        # Length and angle are not given, but calculated from node coordinates.
        dx = self.node_end.x - self.node_start.x
        dy = self.node_end.y - self.node_start.y
        # Use SymPy expressions for length and angle so numeric and symbolic inputs both work
        self.length = simplify(sqrt(dx**2 + dy**2))
        self.angle = simplify(atan2(dy, dx))
        self.name = f"Member_{self.node_start.name}_{self.node_end.name}"

        # Validate truss members
        if self.element_type == "truss":
            if self.node_start.fixity not in (
                "pinned",
                "free",
                "roller",
            ) or self.node_end.fixity not in ("pinned", "free", "roller"):
                raise ValueError(
                    "Truss members must have pinned/roller supports or be free to rotate at both ends. \n Check node fixities."
                )

        if self.element_type == "truss":
            self.dof_order = [
                f"{self.node_start.name}_x",
                f"{self.node_start.name}_y",
                f"{self.node_end.name}_x",
                f"{self.node_end.name}_y",
            ]
        elif self.element_type == "beam":
            # use 'rotation' to match DOF naming in dof_index_map and other scripts
            self.dof_order = [
                f"{self.node_start.name}_x",
                f"{self.node_start.name}_y",
                f"{self.node_start.name}_rotation",
                f"{self.node_end.name}_x",
                f"{self.node_end.name}_y",
                f"{self.node_end.name}_rotation",
            ]

    def local_stiffness_matrix(self):
        """Compute the local stiffness matrix for this member.

        The returned matrix is expressed in the element's local coordinate
        system. The exact size and structure depend on ``element_type``:
        - ``'truss'``: returns a 4x4 stiffness matrix for a 2D truss
            formulation (axial stiffness with zeros for transverse DOFs).
        - ``'beam'``: returns a 4x4 bending/axial stiffness matrix for a
            simplified 2D beam formulation (assembled for the element DOFs).

        Returns
        -------
        sympy.Matrix
            Local stiffness matrix (Symbolic SymPy Matrix; shape depends on ``element_type``).

        Raises
        ------
        ValueError
            If ``element_type`` is not recognized.

        Examples
        --------
        >>> k_local = member.local_stiffness_matrix()
        """
        L = self.length
        E = self.E
        A = self.A
        I = self.I

        if self.element_type == "truss":
            self.dof_order = [
                f"{self.node_start.name}_x",
                f"{self.node_start.name}_y",
                f"{self.node_end.name}_x",
                f"{self.node_end.name}_y",
            ]
            k = simplify((E * A) / L)
            return Matrix([[k, 0, -k, 0], [0, 0, 0, 0], [-k, 0, k, 0], [0, 0, 0, 0]])
        elif self.element_type == "beam":
            # keep naming consistent with preassembly.dof_index_map which uses 'rotation'
            self.dof_order = [
                f"{self.node_start.name}_x",
                f"{self.node_start.name}_y",
                f"{self.node_start.name}_rotation",
                f"{self.node_end.name}_x",
                f"{self.node_end.name}_y",
                f"{self.node_end.name}_rotation",
            ]
            k = simplify((E * I) / (L**3))
            return Matrix(
                [
                    [E * A / L, 0, 0, -E * A / L, 0, 0],
                    [0, 12 * k, 6 * L * k, 0, -12 * k, 6 * L * k],
                    [0, 6 * L * k, 4 * L * L * k, 0, -6 * L * k, 2 * L * L * k],
                    [-E * A / L, 0, 0, E * A / L, 0, 0],
                    [0, -12 * k, -6 * L * k, 0, 12 * k, -6 * L * k],
                    [0, 6 * L * k, 2 * L * L * k, 0, -6 * L * k, 4 * L * L * k],
                ]
            )
        else:
            raise ValueError("Unknown element type")

    def transformation_matrix(self):
        """Return the transformation matrix mapping local -> global coordinates.

        The transformation matrix ``T`` is constructed from the element angle
        (cosine ``c`` and sine ``s``). For a given local vector ``u_local``,
        the global vector is ``u_global = T @ u_local``. The size of ``T``
        depends on ``element_type``:
        - ``'truss'``: 4x4 matrix for 2D truss DOFs.
        - ``'beam'``: 6x6 matrix for 2D beam DOFs (including rotations).

        DOF ordering
        ------------
        The transformation assumes the following local/global DOF ordering:
        - Truss (2D): [u1, v1, u2, v2]
        - Beam (2D):  [u1, v1, r1, u2, v2, r2]

        Returns
        -------
        sympy.Matrix
            Transformation matrix (Symbolic SymPy Matrix; shape depends on ``element_type``).

        Examples
        --------
        >>> T = member.transformation_matrix()
        """
        # Use sympy trig functions so matrix entries can be symbolic
        c = cos(self.angle)
        s = sin(self.angle)

        if self.element_type == "truss":  # 2D truss element transformation matrix
            return Matrix([[c, s, 0, 0], [-s, c, 0, 0], [0, 0, c, s], [0, 0, -s, c]])
        elif self.element_type == "beam":  # 2D beam element transformation matrix
            return Matrix(
                [
                    [c, s, 0, 0, 0, 0],
                    [-s, c, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, c, s, 0],
                    [0, 0, 0, -s, c, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            )
        else:
            raise ValueError("Unknown element type")  # Exception handling

    def global_stiffness_matrix(self):
        """Assemble and return the global stiffness matrix for this member.

        The global stiffness is computed using the standard similarity
        transform::

            K_global = T.T @ K_local @ T

        where ``K_local`` is the element stiffness in local coordinates and
        ``T`` is the transformation matrix returned by
        :meth:`transformation_matrix`.

        Returns
        -------
        sympy.Matrix
            Global stiffness matrix (Symbolic SymPy Matrix; shape compatible with system DOFs).

        Examples
        --------
        >>> K_global = member.global_stiffness_matrix()
        """
        k_local = self.local_stiffness_matrix()  # Local stiffness matrix.
        T = self.transformation_matrix()  # Transformation matrix.
        return simplify(T.T * k_local * T)


class Member3D:
    def __init__(
        self,
        element_type,
        node_start,
        node_end,
        E,
        A,
        Iyy,
        Izz,
        poisson,
        J,
        Comment: str = "",
    ):
        self.element_type = element_type
        self.node_start = node_start
        self.node_end = node_end
        self.E = E
        self.A = A
        self.Iyy = Iyy
        self.Izz = Izz
        self.poisson = poisson
        self.J = J
        self.Comment = Comment  # Just in case you'd like to add a comment to the member. e.g. "W18x35"
        # Length and angle are not given, but calculated from node coordinates.
        dx = self.node_end.x - self.node_start.x
        dy = self.node_end.y - self.node_start.y
        dz = self.node_end.z - self.node_start.z
        # Shear modulus is a calculated value
        self.G = self.E / (2 * (1 + self.poisson))
        # Use SymPy expressions for length and angle so numeric and symbolic inputs both work
        self.length = simplify(sqrt(dx**2 + dy**2 + dz**2))
        # Angle calculations in 3D would require more complex handling (direction cosines)
        # Direction cosines and local orthonormal basis for the element axis
        if self.length == 0:
            raise ValueError("Zero-length member; cannot compute direction cosines.")

        self.Lambda = None  # Placeholder for future use

    def local_stiffness_matrix(self):
        """Compute the 12x12 local stiffness matrix for a 3D beam element.

        The local coordinate system is assumed with the element axis along
        local x. DOF ordering per node is::

            [u, v, w, rx, ry, rz]

        and the full element DOF ordering is::

            [u1, v1, w1, rx1, ry1, rz1, u2, v2, w2, rx2, ry2, rz2]

        Returns
        -------
        sympy.Matrix
            12x12 local stiffness matrix (symbolic-friendly using SymPy expressions)
        """
        L = self.length
        E = self.E
        A = self.A
        Iyy = self.Iyy
        Izz = self.Izz
        G = self.G
        J = self.J

        # Create zero 12x12 matrix
        K = Matrix([[0] * 12 for _ in range(12)])

        # Axial stiffness (u DOFs: indices 0 and 6)
        k_axial = simplify(E * A / L)
        K[0, 0] = k_axial
        K[0, 6] = -k_axial
        K[6, 0] = -k_axial
        K[6, 6] = k_axial

        # Torsional stiffness (rx DOFs: indices 3 and 9)
        k_torsion = simplify(G * J / L)
        K[3, 3] = k_torsion
        K[3, 9] = -k_torsion
        K[9, 3] = -k_torsion
        K[9, 9] = k_torsion

        # Bending about local z (produces displacement in local y -> DOFs v and rz)
        k_bz = simplify(E * Izz / (L**3))
        # Local mapping for [v1, rz1, v2, rz2] -> indices [1,5,7,11]
        inds_bz = [1, 5, 7, 11]
        vals_bz = [
            [12, 6 * L, -12, 6 * L],
            [6 * L, 4 * L * L, -6 * L, 2 * L * L],
            [-12, -6 * L, 12, -6 * L],
            [6 * L, 2 * L * L, -6 * L, 4 * L * L],
        ]
        for i in range(4):
            for j in range(4):
                K[inds_bz[i], inds_bz[j]] = simplify(vals_bz[i][j] * k_bz)

        # Bending about local y (produces displacement in local z -> DOFs w and ry)
        k_by = simplify(E * Iyy / (L**3))
        # Local mapping for [w1, ry1, w2, ry2] -> indices [2,4,8,10]
        inds_by = [2, 4, 8, 10]
        vals_by = [
            [12, -6 * L, -12, -6 * L],
            [-6 * L, 4 * L * L, 6 * L, 2 * L * L],
            [-12, 6 * L, 12, 6 * L],
            [-6 * L, 2 * L * L, 6 * L, 4 * L * L],
        ]
        # Note: signs arranged so consistent sign conventions for rotations are used
        for i in range(4):
            for j in range(4):
                K[inds_by[i], inds_by[j]] = simplify(vals_by[i][j] * k_by)

        return simplify(K)

    def transformation_matrix(self):
        """Return the 12x12 transformation matrix mapping local -> global for 3D element.

        The transformation matrix ``T`` is constructed from the direction
        cosines of the element axis. This is calculated earlier

            [ Λ   0  0  0 ]
            [ 0   Λ  0  0 ]
            [ 0   0  Λ  0 ]
            [ 0   0  0  Λ ]

        and the full 12x12 element transformation is block-diagonal with the
        two 6x6 node transforms.
        """
        Lambda = self.Gamma
        Zed = np.zeros((3, 3))
        G_1 = np.hstack((Lambda, Zed, Zed, Zed))
        G_2 = np.hstack((Zed, Lambda, Zed, Zed))
        G_3 = np.hstack((Zed, Zed, Lambda, Zed))
        G_4 = np.hstack((Zed, Zed, Zed, Lambda))
        Gamma = np.vstack((G_1, G_2, G_3, G_4))

        return simplify(Gamma)

    def global_stiffness_matrix(self):
        """Assemble and return the global 12x12 stiffness matrix for this member.

        Uses the standard similarity transform K_global = T.T * K_local * T.
        """
        k_local = self.local_stiffness_matrix()
        T = self.transformation_matrix()
        return simplify(T.T * k_local * T)
