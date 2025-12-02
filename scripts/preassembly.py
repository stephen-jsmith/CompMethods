from objects import *
from sympy import sqrt, symbols, Matrix, zeros, pi
import numpy as np

def dof_index_map(members):
    """
    Construct a mapping of degrees of freedom (DOFs) for each node found in a collection of members.

    Parameters
    ----------
    members : iterable
        An iterable of member objects. Each member is expected to have the attributes
        `node_start` and `node_end`, which refer to node objects. Each node object must
        provide:
          - `name` : hashable identifier used as the dictionary key (e.g. string)
          - `restrained_dofs` : either None or an iterable of DOF names (e.g. ["x", "y"])

    Returns
    -------
    dict
        A dictionary with two keys: "Free" and "Fixed". Each value is a mapping from
        node name -> list of DOF names:
          - "Free"[node_name] : list of DOFs available (unrestrained) for the node.
          - "Fixed"[node_name] : list of DOFs that are restrained for the node.
        Behavior:
          - If a node's `restrained_dofs` is None, the node is treated as fully free and
            "Free"[name] will be ["x", "y", "rotation"]. The node will not appear in "Fixed".
          - If `restrained_dofs` is provided, those DOFs are copied into "Fixed"[name] and
            removed from the default DOF set ["x", "y", "rotation"] to form "Free"[name].
          - Nodes appearing multiple times across members are deduplicated (unique by object),
            but the order of processing is not guaranteed.

    Notes
    -----
    - The function returns labels of DOFs only; it does not assign numeric indices.
    - If `restrained_dofs` contains names outside the default set ["x", "y", "rotation"],
      they will still be recorded in "Fixed" and omitted from "Free".
    - The function expects well-formed member/node objects and will raise AttributeError
      if required attributes are missing.
    - Complexity is linear in the number of member endpoints processed.

    Examples
    --------
    Assuming minimal Node/Member objects:
        node_a.restrained_dofs = None
        node_b.restrained_dofs = ["x", "y"]
        result = dof_index_map([member_with_nodes(node_a, node_b), ...])
        # result["Free"]["a"] -> ["x", "y", "rotation"]
        # result["Fixed"]["b"] -> ["x", "y"]
        # result["Free"]["b"] -> ["rotation"]
    """
    node_list = []
    for i in members:
        node_list.append(i.node_start)
        node_list.append(i.node_end)
    node_list = list(set(node_list))
    dof = {"Free": {}, "Fixed": {}}
    for node in node_list:
        if node.restrained_dofs == None:
            dof["Free"][node.name] = ["x", "y", "rotation"]
        else:
            dof["Fixed"][node.name] = [rdof for rdof in node.restrained_dofs]
            dof["Free"][node.name] = [
                dof for dof in ["x", "y", "rotation"] if dof not in node.restrained_dofs
            ]
    return dof

def preassemblyBeams(members, numeric: bool = False, subs: dict = None):
    """Assemble beam-element global stiffness matrix.

    This function performs assembly of element stiffness matrices for beam
    elements into a global stiffness matrix. By default the assembly is
    symbolic and returns a :class:`sympy.Matrix`. Optionally the caller can
    request a numeric result (``numeric=True``) and provide a substitution
    dictionary ``subs`` to evaluate any symbolic parameters.

    Parameters
    ----------
    members : list[Member]
        Iterable of ``Member`` objects. Each member must implement
        :meth:`Member.global_stiffness_matrix` which returns a
        ``sympy.Matrix`` (symbolic) or numeric-like matrix. Members are
        expected to expose ``node_start`` and ``node_end`` attributes where
        nodes provide consistent DOF ordering.
    numeric : bool, optional
        If True, attempt to evaluate the assembled global matrix numerically
        and return a NumPy ``ndarray``. Default is False (return symbolic
        ``sympy.Matrix``).
    subs : dict, optional
        Mapping of SymPy symbols to numeric values used when ``numeric=True``.
        All free symbols in the assembled matrix must be present in ``subs``
        or an error will be raised.

    Returns
    -------
    sympy.Matrix or numpy.ndarray
        The assembled global stiffness matrix. Returns a ``sympy.Matrix``
        when ``numeric=False``. Returns a NumPy ``ndarray`` when
        ``numeric=True``.

    Raises
    ------
    ValueError
        If ``numeric=True`` but the assembled matrix still contains free
        symbols not provided in ``subs``.

    Notes
    -----
    - The function assumes 3 DOFs per node (u, v, rotation) for beam
        elements. The ordering used during assembly follows the members'
        element-level DOF ordering.
    - Symbolic assembly is convenient for deriving analytic expressions and
        performing symbolic manipulations. Numeric assembly (NumPy arrays) is
        significantly faster for large problems and direct numerical solvers.

    Examples
    --------
    >>> K_sym = preassemblyBeams(members)
    >>> K_num = preassemblyBeams(members, numeric=True, subs={E:210e9, A:0.01})
    """
    # Determine total DOFs
    total_dofs = 0
    node_id_map = {}
    current_id = 0
    for member in members:
        for node in [member.node_start, member.node_end]:
            if node not in node_id_map:
                node_id_map[node] = current_id
                current_id += 3  # Assuming 3 DOFs per node (x, y, rotation)
    total_dofs = current_id

    # Initialize global stiffness matrix as a sympy Matrix (symbolic assembly)
    K_sym = Matrix.zeros(total_dofs, total_dofs)

    # Assemble global stiffness matrix symbolically
    for member in members:
        k_global = member.global_stiffness_matrix()

        start_id = node_id_map[member.node_start]
        end_id = node_id_map[member.node_end]

        dof_indices = [
            start_id,
            start_id + 1,
            start_id + 2,
            end_id,
            end_id + 1,
            end_id + 2,
        ]

        for i in range(len(dof_indices)):
            for j in range(len(dof_indices)):
                K_sym[dof_indices[i], dof_indices[j]] += k_global[i, j]

    if numeric:
        # Prepare substitution dict and ensure all free symbols are provided
        subs_dict = {} if subs is None else subs
        remaining = K_sym.free_symbols - set(subs_dict.keys())
        if remaining:
            raise ValueError(
                f"Cannot evaluate numerically: remaining free symbols {remaining}. "
                "Provide numeric values for these symbols in the `subs` dict or use numeric node coordinates."
            )
        # Safe to evaluate numerically
        K_num = np.array(K_sym.evalf(subs=subs_dict)).astype(np.float64)
        return K_num

    return K_sym


def preassemblyTrusses(members, numeric: bool = False, subs: dict = None):
    """Assemble truss-element global stiffness matrix.

    This function assembles 2D truss element stiffness matrices into a
    global stiffness matrix. By default the assembly is symbolic and the
    function returns a :class:`sympy.Matrix`. Optionally callers can request
    numeric evaluation using ``numeric=True`` and provide a substitution
    dictionary ``subs`` that assigns numeric values to all free symbols.

    Parameters
    ----------
    members : list[Member]
        Iterable of ``Member`` objects. Each member must implement
        :meth:`Member.global_stiffness_matrix` which returns a
        ``sympy.Matrix`` (symbolic) or a numeric matrix. Nodes must expose
        ``node_start`` and ``node_end`` attributes.
    numeric : bool, optional
        If True, evaluate the symbolic global stiffness matrix numerically
        and return a NumPy ``ndarray``. Default is False.
    subs : dict, optional
        Dictionary mapping SymPy symbols to numeric values used during
        evaluation when ``numeric=True``. All free symbols must be provided
        or a ``ValueError`` is raised.

    Returns
    -------
    sympy.Matrix or (numpy.ndarray, str)
        If ``numeric=False`` returns a ``sympy.Matrix`` representing the
        assembled global stiffness matrix. If ``numeric=True`` returns a
        tuple ``(K_num, note)`` where ``K_num`` is a NumPy array with the
        numeric matrix and ``note`` is an informational string describing
        the substitution performed.

    Raises
    ------
    ValueError
        If ``numeric=True`` but the assembled matrix still contains free
        symbols not supplied in ``subs``.

    Notes
    -----
    - The function assumes 2 DOFs per node (u, v) for truss elements. The
        ordering used during assembly follows the members' element-level DOF
        ordering.
    - For large models prefer numeric evaluation before calling linear
        algebra solvers for performance.

    Examples
    --------
    >>> K_sym = preassemblyTrusses(members)
    >>> K_num, note = preassemblyTrusses(members, numeric=True, subs={E:210e9, A:0.01})
    """
    # Determine total DOFs
    total_dofs = 0
    node_id_map = {}
    current_id = 0
    hw_output = {}  # For homework assignment purposes
    for member in members:
        for node in [member.node_start, member.node_end]:
            if node not in node_id_map:
                node_id_map[node] = current_id
                current_id += 2  # Assuming 2 DOFs per node (x, y)
    total_dofs = current_id

    # Initialize global stiffness matrix symbolically
    K_sym = Matrix.zeros(total_dofs, total_dofs)

    # Assemble global stiffness matrix symbolically
    for member in members:
        k_global = member.global_stiffness_matrix()

        start_id = node_id_map[member.node_start]
        end_id = node_id_map[member.node_end]

        dof_indices = [start_id, start_id + 1, end_id, end_id + 1]

        for i in range(len(dof_indices)):
            for j in range(len(dof_indices)):
                K_sym[dof_indices[i], dof_indices[j]] += k_global[i, j]

    hw_output["n_nodes"] = len(node_id_map)
    hw_output["n_dofs"] = total_dofs
    hw_output["N_elements"] = len(members)
    hw_output["coords"] = [(node.x, node.y) for node in node_id_map.keys()]

    if numeric:
        subs_dict = {} if subs is None else subs
        remaining = K_sym.free_symbols - set(subs_dict.keys())
        if remaining:
            raise ValueError(
                f"Cannot evaluate numerically: remaining free symbols {remaining}. "
                "Provide numeric values for these symbols in the `subs` dict or use numeric node coordinates."
            )
        K_num = np.array(K_sym.evalf(subs=subs_dict)).astype(np.float64)
        note = f"Numeric assembly performed with substitution: {list(subs_dict.keys())}"
        return K_num, note

    return K_sym, 'No numeric evaluation requested.'


def preassemblyGen(members: list, numeric: bool = False, subs: dict = None):
    """Assemble global stiffness matrix for mixed truss and beam elements.

    This function assembles both truss and beam element stiffness matrices
    into a single global stiffness matrix. By default the assembly is symbolic
    and returns a :class:`sympy.Matrix`. Optionally callers can request
    numeric evaluation using ``numeric=True`` and provide a substitution
    dictionary ``subs`` that assigns numeric values to all free symbols.

    Parameters
    ----------
    members : list[Member]
        Iterable of ``Member`` objects. Each member must implement
        :meth:`Member.global_stiffness_matrix` which returns a
        ``sympy.Matrix`` (symbolic) or a numeric matrix. Nodes must expose
        ``node_start`` and ``node_end`` attributes."""
    # Determine DOF ordering
    dof_list = []
    for i in members:
        dof_list.extend(i.dof_order)
    dof_list = list(set(dof_list))  # Unique DOF list
    total_dofs = len(dof_list)
    stiffness_dict = {}
    for member in members:
        k_global = member.global_stiffness_matrix()
        stiffness_dict[member.name] = k_global
    print(stiffness_dict)
    print(f"Total DOFs: {total_dofs}")
    k = np.zeros((total_dofs, total_dofs))
    dof_index_map = {dof: idx for idx, dof in enumerate(dof_list)}
    '''for member in members:
        k_global = member.global_stiffness_matrix()
        dof_order = member.dof_order
        for i in range(len(dof_order)):
            for j in range(len(dof_order)):
                global_i = dof_index_map[dof_order[i]]
                global_j = dof_index_map[dof_order[j]]
                k[global_i, global_j] += k_global[i, j]'''
    return k, dof_list

# Example usage:
if __name__ == "__main__":
    E, A, I = symbols("E A I")

    # Define nodes (square: (0,0), (1,0), (1,1), (0,1))
    n1 = Node(0, 0, 'pinned', ['x', 'y'])
    n2 = Node(1, 0, 'pinned', ['x', 'y'])
    n3 = Node(1, 1, 'free')
    n4 = Node(0, 1, 'free')

    # Define members (edges of square + diagonal through center)
    members = [
        Member2D("beam", n1, n2, E=E, A=A, I=I),  # bottom
        Member2D("beam", n2, n3, E=E, A=A, I=I),  # right
        Member2D("beam", n3, n4, E=E, A=A, I=I),  # top
        Member2D("beam", n4, n1, E=E, A=A, I=I),  # left
        Member2D("beam", n1, n3, E=E, A=A, I=I),  # diagonal
    ]
    

    # Assign properties if needed (e.g., E, A) here

    K = preassemblyGen(members)
    # preassemblyBeams returns a single object (sympy.Matrix or numpy.ndarray);
    # set an informational note for the example to mirror the truss function behavior
    note = "Numeric evaluation not requested; returned symbolic matrix." if not isinstance(K, np.ndarray) else "Numeric matrix returned."
    print("---------------------------------")
    print("Global stiffness matrix K:")
    for row in K.tolist():
        print(row)
    print("\nNote:")
    print(note)
    print("---------------------------------")
    print('Example numeric evaluation:')
    numeric_subs = {E: 210e9, A: 0.01, I: 1e-6}
    K_num, dof_list = preassemblyGen(members, numeric=True, subs=numeric_subs)
    print(f"DOF List: {dof_list}")
    print("\nNumeric global stiffness matrix K:")
    for row in K_num.tolist():
        print(row)
    numeric_note = "Numeric evaluation performed with substitution: " + ", ".join(numeric_subs.keys())
    print("\nNote:")
    print(numeric_note)
