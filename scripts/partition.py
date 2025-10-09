from typing import List, Sequence, Tuple, Union

import numpy as np

try:
    from sympy import Matrix as SympyMatrix
except Exception:  # pragma: no cover - sympy may not be installed in some envs
    SympyMatrix = None

# Use a generic alias to avoid forward-reference typing issues in some linters
MatrixLike = object


def _normalize_indices(
    n: int, constrained: Union[Sequence[bool], Sequence[int]]
) -> Tuple[List[int], List[int]]:
    """Return (free_indices, constrained_indices) from the user-provided constrained list.

    Args:
        n: total number of DOFs (matrix dimension)
        constrained: either a boolean mask of length n (True -> constrained)
                        or a sequence of integer indices that are constrained.

    Returns:
        (free_indices, constrained_indices) both sorted lists of ints
    """
    if (
        hasattr(constrained, "__len__")
        and len(constrained) == n
        and all(isinstance(x, bool) for x in constrained)
    ):
        constrained_idx = [i for i, v in enumerate(constrained) if v]
    else:
        # assume list of indices
        constrained_idx = sorted(int(i) for i in constrained)

    all_idx = list(range(n))
    free_idx = [i for i in all_idx if i not in constrained_idx]
    return free_idx, constrained_idx


def partition_stiffness(
    K: MatrixLike, constrained: Union[Sequence[bool], Sequence[int]]
):
    """Partition a square stiffness matrix into free/constrained blocks.

    The returned blocks follow the standard partitioning:
        K = [Kff  Kfc]
            [Kcf  Kcc]

    Args:
        K: square matrix (numpy.ndarray or sympy.Matrix) or a tuple whose
            first element is such a matrix (to accept outputs from preassembly).
        constrained: boolean mask (len == n) or list of constrained DOF indices.

    Returns:
        (Kff, Kfc, Kcf, Kcc, free_indices, constrained_indices)
        Submatrices have the same type as the input matrix (ndarray or sympy.Matrix).

    Raises:
        ValueError: if K is not square or indices invalid.
    """
    # Accept (K, note) or similar tuples returned by preassemblyTrusses
    if isinstance(K, tuple) or isinstance(K, list):
        if len(K) == 0:
            raise ValueError("Empty tuple provided for K")
        K = K[0]

    # Detect sympy
    is_sympy = SympyMatrix is not None and isinstance(K, SympyMatrix)
    is_numpy = isinstance(K, np.ndarray)
    if not (is_sympy or is_numpy):
        raise TypeError(
            "K must be a numpy.ndarray or sympy.Matrix (or a tuple whose first element is one of those)"
        )

    nrows, ncols = (K.shape[0], K.shape[1]) if is_numpy else (K.shape[0], K.shape[1])
    if nrows != ncols:
        raise ValueError("Stiffness matrix must be square")
    n = nrows

    free_idx, constrained_idx = _normalize_indices(n, constrained)

    if is_numpy:
        Kff = K[np.ix_(free_idx, free_idx)]
        Kfc = K[np.ix_(free_idx, constrained_idx)]
        Kcf = K[np.ix_(constrained_idx, free_idx)]
        Kcc = K[np.ix_(constrained_idx, constrained_idx)]
    else:
        # sympy Matrix supports indexing with lists
        Kff = K[free_idx, free_idx]
        Kfc = K[free_idx, constrained_idx]
        Kcf = K[constrained_idx, free_idx]
        Kcc = K[constrained_idx, constrained_idx]

    return Kff, Kfc, Kcf, Kcc, free_idx, constrained_idx


def partition_from_members(K: MatrixLike, members: Sequence, dofs_per_node: int = None):
    """Partition stiffness matrix K using restrained DOFs from `members`.

    This function mirrors the node ordering used in the preassembly functions:
    iterate members in order and assign node ids when a node is first seen.

    Args:
        K: square stiffness matrix (numpy.ndarray or sympy.Matrix) or a tuple as returned by preassembly.
        members: iterable of Member objects. Member objects must expose
            `node_start` and `node_end`. Node objects must expose
            `restrained_dofs` (list of strings) or `fixity` for fallback.
        dofs_per_node: optional int (2 for truss, 3 for beam). If None,
            it will be inferred from K.shape and number of unique nodes.

    Returns:
        Same as partition_stiffness: (Kff, Kfc, Kcf, Kcc, free_idx, constrained_idx)
    """
    # Accept (K, note) tuples
    if isinstance(K, (tuple, list)):
        if len(K) == 0:
            raise ValueError("Empty tuple provided for K")
        K = K[0]

    is_sympy = SympyMatrix is not None and isinstance(K, SympyMatrix)
    is_numpy = isinstance(K, np.ndarray)
    if not (is_sympy or is_numpy):
        raise TypeError(
            "K must be a numpy.ndarray or sympy.Matrix (or a tuple whose first element is one of those)"
        )

    n = K.shape[0] if is_numpy else K.shape[0]

    # Build ordered node_id_map like preassembly: assign base DOF index per node
    node_id_map = {}
    next_id = 0
    unique_nodes = []
    for member in members:
        for node in (member.node_start, member.node_end):
            if node not in node_id_map:
                node_id_map[node] = None
                unique_nodes.append(node)

    num_nodes = len(unique_nodes)
    if num_nodes == 0:
        raise ValueError("No nodes found in members")

    # Infer dofs_per_node if not provided
    if dofs_per_node is None:
        if n % num_nodes != 0:
            raise ValueError(
                "Cannot infer dofs_per_node: K size not divisible by number of nodes. Provide dofs_per_node explicitly."
            )
        dofs_per_node = n // num_nodes

    # Assign base indices (0, dofs_per_node, 2*dofs_per_node, ...)
    next_id = 0
    for node in unique_nodes:
        node_id_map[node] = next_id
        next_id += dofs_per_node

    # Determine constrained DOF indices from node.restrained_dofs or node.fixity
    constrained_indices = []
    # mapping of local DOF names to indices
    local_map = {"x": 0, "y": 1, "rz": 2, "r": 2}
    for node, base in node_id_map.items():
        restrained = []
        if hasattr(node, "restrained_dofs") and node.restrained_dofs:
            restrained = node.restrained_dofs
        else:
            # fallback to fixity conventions
            f = getattr(node, "fixity", None)
            if f == "fixed":
                restrained = ["x", "y", "rz"]
            elif f == "pinned":
                restrained = ["x", "y"]
            else:
                restrained = []

        for name in restrained:
            if name not in local_map:
                # ignore unknown names
                continue
            local_idx = local_map[name]
            if local_idx >= dofs_per_node:
                # local DOF doesn't exist for this element type
                continue
            constrained_indices.append(base + local_idx)

    constrained_indices = sorted(set(constrained_indices))

    # Delegate to partition_stiffness
    Kff, Kfc, Kcf, Kcc, free_idx, constrained_idx = partition_stiffness(
        K, constrained_indices
    )

    # Build human-readable DOF labels in the same node ordering used above.
    # Node numbering will be 1-based according to the unique_nodes order.
    # Direction names per local DOF index:
    dir_names_full = ["x", "y", "rz"]
    dir_names = dir_names_full[:dofs_per_node]

    # Create index -> label mapping
    index_to_label = {}
    for node_number, node in enumerate(unique_nodes, start=1):
        base = node_id_map[node]
        for local_idx, dname in enumerate(dir_names):
            index_to_label[base + local_idx] = f"Node{node_number}_{dname}"

    free_labels = [index_to_label[i] for i in free_idx]
    constrained_labels = [index_to_label[i] for i in constrained_idx]

    return (
        Kff,
        Kfc,
        Kcf,
        Kcc,
        free_idx,
        constrained_idx,
        free_labels,
        constrained_labels,
    )


if __name__ == "__main__":
    # Quick self-test: small numpy and sympy examples
    print("Running quick partition tests...")
    import numpy as _np

    K_np = _np.arange(16).reshape(4, 4).astype(float)
    # constrain DOFs 1 and 3
    Kff, Kfc, Kcf, Kcc, free, cons = partition_stiffness(K_np, [1, 3])
    print("\nNumpy K:")
    print(K_np)
    print("free indices:", free, "constrained indices:", cons)
    print("Kff:\n", Kff)
    print("Kfc:\n", Kfc)

    if SympyMatrix is not None:
        from sympy import Matrix as _M

        K_sym = _M([[1, 2, 3], [2, 5, 6], [3, 6, 9]])
        Kff, Kfc, Kcf, Kcc, free, cons = partition_stiffness(K_sym, [0])
        print("\nSympy K:")
        print(K_sym)
        print("free indices:", free, "constrained indices:", cons)
        print("Kff:\n", Kff)
    else:
        print("\nSympy not available; skipping sympy self-test")
