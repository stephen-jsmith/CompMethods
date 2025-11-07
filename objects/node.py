from typing import List, Literal, Optional, Union
try:
    from sympy import Basic as SympyBasic
except Exception:
    SympyBasic = ()


class Node:
    """Represents a model node in 2D space.

    Nodes define geometry and boundary conditions for structural models. Each
    node has coordinates, a support condition (``fixity``), and an optional
    list of restrained degrees of freedom.

    Attributes
    ----------
    x : float | sympy expression
        X-coordinate in global units (e.g., meters) or a SymPy symbolic expression.
    y : float | sympy expression
        Y-coordinate in global units (e.g., meters) or a SymPy symbolic expression.
    fixity : {'fixed', 'free', 'pinned', 'roller'}
        Support condition. Use the documented string values for consistency.
    restrained_dofs : list[str]
        Optional list of restrained DOF names (e.g., ["x", "y", "rz"]). If
        omitted, the node is considered free.

    Examples
    --------
    Create a pinned node with x and y restrained::

        node = Node(0.0, 0.0, fixity='pinned', restrained_dofs=['x', 'y'])

    Create a free node::

        node2 = Node(1.0, 1.0, fixity='free')

    Notes
    -----
    - The class does not currently enforce consistency between ``fixity`` and
      ``restrained_dofs``; validation can be added if stricter checks are
      desired.
    """

    def __init__(
        self,
    x: Union[float, int, object],
    y: Union[float, int, object],
        fixity: Literal['fixed', 'free', 'pinned', 'roller'],
        restrained_dofs: Optional[List[str]] = None,
    z: Union[float, int, object] = None
    ) -> None:
        self.name = f"Node_{x}_{y}"  # Simple naming convention; can be customized
        # Basic type and value validation. Accept numeric types or SymPy symbolic
        # expressions (Sympy Basic) so the project can support symbolic geometry.
        is_numeric_x = isinstance(x, (int, float))
        is_numeric_y = isinstance(y, (int, float))
        is_symbolic_x = isinstance(x, SympyBasic)
        is_symbolic_y = isinstance(y, SympyBasic)
        is_numeric_z = isinstance(z, (int, float))
        is_symbolic_z = isinstance(z, SympyBasic)

        if not (is_numeric_x or is_symbolic_x):
            raise TypeError('x must be a number or a sympy expression')
        if not (is_numeric_y or is_symbolic_y):
            raise TypeError('y must be a number or a sympy expression')
        if z is not None and not (is_numeric_z or is_symbolic_z):
            raise TypeError('z must be a number, a sympy expression, or None')

        allowed_fixities = ('fixed', 'free', 'pinned', 'roller')
        if fixity not in allowed_fixities:
            raise ValueError(f"fixity must be one of {allowed_fixities}; got {fixity!r}")

        if restrained_dofs is not None:
            if not isinstance(restrained_dofs, list):
                raise TypeError('restrained_dofs must be a list of strings or None')
            if not all(isinstance(d, str) for d in restrained_dofs):
                raise TypeError('each restrained DOF must be a string')

        # Assign normalized values. If inputs are numeric, store as float; if
        # symbolic, store as-is (SymPy expression) so downstream code can use
        # symbolic operations.
        self.x = float(x) if is_numeric_x else x
        self.y = float(y) if is_numeric_y else y
        self.fixity = fixity
        self.restrained_dofs = restrained_dofs if restrained_dofs is not None else []

        # Sanity check: pinned nodes should restrain both x and y DOFs
        if self.fixity == 'pinned':
            required = {'x', 'y'}
            if not required.issubset(set(self.restrained_dofs)):
                raise ValueError("Pinned node should restrain both 'x' and 'y' DOFs")

        # Sanity check: fixed nodes should restrain x, y and rz (in-plane rotation)
        if self.fixity == 'fixed':
            required_fixed = {'x', 'y', 'rz'}
            if not required_fixed.issubset(set(self.restrained_dofs)):
                raise ValueError("Fixed node should restrain 'x', 'y', and 'rz' DOFs")