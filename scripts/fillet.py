from typing import List, Union, Tuple, Optional

from objects.node import Node
from objects.member import Member2D
import math


def _resolve_member(
    members: List[Member2D], key: Union[int, str, Member2D]
) -> Member2D:
    """Resolve a member from an index, name, or Member2D instance."""
    if isinstance(key, Member2D):
        return key
    if isinstance(key, int):
        return members[key]
    # assume string -> member.name
    for m in members:
        if getattr(m, "name", None) == key:
            return m
    raise ValueError(f"Member {key!r} not found")


def apply_fillet(
    nodes: List[Node],
    members: List[Member2D],
    member_a: Union[int, str, Member2D],
    member_b: Union[int, str, Member2D],
    subdivide: str = "a",
    n_segments: int = 4,
    extra_area: float = None,
) -> Tuple[List[Node], List[Member2D]]:
    """Apply a fillet between two members that share a node.

    The function checks the two members share a common node, then subdivides
    the chosen member into `n_segments` pieces by inserting intermediate
    nodes along its length starting at the common node. The new segments have
    linearly increasing cross-sectional area from small -> original A using
    a simple Riemann-style assignment: A_i = A_orig * ((i+1)/n_segments).

    Parameters
    - nodes, members: original lists (these are not modified; returned lists
      are new copies containing original objects plus new ones)
    - member_a, member_b: either indices into `members`, member names, or
      `Member2D` objects for the two members forming the fillet
    - subdivide: which member to subdivide: 'a' or 'b' (default 'a')
    - n_segments: number of segments to split the chosen member into (>=1)

    Returns
    - new_nodes, new_members: lists containing Node and Member2D objects with
      the fillet applied.
    """
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")

    # resolve members
    m_a = _resolve_member(members, member_a)
    m_b = _resolve_member(members, member_b)

    # find common node
    a_nodes = {m_a.node_start.name: m_a.node_start, m_a.node_end.name: m_a.node_end}
    b_nodes = {m_b.node_start.name: m_b.node_start, m_b.node_end.name: m_b.node_end}
    common_names = set(a_nodes.keys()).intersection(set(b_nodes.keys()))
    if not common_names:
        raise ValueError("Members do not share a common node")
    # pick the first common node
    common_name = next(iter(common_names))
    common_node = a_nodes[common_name]

    # determine which member to subdivide
    if subdivide not in ("a", "b"):
        raise ValueError("subdivide must be 'a' or 'b'")

    target = m_a if subdivide == "a" else m_b
    # other node on the target member (the node that is not the common node)
    if target.node_start.name == common_name:
        other_node = target.node_end
    else:
        other_node = target.node_start

    # Prepare output copies (shallow copies of original lists)
    new_nodes = list(nodes)
    new_members = []
    # copy existing members except the target (we'll replace it with subdivided pieces)
    for m in members:
        if m is target:
            continue
        new_members.append(m)

    # compute vector from common node to the other node
    dx = float(other_node.x) - float(common_node.x)
    dy = float(other_node.y) - float(common_node.y)
    full_len = math.hypot(dx, dy)
    if full_len <= 1e-12:
        # degenerate member, just return copies
        return new_nodes, new_members

    ux = dx / full_len
    uy = dy / full_len

    # create interior nodes equally spaced along the entire member
    interior_nodes = []
    for i in range(1, n_segments):
        di = (i / n_segments) * full_len
        xi = float(common_node.x) + ux * di
        yi = float(common_node.y) + uy * di
        n_node = Node(xi, yi, "free", None)
        interior_nodes.append(n_node)
        new_nodes.append(n_node)

    # assemble node list along the target from common -> other
    segment_nodes = [common_node] + interior_nodes + [other_node]

    A_orig = getattr(target, "A", 1.0)
    if extra_area is None:
        extra_area = A_orig
    E = getattr(target, "E", 1.0)
    I = getattr(target, "I", 0.0)
    elem_type = getattr(target, "element_type", "truss")

    # create subdivided members (we'll assign areas after sorting by distance)
    start_index = len(new_members)
    for idx in range(n_segments):
        n_start = segment_nodes[idx]
        n_end = segment_nodes[idx + 1]
        new_m = Member2D(elem_type, n_start, n_end, E=E, A=A_orig, I=I)
        new_members.append(new_m)

    # Assign areas so they increase with distance from the common node
    created = new_members[start_index : start_index + n_segments]

    # compute distance of each segment midpoint from the common node
    def _seg_distance(seg: Member2D) -> float:
        sx = float(seg.node_start.x)
        sy = float(seg.node_start.y)
        ex = float(seg.node_end.x)
        ey = float(seg.node_end.y)
        mx = 0.5 * (sx + ex)
        my = 0.5 * (sy + ey)
        return math.hypot(mx - float(common_node.x), my - float(common_node.y))

    created.sort(key=_seg_distance)
    # assign areas so largest is at the connection (closest to common node)
    for i, seg in enumerate(created):
        multiplier = (n_segments - i) / n_segments
        A_i = float(A_orig + extra_area * multiplier)
        setattr(seg, "A", A_i)

    return new_nodes, new_members


def apply_fillet_radius(
    nodes: List[Node],
    members: List[Member2D],
    member_a: Union[int, str, Member2D],
    member_b: Union[int, str, Member2D],
    subdivide: str = "a",
    radius: float = 0.2,
    n_segments: int = 4,
    extra_area: float = None,
) -> Tuple[List[Node], List[Member2D]]:
    """Apply a fillet of given `radius` along `subdivide` member from the common node.

    This subdivides only the initial length `radius` from the common node into
    `n_segments` pieces with increasing area and leaves the remainder of the
    original member as a single member with the original area.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")

    # resolve members
    m_a = _resolve_member(members, member_a)
    m_b = _resolve_member(members, member_b)

    # find common node
    a_nodes = {m_a.node_start.name: m_a.node_start, m_a.node_end.name: m_a.node_end}
    b_nodes = {m_b.node_start.name: m_b.node_start, m_b.node_end.name: m_b.node_end}
    common_names = set(a_nodes.keys()).intersection(set(b_nodes.keys()))
    if not common_names:
        raise ValueError("Members do not share a common node")
    common_name = next(iter(common_names))
    common_node = a_nodes[common_name]

    # determine target member
    if subdivide not in ("a", "b"):
        raise ValueError("subdivide must be 'a' or 'b'")
    target = m_a if subdivide == "a" else m_b
    if target.node_start.name == common_name:
        other_node = target.node_end
    else:
        other_node = target.node_start

    # compute full length of target (using node coordinates)
    dx = float(other_node.x) - float(common_node.x)
    dy = float(other_node.y) - float(common_node.y)
    full_len = math.hypot(dx, dy)

    # if radius >= full length, fall back to full-length subdivision
    if radius >= full_len:
        return apply_fillet(
            nodes,
            members,
            member_a,
            member_b,
            subdivide=subdivide,
            n_segments=n_segments,
        )

    # Prepare output copies (shallow copies of original lists)
    new_nodes = list(nodes)
    new_members = []
    for m in members:
        if m is target:
            continue
        new_members.append(m)

    # unit direction from common to other
    ux = dx / full_len
    uy = dy / full_len

    # create interior nodes along distance <= radius
    interior_nodes = []
    for i in range(1, n_segments):
        di = (i / n_segments) * radius
        xi = float(common_node.x) + ux * di
        yi = float(common_node.y) + uy * di
        n_node = Node(xi, yi, "free", None)
        interior_nodes.append(n_node)
        new_nodes.append(n_node)

    # last fillet point at distance = radius
    last_x = float(common_node.x) + ux * radius
    last_y = float(common_node.y) + uy * radius
    last_node = Node(last_x, last_y, "free", None)
    interior_nodes.append(last_node)
    new_nodes.append(last_node)

    # build segments: fillet segments from common -> interior nodes
    fillet_nodes = [common_node] + interior_nodes

    A_orig = getattr(target, "A", 1.0)
    if extra_area is None:
        extra_area = A_orig
    E = getattr(target, "E", 1.0)
    I = getattr(target, "I", 0.0)
    elem_type = getattr(target, "element_type", "truss")

    # create fillet small segments (n_segments pieces)
    # create fillet small segments (n_segments pieces); assign placeholder A then reorder
    start_index = len(new_members)
    for idx in range(n_segments):
        n_start = fillet_nodes[idx]
        n_end = fillet_nodes[idx + 1]
        new_m = Member2D(elem_type, n_start, n_end, E=E, A=A_orig, I=I)
        new_members.append(new_m)

    # assign areas so they increase with distance from the common node
    created = new_members[start_index : start_index + n_segments]

    def _seg_distance(seg: Member2D) -> float:
        sx = float(seg.node_start.x)
        sy = float(seg.node_start.y)
        ex = float(seg.node_end.x)
        ey = float(seg.node_end.y)
        mx = 0.5 * (sx + ex)
        my = 0.5 * (sy + ey)
        return math.hypot(mx - float(common_node.x), my - float(common_node.y))

    created.sort(key=_seg_distance)
    # assign areas so largest is at the connection (closest to common node)
    for i, seg in enumerate(created):
        multiplier = (n_segments - i) / n_segments
        A_i = float(A_orig + extra_area * multiplier)
        setattr(seg, "A", A_i)

    # remaining member from last fillet node to original other_node (if distance remains)
    # If radius < full_len, create remainder segment with original area.
    rem_dx = float(other_node.x) - last_x
    rem_dy = float(other_node.y) - last_y
    rem_len = math.hypot(rem_dx, rem_dy)
    if rem_len > 1e-12:
        new_m_rem = Member2D(elem_type, last_node, other_node, E=E, A=A_orig, I=I)
        new_members.append(new_m_rem)

    return new_nodes, new_members


def fillet_and_report(
    nodes: List[Node],
    members: List[Member2D],
    member_to_subdivide: Union[int, str, Member2D],
    other_member: Union[int, str, Member2D],
    *,
    radius: float = 0.2,
    n_segments: int = 4,
    subdivide: str = "a",
) -> Tuple[List[Node], List[Member2D], dict]:
    """Apply fillet to a single member and return a mapping of replacements.

    Returns (new_nodes, new_members, replacements) where `replacements` maps
    the original member name to the list of new Member2D objects that
    replaced it. This makes downstream plotting / bookkeeping easier.
    """
    # Use apply_fillet_radius which already handles the geometry; make sure
    # we pass the parameters in the correct order depending on `subdivide`.
    if subdivide == "a":
        new_nodes, new_members = apply_fillet_radius(
            nodes,
            members,
            member_to_subdivide,
            other_member,
            subdivide="a",
            radius=radius,
            n_segments=n_segments,
        )
        orig = _resolve_member(members, member_to_subdivide)
    else:
        new_nodes, new_members = apply_fillet_radius(
            nodes,
            members,
            other_member,
            member_to_subdivide,
            subdivide="b",
            radius=radius,
            n_segments=n_segments,
        )
        orig = _resolve_member(members, member_to_subdivide)

    # Find which new members were created from replacing `orig` by comparing
    # node connectivity: new members that include the common node and lie
    # along the original line between orig.node_start and orig.node_end.
    replacements = {}
    repl_list = []
    for m in new_members:
        if orig.node_start.name in (
            m.node_start.name,
            m.node_end.name,
        ) or orig.node_end.name in (m.node_start.name, m.node_end.name):
            # crude check: same element_type and at least one shared original node name
            if getattr(m, "element_type", None) == getattr(orig, "element_type", None):
                repl_list.append(m)
    replacements[orig.name] = repl_list

    # User-facing report: show the node where fillet was applied and areas
    try:
        # Determine common node name (the node shared between orig and other_member)
        common_node_name = None
        # orig.node_start/ node_end are Node objects; check which one appears in repl_list
        for m in repl_list:
            if orig.node_start.name in (m.node_start.name, m.node_end.name):
                common_node_name = orig.node_start.name
                break
            if orig.node_end.name in (m.node_start.name, m.node_end.name):
                common_node_name = orig.node_end.name
                break

        print(f"Fillet report for original member: {orig.name}")
        if common_node_name:
            print(f"- Fillet applied at node: {common_node_name}")
        else:
            print("- Fillet applied at node: (unknown)")

        for seg in repl_list:
            area = getattr(seg, "A", None)
            start = getattr(seg.node_start, "name", None)
            end = getattr(seg.node_end, "name", None)
            print(f"  - Segment: {seg.name} | Nodes: {start} -> {end} | Area: {area}")
    except Exception:
        # printing should not break functionality
        pass

    return new_nodes, new_members, replacements


def apply_fillets_at_nodes(
    nodes: List[Node],
    members: List[Member2D],
    node_names: Optional[List[str]] = None,
    *,
    radius: float = 0.25,
    n_segments: int = 4,
) -> Tuple[List[Node], List[Member2D], dict]:
    """Apply fillets at a list of node names (or all junction nodes if None).

    For each named node, this function finds all members attached to that
    node and applies `fillet_between_members` to each attached member using
    another attached member as the "other" reference. The nodes/members
    lists are updated iteratively so successive fillets operate on the
    latest geometry. Returns the final (nodes, members, replacements_map)
    where `replacements_map` maps original member names to lists of created
    segments across all fillet operations.
    """
    # decide which nodes to operate on
    if node_names is None:
        # pick nodes that are shared by more than one member
        counts: dict = {}
        for m in members:
            counts[m.node_start.name] = counts.get(m.node_start.name, 0) + 1
            counts[m.node_end.name] = counts.get(m.node_end.name, 0) + 1
        node_names = [n for n, c in counts.items() if c > 1]

    new_nodes = list(nodes)
    new_members = list(members)
    all_replacements: dict = {}

    filleted_members = set()
    for nn in node_names:
        # find members attached to this node (by name)
        attached = [m for m in new_members if m.node_start.name == nn or m.node_end.name == nn]
        if len(attached) < 2:
            continue
        # choose a single member to fillet at this node: pick the longest attached member
        def _length(m: Member2D) -> float:
            x0 = float(m.node_start.x)
            y0 = float(m.node_start.y)
            x1 = float(m.node_end.x)
            y1 = float(m.node_end.y)
            return math.hypot(x1 - x0, y1 - y0)

        # sort attached by length descending and pick the longest as candidate
        attached_sorted = sorted(attached, key=_length, reverse=True)
        candidate = attached_sorted[0]
        if getattr(candidate, "name", None) in filleted_members:
            # already filleted at another node, skip
            continue

        # pick another attached member as reference (choose the next-longest)
        other = attached_sorted[1] if len(attached_sorted) > 1 else attached_sorted[0]
        try:
            new_nodes, new_members, repl = fillet_between_members(
                new_nodes, new_members, candidate, other, radius=radius, n_segments=n_segments
            )
        except Exception:
            # skip failures but keep going
            continue
        # record that candidate has been filleted so we don't fillet it again at its other end
        filleted_members.add(getattr(candidate, "name", None))
        # merge repl into all_replacements
        for k, v in repl.items():
            if k not in all_replacements:
                all_replacements[k] = list(v)
            else:
                all_replacements[k].extend(v)

    return new_nodes, new_members, all_replacements


def fillet_between_members(
    nodes: List[Node],
    members: List[Member2D],
    member_a: Union[int, str, Member2D],
    member_b: Union[int, str, Member2D],
    *,
    radius: float = 0.2,
    n_segments: int = 4,
) -> Tuple[List[Node], List[Member2D], dict]:
    """Apply a fillet by subdividing only `member_a` near the shared node.

    This function subdivides `member_a` for a distance `radius` from the
    common node into `n_segments` pieces with increasing area, leaving
    `member_b` unchanged. Returns new nodes, new members, and a replacements
    dict mapping `member_a.name` -> list of new Member2D segments.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")

    # Delegate to the single-member radius fillet implementation.
    new_nodes, new_members = apply_fillet_radius(
        nodes,
        members,
        member_a,
        member_b,
        subdivide="a",
        radius=radius,
        n_segments=n_segments,
    )

    # Determine the original member object
    orig = _resolve_member(members, member_a)

    # Geometric selection: pick only new members that lie on the original
    # member line (avoids including other members sharing the common node).
    x0 = float(orig.node_start.x)
    y0 = float(orig.node_start.y)
    x1 = float(orig.node_end.x)
    y1 = float(orig.node_end.y)
    dx = x1 - x0
    dy = y1 - y0
    L = math.hypot(dx, dy) or 1.0

    def _on_line(x: float, y: float) -> bool:
        cross = (x - x0) * dy - (y - y0) * dx
        return abs(cross) <= 1e-6 * L

    repl_list: List[Member2D] = []
    for m in new_members:
        if getattr(m, "element_type", None) != getattr(orig, "element_type", None):
            continue
        sx = float(m.node_start.x)
        sy = float(m.node_start.y)
        ex = float(m.node_end.x)
        ey = float(m.node_end.y)
        if _on_line(sx, sy) and _on_line(ex, ey):
            repl_list.append(m)

    # Determine the common node explicitly (shared between member_a and member_b)
    m_b_obj = _resolve_member(members, member_b)
    a_nodes = {orig.node_start.name, orig.node_end.name}
    b_nodes = {m_b_obj.node_start.name, m_b_obj.node_end.name}
    common_set = a_nodes.intersection(b_nodes)
    common_name = next(iter(common_set)) if common_set else None

    # sort by distance from common node so segments are ordered from corner outward
    def _dist_from_common(seg: Member2D) -> float:
        if common_name is None:
            return 0.0
        if seg.node_start.name == common_name:
            ox = float(seg.node_end.x)
            oy = float(seg.node_end.y)
        elif seg.node_end.name == common_name:
            ox = float(seg.node_start.x)
            oy = float(seg.node_start.y)
        else:
            ox = float(seg.node_start.x)
            oy = float(seg.node_start.y)
        # compute distance along original member axis from common node
        cx = float(
            orig.node_start.x
            if orig.node_start.name == common_name
            else orig.node_end.x
        )
        cy = float(
            orig.node_start.y
            if orig.node_start.name == common_name
            else orig.node_end.y
        )
        return math.hypot(ox - cx, oy - cy)

    repl_list.sort(key=_dist_from_common)

    replacements = {orig.name: repl_list}

    # Print report
    try:
        print(f"Fillet (member_a) report for original member: {orig.name}")
        if common_name:
            print(f"- Fillet applied at node: {common_name}")
        else:
            print("- Fillet applied at node: (unknown)")
        for seg in repl_list:
            area = getattr(seg, "A", None)
            s = getattr(seg.node_start, "name", None)
            e = getattr(seg.node_end, "name", None)
            print(f"  - Segment: {seg.name} | Nodes: {s} -> {e} | Area: {area}")
    except Exception:
        pass

    return new_nodes, new_members, replacements
