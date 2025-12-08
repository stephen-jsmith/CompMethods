import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

# ------------------------------------------------------------
# Core solvers
# ------------------------------------------------------------


def stacked_truss_buckling_all(
    s_upper_kNm: float,  # spring at upper intermediate joint [kN/m]
    s_lower_kNm: float,  # spring at lower intermediate joint [kN/m]
    L_top_mm: float,
    L_mid_mm: float,
    L_bot_mm: float,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Compute all positive-real critical loads and corresponding mode shapes for the
    stacked-truss (2 DOF) buckling problem.

    Sign convention: compression reduces stiffness => K_total(P) = Ke - P * Khat_g.

    Inputs:
      s_upper_kNm, s_lower_kNm : springs in kN/m (converted to kN/mm)
      L_top_mm, L_mid_mm, L_bot_mm : member lengths [mm]

    Returns:
      P_cr_list : np.ndarray [kN] of all positive real roots (sorted ascending)
      modes     : list of np.ndarray, each normalized mode [u_upper, u_lower] for corresponding P_cr
      Ke        : 2x2 material stiffness (kN/mm)
      Khat_g    : 2x2 normalized geometric stiffness (1/mm)
    """
    # ---- Unit conversion: kN/m -> kN/mm ----
    s_upper = s_upper_kNm / 1000.0  # kN/mm
    s_lower = s_lower_kNm / 1000.0  # kN/mm

    # 1) Material stiffness (springs at intermediate joints)
    Ke = np.array([[s_upper, 0.0], [0.0, s_lower]], dtype=float)

    # 2) Normalized geometric stiffness (P–Δ from the three vertical bars)
    Khat_g = np.array(
        [
            [1.0 / L_top_mm + 1.0 / L_mid_mm, -1.0 / L_mid_mm],
            [-1.0 / L_mid_mm, 1.0 / L_mid_mm + 1.0 / L_bot_mm],
        ],
        dtype=float,
    )

    # 3) Solve det(Ke - P*Khat_g) = 0 => quadratic in P
    det_Ke = np.linalg.det(Ke)
    det_Khat = np.linalg.det(Khat_g)
    # adj(A) for 2x2
    adj_Ke = np.array([[Ke[1, 1], -Ke[0, 1]], [-Ke[1, 0], Ke[0, 0]]])
    a = det_Khat
    b = -np.trace(adj_Ke @ Khat_g)  # minus sign: compression reduces stiffness
    c = det_Ke

    # Roots (could be complex)
    roots = np.roots(np.array([a, b, c], dtype=float))

    # Filter positive real roots
    P_cr_list = [
        float(np.real(r)) for r in roots if np.isreal(r) and float(np.real(r)) > 0.0
    ]
    P_cr_list = np.array(sorted(P_cr_list), dtype=float)

    modes: List[np.ndarray] = []
    for P_cr in P_cr_list:
        # Nullspace of (Ke - P_cr*Khat_g) via SVD
        K_total = Ke - P_cr * Khat_g
        U, S, Vt = np.linalg.svd(K_total)
        v = Vt[-1, :]  # right-singular vector for smallest singular value
        v = v / np.max(np.abs(v))  # normalize to max component = 1
        modes.append(v)

    return P_cr_list, modes, Ke, Khat_g


def solve_buckling_eigensystem(
    s_upper_kNm: float,
    s_lower_kNm: float,
    L_top_mm: float,
    L_mid_mm: float,
    L_bot_mm: float,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Convenience wrapper that computes and PRINTS the eigenvalues (critical loads)
    and eigenvectors (mode shapes) for the stacked-truss system.

    Returns:
      P_cr_list : array of eigenvalues (kN), ascending
      modes     : list of normalized eigenvectors [u_upper, u_lower] in same order
    """
    P_cr_list, modes, Ke, Khat_g = stacked_truss_buckling_all(
        s_upper_kNm, s_lower_kNm, L_top_mm, L_mid_mm, L_bot_mm
    )

    # Pretty print
    print("\n=== Buckling Eigen-System Summary ===")
    print("Material stiffness Ke (kN/mm):")
    print(Ke)
    print("\nNormalized geometric stiffness Khat_g (1/mm):")
    print(Khat_g)
    print("\nEigenvalues (critical loads) and eigenvectors (normalized):")
    for i, (P_cr, v) in enumerate(zip(P_cr_list, modes), start=1):
        print(f"  Mode {i}:")
        print(f"    Eigenvalue (P_cr) = {P_cr:.6f} kN")
        print(f"    Eigenvector [u_upper, u_lower] = [{v[0]:+.6f}, {v[1]:+.6f}]")
    print("\n------------------------------------------------------------\n")

    return P_cr_list, modes


# ------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------


def compute_node_offsets(
    mode: np.ndarray,
    L_top_mm: float,
    L_mid_mm: float,
    L_bot_mm: float,
    scale_fraction: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute y-coordinates and scaled x-offsets (mm) used in the plot,
    along with the scale factor applied to normalized DOFs.

    Returns:
      y (np.ndarray): [0, y_lower, y_upper, y_top]
      x (np.ndarray): scaled offsets [bottom, lower, upper, top] in mm
      scale (float): scale factor applied to normalized DOFs (mm per unit DOF)
    """
    y = np.array([0.0, L_bot_mm, L_bot_mm + L_mid_mm, L_bot_mm + L_mid_mm + L_top_mm])

    # Normalized lateral offsets at nodes: supports=0; joints=mode components
    x_offsets_norm = np.array([0.0, mode[1], mode[0], 0.0])

    # Scale offsets for visibility
    H = y[-1] - y[0]
    max_off = (
        np.max(np.abs(x_offsets_norm)) if np.max(np.abs(x_offsets_norm)) > 0 else 1.0
    )
    scale = scale_fraction * H / max_off
    x_mm = x_offsets_norm * scale

    return y, x_mm, scale


def plot_buckled_mode(
    mode: np.ndarray,
    L_top_mm: float,
    L_mid_mm: float,
    L_bot_mm: float,
    P_cr_kN: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    scale_fraction: float = 0.02,
    ax: Optional[plt.Axes] = None,
    title_suffix: Optional[str] = None,
    annotate_values: bool = True,
):
    """
    Plot a single buckled mode profile and (optionally) annotate displacement values.
    """
    if mode is None:
        raise ValueError("Mode is None; cannot plot. Compute P_cr and mode first.")

    y, x, scale = compute_node_offsets(
        mode, L_top_mm, L_mid_mm, L_bot_mm, scale_fraction
    )

    made_fig = False
    if ax is None:
        plt.figure(figsize=(4, 8))
        ax = plt.gca()
        made_fig = True

    # Undeformed centerline
    ax.plot([0, 0], [y[0], y[-1]], "k--", linewidth=1, label="Undeformed")
    # Buckled shape
    ax.plot(x, y, "b-", linewidth=2, label="Buckled mode (scaled)")
    ax.scatter(x, y, color="blue")

    labels = ["Bottom support", "Lower joint", "Upper joint", "Top support"]
    for xi, yi, lab in zip(x, y, labels):
        ax.text(xi + 0.01 * (y[-1] - y[0]), yi, lab, fontsize=9, va="center")

    # Optional annotation of displacement values
    if annotate_values:
        ax.text(
            x[1],
            y[1] - 0.03 * (y[-1] - y[0]),
            f"u_lower = {mode[1]:+.4f}\nplot = {x[1]:+.2f} mm",
            fontsize=9,
            ha="center",
            va="top",
            color="darkblue",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )
        ax.text(
            x[2],
            y[2] + 0.03 * (y[-1] - y[0]),
            f"u_upper = {mode[0]:+.4f}\nplot = {x[2]:+.2f} mm",
            fontsize=9,
            ha="center",
            va="bottom",
            color="darkblue",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

    title = "Stacked Truss Buckling Mode (normalized)"
    if P_cr_kN is not None:
        title += f"\nP_cr = {P_cr_kN:.3f} kN"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title)
    ax.set_xlabel("Lateral offset (scaled, mm)")
    ax.set_ylabel("Vertical coordinate (mm)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()

    if save_path is not None and made_fig:
        plt.savefig(save_path, dpi=150)
    if show and made_fig:
        plt.show()
    elif made_fig:
        plt.close()


def plot_all_modes(
    P_cr_list: np.ndarray,
    modes: List[np.ndarray],
    L_top_mm: float,
    L_mid_mm: float,
    L_bot_mm: float,
    save_prefix: Optional[str] = None,
    show: bool = True,
    scale_fraction: float = 0.02,
    annotate_values: bool = True,
):
    """
    Plot every mode in a single figure with subplots, and optionally save a combined PNG.
    """
    n = len(modes)
    if n == 0:
        raise ValueError("No modes to plot.")

    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(4 * n, 8), squeeze=False)
    axes = axes[0]  # 1D list of axes

    for i, (P_cr, mode) in enumerate(zip(P_cr_list, modes), start=1):
        plot_buckled_mode(
            mode=mode,
            L_top_mm=L_top_mm,
            L_mid_mm=L_mid_mm,
            L_bot_mm=L_bot_mm,
            P_cr_kN=P_cr,
            save_path=None,
            show=False,  # plot into provided ax; don't show here
            scale_fraction=scale_fraction,
            ax=axes[i - 1],
            title_suffix=f"Mode {i}",
            annotate_values=annotate_values,
        )

    plt.tight_layout()
    if save_prefix is not None:
        fig.savefig(f"{save_prefix}_modes.png", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


# ------------------------------------------------------------
# Example / CLI entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    # --- Your exact inputs (springs in kN/m; lengths in mm) ---
    L_top = 1000.0  # mm
    L_mid = 3000.0  # mm
    L_bot = 3000.0  # mm
    S_upper = 50.0  # kN/m
    S_lower = 100.0  # kN/m

    # Solve and print the eigenvalues & eigenvectors
    P_cr_list, modes = solve_buckling_eigensystem(S_upper, S_lower, L_top, L_mid, L_bot)

    # Also print displacement values (normalized & plotted offsets) for each mode
    for i, (P_cr, mode) in enumerate(zip(P_cr_list, modes), start=1):
        y, x_mm, scale = compute_node_offsets(
            mode, L_top, L_mid, L_bot, scale_fraction=0.02
        )
        print(f"Mode {i}: P_cr = {P_cr:.6f} kN")
        print(
            f"  Normalized eigenvector: [u_upper, u_lower] = [{mode[0]:+.6f}, {mode[1]:+.6f}]"
        )
        print(f"  Plotted offsets (mm) with scale = {scale:.3f} mm/DOF:")
        print(f"    Lower joint @ y={y[1]:.1f} mm: x_plot = {x_mm[1]:+.3f} mm")
        print(f"    Upper joint @ y={y[2]:.1f} mm: x_plot = {x_mm[2]:+.3f} mm")
        print()

    # Plot all modes side-by-side and optionally save the combined figure
    plot_all_modes(
        P_cr_list=P_cr_list,
        modes=modes,
        L_top_mm=L_top,
        L_mid_mm=L_mid,
        L_bot_mm=L_bot,
        save_prefix=None,  # e.g., 'buckled' to save 'buckled_modes.png'
        show=True,
        scale_fraction=0.02,
        annotate_values=True,
    )
