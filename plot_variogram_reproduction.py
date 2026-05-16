import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import geostatspy.geostats as geostats

from variogram_tuner import spherical, exponential, gaussian_vgm, nugget_only


# Map model names from the variogram diagnostics to their variogram functions
_MODEL_FUNCS = {
    "spherical": spherical,
    "exponential": exponential,
    "gaussian": gaussian_vgm,
    "nugget_only": nugget_only,
    "spherical (fallback)": spherical,
}


def _theoretical_variogram_curve(h, vdiag):
    """Evaluate the fitted theoretical variogram model at lag distances h."""

    # Normalize the model name and find the corresponding variogram function
    model = str(vdiag["model"]).strip().lower()
    fn = _MODEL_FUNCS.get(model)

    if fn is None:
        raise ValueError(f"Unsupported variogram model: {vdiag['model']}")

    # Extract fitted variogram parameters
    nugget = float(vdiag.get("nugget", 0.0))
    partial_sill = float(vdiag.get("partial_sill", 1.0))
    a = float(vdiag.get("fitted_range_param", vdiag.get("practical_range", 1.0)))

    # Nugget-only models use the total sill as a constant semivariance
    if model == "nugget_only":
        return fn(h, nugget + partial_sill)

    # Standard models use nugget, partial sill, and range
    return fn(h, nugget, partial_sill, a)


def _grid_to_point_df(grid2d, x_coords, y_coords, value_col="value"):
    """Convert a 2D grid into a point DataFrame for variogram calculation."""

    # Create coordinate pairs for each grid cell
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Flatten coordinates and grid values into point format
    return pd.DataFrame({
        "X": xx.ravel(),
        "Y": yy.ravel(),
        value_col: np.asarray(grid2d).ravel(),
    })


def _compute_exp_variogram(df, value_col, xlag, xltol, nlag, pair_threshold):
    """Compute and filter an experimental variogram."""

    # Compute omnidirectional experimental variogram using GeostatsPy
    lags, gamma, npairs = geostats.gamv(
        df,
        "X",
        "Y",
        value_col,
        tmin=-999,
        tmax=999,
        xlag=xlag,
        xltol=xltol,
        nlag=nlag,
        azm=0,
        atol=90,
        bandwh=9999.9,
        isill=0,
    )

    # Keep only valid lag bins with enough point pairs
    mask = (
        np.isfinite(lags)
        & np.isfinite(gamma)
        & (lags > 0)
        & (npairs >= pair_threshold)
    )

    return lags[mask], gamma[mask], npairs[mask]


def build_workflow_results(
    results,
    sampled_df,
    *,
    no_gs_residual_col="Estimated_Residual",
    gs_residual_col="Estimated_Residual_GS",
):
    """Organize No-GS and GS outputs into a common comparison structure."""

    # Ensure all required simulation and variogram-diagnostic outputs are present
    required = ["sgs_A", "sgs_B", "vdiag_no_GS", "vdiag_GS"]
    missing = [k for k in required if k not in results]

    if missing:
        raise KeyError(f"Missing required keys in results: {missing}")

    # Ensure the requested residual columns exist in the sampled dataset
    for col in [no_gs_residual_col, gs_residual_col]:
        if col not in sampled_df.columns:
            raise KeyError(f"Column '{col}' not found in sampled_df")

    # Store both workflows using consistent keys for downstream plotting
    workflow_results = {
        "No GS": {
            "residual_col": no_gs_residual_col,
            "vdiag": results["vdiag_no_GS"],
            "sgs_simulations": np.asarray(results["sgs_A"]),
        },
        "GS": {
            "residual_col": gs_residual_col,
            "vdiag": results["vdiag_GS"],
            "sgs_simulations": np.asarray(results["sgs_B"]),
        },
    }

    return workflow_results


def plot_variogram_workflow_comparison(
    results,
    sampled_df,
    *,
    no_gs_residual_col="Estimated_Residual",
    gs_residual_col="Estimated_Residual_GS",
    workflow_order=("No GS", "GS"),
    workflow_labels=None,
    x_col="X",
    y_col="Y",
    xlag=25.0,
    xltol=None,
    nlag=20,
    pair_threshold=20,
    figsize=None,
    alpha_real=0.6,
    color_real="#808080",
    color_avg="red",
    color_theory="blue",
    color_exp="black",
    scatter_s=22,
    sharey=False,
    suptitle="Variogram Reproduction: No GS vs GS",
):
    """Plot variogram reproduction for No-GS and GS workflows side by side."""

    # Use a near-full lag tolerance if none is provided
    if xltol is None:
        xltol = 0.9 * xlag

    # Use simple workflow labels unless custom labels are supplied
    if workflow_labels is None:
        workflow_labels = {
            "No GS": "No GS",
            "GS": "GS",
        }

    # Convert raw results into a common workflow-comparison dictionary
    workflow_results = build_workflow_results(
        results,
        sampled_df,
        no_gs_residual_col=no_gs_residual_col,
        gs_residual_col=gs_residual_col,
    )

    # Create one panel per workflow
    n_workflows = len(workflow_order)

    if figsize is None:
        figsize = (4.4 * n_workflows, 4.2)

    fig, axes = plt.subplots(1, n_workflows, figsize=figsize, sharey=sharey)

    if n_workflows == 1:
        axes = [axes]

    # Use sampled-coordinate extent to assign coordinates to simulation grids
    xmin, xmax = sampled_df[x_col].min(), sampled_df[x_col].max()
    ymin, ymax = sampled_df[y_col].min(), sampled_df[y_col].max()

    for i, (ax, workflow_name) in enumerate(zip(axes, workflow_order)):
        # Retrieve workflow-specific residuals, variogram model, and simulations
        res = workflow_results[workflow_name]
        residual_col = res["residual_col"]
        vdiag = res["vdiag"]
        sims = np.asarray(res["sgs_simulations"])

        # Simulations should be stored as realizations over a 2D grid
        if sims.ndim != 3:
            raise ValueError(
                f"{workflow_name}: expected sgs_simulations shape "
                f"(n_real, ny, nx), got {sims.shape}"
            )

        # Build coordinate vectors for the simulation grid
        n_real, ny, nx = sims.shape
        x_coords = np.linspace(xmin, xmax, nx)
        y_coords = np.linspace(ymin, ymax, ny)

        # ---------------------------------------------------------------------
        # 1. Experimental variogram from sampled residuals
        # ---------------------------------------------------------------------

        # Rename columns to the format expected by the variogram function
        exp_df = sampled_df[[x_col, y_col, residual_col]].copy()
        exp_df.columns = ["X", "Y", "value"]

        # Compute experimental variogram for the sampled residuals
        lags_exp, gamma_exp, _ = _compute_exp_variogram(
            exp_df,
            value_col="value",
            xlag=xlag,
            xltol=xltol,
            nlag=nlag,
            pair_threshold=pair_threshold,
        )

        # Plot the sampled-residual experimental variogram
        ax.scatter(
            lags_exp,
            gamma_exp,
            s=scatter_s,
            color=color_exp,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            zorder=6,
        )

        # ---------------------------------------------------------------------
        # 2. Experimental variograms from SGS realizations
        # ---------------------------------------------------------------------

        gamma_stack = []

        for r in range(n_real):
            # Convert one simulated realization from grid format to point format
            sim_df = _grid_to_point_df(
                sims[r],
                x_coords,
                y_coords,
                value_col="value",
            )

            # Compute the realization variogram
            lags_r, gamma_r, _ = _compute_exp_variogram(
                sim_df,
                value_col="value",
                xlag=xlag,
                xltol=xltol,
                nlag=nlag,
                pair_threshold=pair_threshold,
            )

            # Skip realizations that do not produce valid variogram bins
            if len(lags_r) == 0:
                continue

            # Plot each realization variogram as a thin background line
            ax.plot(
                lags_r,
                gamma_r,
                color=color_real,
                linewidth=1.2,
                alpha=alpha_real,
                zorder=2,
            )

            # Store realization variogram values on a common lag grid
            gamma_full = np.full(nlag, np.nan)
            bin_idx = np.round(lags_r / xlag).astype(int) - 1
            valid = (bin_idx >= 0) & (bin_idx < nlag)
            gamma_full[bin_idx[valid]] = gamma_r[valid]
            gamma_stack.append(gamma_full)

        # ---------------------------------------------------------------------
        # 3. Average variogram over SGS realizations
        # ---------------------------------------------------------------------

        lag_centers = np.arange(1, nlag + 1) * xlag
        gamma_avg = None

        if gamma_stack:
            # Average across realizations while ignoring missing lag bins
            gamma_stack = np.vstack(gamma_stack)
            gamma_avg = np.nanmean(gamma_stack, axis=0)
            valid_avg = np.isfinite(gamma_avg)

            # Plot the average simulated variogram
            ax.plot(
                lag_centers[valid_avg],
                gamma_avg[valid_avg],
                color=color_avg,
                linewidth=2.0,
                zorder=4,
            )

        # ---------------------------------------------------------------------
        # 4. Fitted theoretical variogram
        # ---------------------------------------------------------------------

        # Evaluate the fitted model over a smooth lag-distance vector
        h = np.linspace(0.0, nlag * xlag, 400)
        gamma_th = _theoretical_variogram_curve(h, vdiag)

        # Plot the fitted theoretical variogram
        ax.plot(
            h,
            gamma_th,
            color=color_theory,
            linewidth=2.2,
            zorder=5,
        )

        # ---------------------------------------------------------------------
        # Panel formatting
        # ---------------------------------------------------------------------

        # Set y-axis limit to include sill, sampled variogram, and average simulation curve
        total_sill = float(
            vdiag.get("total_sill", vdiag["nugget"] + vdiag["partial_sill"])
        )

        ymax_panel = max(
            1.15 * total_sill,
            np.nanmax(gamma_exp) if len(gamma_exp) else 0.0,
            np.nanmax(gamma_avg) if gamma_avg is not None else 0.0,
        )

        ax.set_xlim(0, nlag * xlag)
        ax.set_ylim(0, ymax_panel * 1.1 if ymax_panel > 0 else 1.0)
        ax.set_xlabel(r"Lag Separation Distance, $h$")

        if i == 0:
            ax.set_ylabel(r"Semivariance $\gamma(h)$")

        ax.set_title(
            workflow_labels.get(workflow_name, workflow_name),
            fontweight="bold",
        )
        ax.grid(True, linestyle=":", alpha=0.4)

    # Shared legend for all panels
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor=color_exp,
            markeredgecolor="black",
            label="Experimental variogram",
        ),
        Line2D(
            [0],
            [0],
            color=color_real,
            linewidth=1.0,
            alpha=0.7,
            label="Individual realization",
        ),
        Line2D(
            [0],
            [0],
            color=color_avg,
            linewidth=2.0,
            label="Average over realizations",
        ),
        Line2D(
            [0],
            [0],
            color=color_theory,
            linewidth=2.2,
            label="Theoretical model",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        fontsize=8.5,
        frameon=True,
    )

    # Add optional figure-level title
    if suptitle:
        fig.suptitle(suptitle, fontweight="bold", y=1.02)

    plt.tight_layout()

    return fig, axes