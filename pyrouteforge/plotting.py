"""
Beautiful, professional Plotly visualizations for routeforge.

Two main figures are exposed:

* :func:`plot_routes` — the routed solution on a 2D plane, with depots,
  clients, animated-style gradient route lines, and rich hover tooltips.
* :func:`plot_convergence` — best-distance vs generation, useful for
  diagnosing GA convergence.

Both return a ``plotly.graph_objects.Figure`` so users can call ``.show()``
in Jupyter / Colab or ``.write_html(...)`` / ``.write_image(...)`` for
exports. The figures render natively in Google Colab.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Visual design tokens
# ---------------------------------------------------------------------------
# A curated, perceptually-distinct palette that holds up on dark backgrounds.
# These were selected to stay legible when many routes are plotted together.
_ROUTE_PALETTE = [
    "#FF6B9D",  # coral pink
    "#4ECDC4",  # teal
    "#FFE66D",  # warm yellow
    "#A78BFA",  # lavender
    "#FB923C",  # orange
    "#34D399",  # emerald
    "#60A5FA",  # sky blue
    "#F472B6",  # rose
    "#FBBF24",  # amber
    "#22D3EE",  # cyan
    "#C084FC",  # purple
    "#84CC16",  # lime
    "#F87171",  # red
    "#06B6D4",  # ocean
    "#E879F9",  # magenta
    "#FACC15",  # gold
]

_BACKGROUND = "#0F172A"      # slate-900
_PANEL = "#1E293B"           # slate-800
_GRID = "#334155"            # slate-700
_TEXT = "#E2E8F0"            # slate-200
_TEXT_DIM = "#94A3B8"        # slate-400
_DEPOT_COLOR = "#FBBF24"     # amber-400
_CLIENT_COLOR = "#CBD5E1"    # slate-300


def _route_color(idx: int) -> str:
    return _ROUTE_PALETTE[idx % len(_ROUTE_PALETTE)]


def _layout(title: str, width: Optional[int], height: Optional[int]) -> dict:
    """Shared dark-themed layout used by all routeforge figures."""
    return dict(
        title=dict(
            text=title,
            font=dict(family="Inter, system-ui, sans-serif", size=20, color=_TEXT),
            x=0.5, xanchor="center", y=0.96,
        ),
        paper_bgcolor=_BACKGROUND,
        plot_bgcolor=_PANEL,
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=_TEXT),
        margin=dict(l=60, r=40, t=70, b=60),
        width=width, height=height,
        hoverlabel=dict(
            bgcolor=_PANEL, bordercolor=_GRID,
            font=dict(family="Inter, monospace", size=12, color=_TEXT),
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.65)", bordercolor=_GRID, borderwidth=1,
            font=dict(color=_TEXT, size=11),
            itemsizing="constant",
        ),
    )


# ---------------------------------------------------------------------------
# Tour / routes plot
# ---------------------------------------------------------------------------

def plot_routes(
    coordinates: np.ndarray,
    solution: list,
    *,
    n_depots: int = 1,
    route: str = "closed",
    labels: Optional[Sequence[str]] = None,
    title: str = "Optimized Routes",
    show_arrows: bool = True,
    show_node_labels: bool = True,
    width: Optional[int] = 900,
    height: Optional[int] = 650,
) -> go.Figure:
    """Render the routed solution as a polished Plotly figure.

    Parameters
    ----------
    coordinates
        Array of shape ``(n_nodes, 2)`` giving x/y positions for every node.
    solution
        The internal solution structure ``[depots, routes, vehicles]``.
    n_depots
        Number of depots (the first ``n_depots`` rows of ``coordinates``).
    route
        ``'closed'`` adds the depot-return leg; ``'open'`` does not.
    labels
        Optional human-readable name per node (e.g. customer names).
    show_arrows
        Draw a small arrowhead at every leg so direction is obvious.
    show_node_labels
        Annotate each node with its index (or label).
    """
    coords = np.asarray(coordinates, dtype=float)
    n_nodes = coords.shape[0]
    depots = solution[0]
    tours = solution[1]
    vehicles = solution[2]

    if labels is None:
        labels = [str(i) for i in range(n_nodes)]
    else:
        labels = [str(x) for x in labels]

    fig = go.Figure()

    # --- Route paths --------------------------------------------------------
    arrow_annotations: list = []
    for j, sub in enumerate(tours):
        if not sub:
            continue
        depot_idx = depots[j][0]
        v_type = vehicles[j][0]
        path = [depot_idx] + list(sub)
        if route == "closed":
            path.append(depot_idx)
        xs = coords[path, 0]
        ys = coords[path, 1]
        color = _route_color(j)

        # Soft glow underlay (thicker, low-opacity line) for a luminous feel.
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=color, width=8),
            opacity=0.18, hoverinfo="skip", showlegend=False,
        ))

        # Crisp main line + markers
        hover = [
            f"<b>Route #{j + 1}</b><br>"
            f"Vehicle type: {v_type}<br>"
            f"Stop {k}: {labels[node]}<br>"
            f"x = {coords[node, 0]:.2f}<br>"
            f"y = {coords[node, 1]:.2f}"
            for k, node in enumerate(path)
        ]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color=color, width=2.5, shape="linear"),
            marker=dict(size=7, color=color,
                        line=dict(color="white", width=1)),
            name=f"Route #{j + 1} · veh {v_type}",
            hovertext=hover, hoverinfo="text",
            legendgroup=f"r{j}",
        ))

        # Arrowheads at midpoints of every leg
        if show_arrows:
            for k in range(len(path) - 1):
                ax, ay = coords[path[k]]
                bx, by = coords[path[k + 1]]
                arrow_annotations.append(dict(
                    ax=ax, ay=ay, x=bx, y=by,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1.0,
                    arrowwidth=1.2, arrowcolor=color, opacity=0.85,
                    standoff=8, startstandoff=8,
                ))

    # --- Clients ------------------------------------------------------------
    client_idx = list(range(n_depots, n_nodes))
    if client_idx:
        fig.add_trace(go.Scatter(
            x=coords[client_idx, 0], y=coords[client_idx, 1],
            mode="markers",
            marker=dict(
                size=11, color=_CLIENT_COLOR,
                line=dict(color="#0F172A", width=1.5),
                symbol="circle",
            ),
            name="Clients",
            hovertext=[f"<b>{labels[i]}</b><br>Client #{i}" for i in client_idx],
            hoverinfo="text",
        ))

    # --- Depots -------------------------------------------------------------
    depot_idx = list(range(n_depots))
    fig.add_trace(go.Scatter(
        x=coords[depot_idx, 0], y=coords[depot_idx, 1],
        mode="markers",
        marker=dict(
            size=18, color=_DEPOT_COLOR, symbol="square",
            line=dict(color="#78350F", width=2),
        ),
        name="Depots",
        hovertext=[f"<b>{labels[i]}</b><br>Depot #{i}" for i in depot_idx],
        hoverinfo="text",
    ))

    # --- Node labels --------------------------------------------------------
    text_annotations: list = []
    if show_node_labels:
        for i in range(n_nodes):
            text_annotations.append(dict(
                x=coords[i, 0], y=coords[i, 1],
                text=labels[i],
                showarrow=False,
                yshift=14,
                font=dict(family="Inter, monospace",
                          color=_DEPOT_COLOR if i < n_depots else _TEXT_DIM,
                          size=10),
            ))

    layout = _layout(title, width, height)
    layout["annotations"] = arrow_annotations + text_annotations

    # Equal-aspect axes with subtle grid
    layout["xaxis"] = dict(
        title=dict(text="X", font=dict(color=_TEXT_DIM)),
        gridcolor=_GRID, zerolinecolor=_GRID, color=_TEXT_DIM,
        showspikes=False,
    )
    layout["yaxis"] = dict(
        title=dict(text="Y", font=dict(color=_TEXT_DIM)),
        gridcolor=_GRID, zerolinecolor=_GRID, color=_TEXT_DIM,
        scaleanchor="x", scaleratio=1,
        showspikes=False,
    )

    fig.update_layout(layout)
    return fig


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------

def plot_convergence(
    history: Sequence[float],
    *,
    title: str = "GA Convergence",
    width: Optional[int] = 900,
    height: Optional[int] = 380,
) -> go.Figure:
    """Best-distance over generations, with a soft fill underneath."""
    h = np.asarray(history, dtype=float)
    gens = np.arange(len(h))

    fig = go.Figure()

    # Soft fill — gives the line presence without overpowering it.
    fig.add_trace(go.Scatter(
        x=gens, y=h, mode="lines",
        line=dict(color="#4ECDC4", width=0),
        fill="tozeroy", fillcolor="rgba(78, 205, 196, 0.08)",
        hoverinfo="skip", showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=gens, y=h, mode="lines",
        line=dict(color="#4ECDC4", width=2.5),
        name="Best distance",
        hovertemplate="Generation %{x}<br>Distance: %{y:.2f}<extra></extra>",
    ))

    # Mark improvement points
    improvements_mask = np.concatenate(([True], np.diff(h) < 0))
    if improvements_mask.any():
        fig.add_trace(go.Scatter(
            x=gens[improvements_mask], y=h[improvements_mask],
            mode="markers",
            marker=dict(size=6, color="#FFE66D",
                        line=dict(color="#0F172A", width=1)),
            name="Improvement",
            hovertemplate="Improvement at gen %{x}<br>Distance: %{y:.2f}<extra></extra>",
        ))

    layout = _layout(title, width, height)
    layout["xaxis"] = dict(
        title=dict(text="Generation", font=dict(color=_TEXT_DIM)),
        gridcolor=_GRID, zerolinecolor=_GRID, color=_TEXT_DIM,
    )
    layout["yaxis"] = dict(
        title=dict(text="Best total distance", font=dict(color=_TEXT_DIM)),
        gridcolor=_GRID, zerolinecolor=_GRID, color=_TEXT_DIM,
    )
    fig.update_layout(layout)
    return fig


# ---------------------------------------------------------------------------
# Vehicle utilization summary
# ---------------------------------------------------------------------------

def plot_vehicle_loads(
    solution: list,
    parameters: np.ndarray,
    capacity: Sequence[float],
    *,
    title: str = "Vehicle Load vs Capacity",
    width: Optional[int] = 900,
    height: Optional[int] = 380,
) -> go.Figure:
    """Bar chart showing total demand carried per route vs vehicle capacity."""
    demand_col = parameters[:, 0]
    routes = solution[1]
    vehicles = solution[2]

    route_labels = []
    loads = []
    caps = []
    colors = []
    for j, sub in enumerate(routes):
        if not sub:
            continue
        v_type = vehicles[j][0]
        load = float(sum(demand_col[i] for i in sub))
        cap = float(capacity[v_type])
        route_labels.append(f"#{j + 1}<br>v{v_type}")
        loads.append(load)
        caps.append(cap if np.isfinite(cap) else load)
        colors.append(_route_color(j))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=route_labels, y=caps, name="Capacity",
        marker=dict(color=_GRID, line=dict(color=_GRID, width=0)),
        hovertemplate="Route %{x}<br>Capacity: %{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=route_labels, y=loads, name="Load",
        marker=dict(color=colors, line=dict(color="#0F172A", width=1)),
        hovertemplate="Route %{x}<br>Load: %{y:.1f}<extra></extra>",
    ))

    layout = _layout(title, width, height)
    layout["barmode"] = "overlay"
    layout["bargap"] = 0.35
    layout["xaxis"] = dict(
        title=dict(text="Route · vehicle type", font=dict(color=_TEXT_DIM)),
        gridcolor=_GRID, color=_TEXT_DIM,
    )
    layout["yaxis"] = dict(
        title=dict(text="Demand units", font=dict(color=_TEXT_DIM)),
        gridcolor=_GRID, zerolinecolor=_GRID, color=_TEXT_DIM,
    )
    fig.update_layout(layout)
    return fig
