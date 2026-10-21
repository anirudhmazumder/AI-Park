"""
kg_plot.py - plotting helpers shared by the KG_Visualization scripts.

Edge colours encode the LLM-assigned verb category of the first triplet that
supports an edge:
  RED   = activating / promoting / binding interactions
  BLUE  = inhibiting / degrading / removing interactions
  BLACK = "I don't know" (unclassified by the model)
"""

from __future__ import annotations

import copy
import os
import re

import plotly.graph_objects as go

RED_CATEGORIES = {
    "Bind", "Form Complex", "Activate", "Phosphorylate", "Amplify",
    "Recruit", "Trigger", "Aggregate", "Transmit", "Transport",
    "Anchor", "Reconfigure", "Catalyze", "Modify",
}
BLUE_CATEGORIES = {
    "Inhibit", "Dephosphorylate", "Degrade", "Cleave", "Expel",
    "Neutralize", "Deacetylate", "Dissociate", "Terminate",
}

EDGE_COLOR_RED   = "rgba(220, 50,  50,  0.65)"
EDGE_COLOR_BLUE  = "rgba(30,  100, 210, 0.65)"
EDGE_COLOR_BLACK = "rgba(30,  30,  30,  0.55)"
EDGE_COLOR_GRAY  = "rgba(140, 140, 140, 0.25)"   # fallback (does not occur in the current data)

COLOR_META = {
    "red":   ("Activating / binding",   EDGE_COLOR_RED),
    "blue":  ("Inhibiting / degrading", EDGE_COLOR_BLUE),
    "black": ("Unknown / unclassified", EDGE_COLOR_BLACK),
    "gray":  ("Unclassified",           EDGE_COLOR_GRAY),
}

_SUGGESTED = [
    r"suggested category:\s*['\"]?([A-Za-z]+)['\"]?",
    r"suggested:\s*['\"]?([A-Za-z]+)['\"]?",
    r"suggest(?:ed)?\s+category:\s*['\"]?([A-Za-z]+)['\"]?",
]


def categorize_verb(verb_category) -> str:
    """Return 'red', 'blue', 'black' or 'gray' for a raw verb-category string."""
    if not verb_category:
        return "gray"
    vc = verb_category.strip()
    if vc in RED_CATEGORIES:
        return "red"
    if vc in BLUE_CATEGORIES:
        return "blue"
    low = vc.lower()
    if low.startswith("i don't know") or low.startswith("i don’t know"):
        for pattern in _SUGGESTED:
            m = re.search(pattern, vc, re.IGNORECASE)
            if m:
                suggested = m.group(1).strip().title()
                if suggested in RED_CATEGORIES:
                    return "red"
                if suggested in BLUE_CATEGORIES:
                    return "blue"
                break
        return "black"
    return "gray"


def verb_color(verb_category) -> str:
    return COLOR_META[categorize_verb(verb_category)][1]


def legend_traces():
    """Three invisible traces that drive the edge-type legend."""
    return [
        go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=color, width=3),
                   name=label, showlegend=True)
        for key, (label, color) in COLOR_META.items() if key != "gray"
    ]


def arrow_annotations(G, pos):
    """One coloured arrow annotation per edge (subject -> object as stored)."""
    out = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        out.append(dict(
            ax=x0, ay=y0, x=x1, y=y1,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2,
            arrowcolor=verb_color(data.get("verb_category")), opacity=0.85,
        ))
    return out


def thin_edge_trace(G, pos, width=0.4, color="rgba(180,180,180,0.15)"):
    xs, ys = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        xs.extend([x0, x1, None])
        ys.extend([y0, y1, None])
    return go.Scatter(x=xs, y=ys, line=dict(width=width, color=color),
                      hoverinfo="none", mode="lines", showlegend=False)


def base_layout(title_html, width=1500, height=900, legend_title="Edge type", **extra):
    layout = dict(
        title=dict(text=title_html, font=dict(size=18), x=0.5, xanchor="center"),
        hovermode="closest",
        margin=dict(b=40, l=40, r=200, t=110),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=True,
        legend=dict(x=1.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.15)",
                    borderwidth=1, font=dict(size=11),
                    title=dict(text=legend_title, font=dict(size=12))),
        width=width, height=height,
    )
    layout.update(extra)
    return layout


_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def strip_control_chars(fig):
    """
    Remove C0 control characters (OCR artefacts in a few entity names) from the
    text and hover labels of a figure.  They render as boxes in a browser and
    make the static-image renderer fail.
    """
    for tr in fig.data:
        for attr in ("text", "hovertext"):
            val = getattr(tr, attr, None)
            if isinstance(val, (list, tuple)):
                setattr(tr, attr, [_CONTROL.sub("", v) if isinstance(v, str) else v for v in val])
            elif isinstance(val, str):
                setattr(tr, attr, _CONTROL.sub("", val))
    return fig


def save_figure(fig, html_path, png_path=None, open_browser=False, png_size=(1600, 1200), scale=2):
    """Write the interactive HTML and, optionally, a static PNG (needs kaleido + Chrome)."""
    strip_control_chars(fig)
    os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
    config = {"displaylogo": False, "scrollZoom": True,
              "toImageButtonOptions": {"format": "png", "height": png_size[1],
                                       "width": png_size[0], "scale": scale,
                                       "filename": os.path.splitext(os.path.basename(html_path))[0]}}
    fig.write_html(html_path, config=config, include_plotlyjs=True, auto_open=open_browser)
    print(f"  Saved -> {html_path}")
    if png_path:
        os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
        try:
            fig.write_image(png_path, width=png_size[0], height=png_size[1], scale=scale)
            print(f"  Saved -> {png_path}")
        except Exception as e:  # kaleido/Chrome missing
            print(f"  PNG export skipped ({type(e).__name__}: {str(e)[:120]}). "
                  f"Install `kaleido` (and Chrome) or use the camera button in the HTML.")


# Manuscript (print) versions
# The figures in the manuscript are the plot area only: no title, legend or
# colour bar.  PRINT_SIZE is the plot area of the interactive figures
# (1500 x 950 canvas minus margins), which gives the same aspect ratio.
PRINT_SIZE = (1260, 800)
PRINT_DIR_NAME = "manuscript"


PRINT_LABEL_SIZE = 16   # node-label font size in the print versions (interactive figures use 8-9)


def manuscript_version(fig, label_size=PRINT_LABEL_SIZE):
    """
    Copy of ``fig`` reduced to the plot area (no title, legend, colour bar,
    margins) with node labels enlarged to ``label_size`` for print.
    """
    f = copy.deepcopy(fig)
    f.layout.title = None
    f.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                    width=PRINT_SIZE[0], height=PRINT_SIZE[1])
    xs, ys = [], []
    for tr in f.data:
        if "markers" in (getattr(tr, "mode", "") or "") and tr.x is not None:
            xs += [v for v in tr.x if v is not None]
            ys += [v for v in tr.y if v is not None]
    if xs:
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        dx, dy = (x1 - x0) or 1.0, (y1 - y0) or 1.0
    for tr in f.data:
        marker = getattr(tr, "marker", None)
        if marker is not None and getattr(marker, "showscale", None):
            marker.showscale = False
        if label_size and "text" in (getattr(tr, "mode", "") or ""):
            tr.textfont.size = label_size
            if xs:
                # Keep labels inside the frame: nodes near the right/left edge get
                # their label on the inner side, nodes near the top get it below.
                pos = []
                for x, y in zip(tr.x, tr.y):
                    vert = "bottom" if y > y1 - 0.06 * dy else "top"
                    horiz = "left" if x > x1 - 0.12 * dx else ("right" if x < x0 + 0.12 * dx else "center")
                    pos.append(f"{vert} {horiz}")
                tr.textposition = pos
    if xs:
        f.update_xaxes(range=[x0 - 0.03 * dx, x1 + 0.03 * dx])
        f.update_yaxes(range=[y0 - 0.04 * dy, y1 + 0.04 * dy])
    return f


def save_manuscript_png(fig, png_path, scale=2, label_size=PRINT_LABEL_SIZE):
    """Write the print version of ``fig`` as a PNG (needs kaleido + Chrome)."""
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    f = manuscript_version(strip_control_chars(fig), label_size=label_size)
    try:
        f.write_image(png_path, width=PRINT_SIZE[0], height=PRINT_SIZE[1], scale=scale)
        print(f"  Saved -> {png_path}")
    except Exception as e:  # kaleido/Chrome missing
        print(f"  Manuscript PNG skipped ({type(e).__name__}: {str(e)[:120]}).")
