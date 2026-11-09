"""Draw the charts of BENCH.md from bench/results/summary.json.

Standard library only: writes plain SVG text, one light and one dark file per chart, into
bench/charts/. The output is deterministic, so a regenerated chart only changes when the data
does.

Run:
    python bench/charts.py
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
DATA = HERE / "results" / "summary.json"
OUT = HERE / "charts"

FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text2": "#52514e",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834"],
        "neutral": "#898781",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "text2": "#c3c2b7",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926"],
        "neutral": "#898781",
    },
}

WIDTH = 760
PAD = 24
LABEL_W = 224  # room for the row labels, right-aligned against the plot
X0 = PAD + LABEL_W + 12  # baseline of the bars
X1 = WIDTH - PAD - 64  # end of the value range, leaving room for bar-end labels
CHAR = 0.56  # average glyph width as a fraction of the font size, to place text


def num(v: float) -> str:
    """Coordinates with at most two decimals, without trailing zeros."""
    return f"{v:.2f}".rstrip("0").rstrip(".")


def fmt(value: float, chart: dict) -> str:
    text = f"{value:.{chart['decimals']}f}"
    return text + ("×" if chart["unit"] == "x" else chart["unit"])


def scale(value: float, chart: dict) -> float:
    return X0 + (X1 - X0) * value / chart["max"]


def bar(x: float, y: float, w: float, h: float, fill: str) -> str:
    """A bar from the baseline, square at the baseline and rounded (4px) at the data end."""
    r = min(4.0, w / 2, h / 2)
    end = x + w
    d = (
        f"M{num(x)} {num(y)}H{num(end - r)}A{num(r)} {num(r)} 0 0 1 {num(end)} {num(y + r)}"
        f"V{num(y + h - r)}A{num(r)} {num(r)} 0 0 1 {num(end - r)} {num(y + h)}H{num(x)}Z"
    )
    return f'<path d="{d}" fill="{fill}"/>'


def text(
    x: float, y: float, s: str, fill: str, size: int = 13, anchor: str = "start", weight: int = 400
) -> str:
    return (
        f'<text x="{num(x)}" y="{num(y)}" fill="{fill}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" dominant-baseline="central">{escape(s)}</text>'
    )


def describe(chart: dict) -> str:
    """The accessible description: every plotted value, in words."""
    if chart["kind"] == "grouped":
        parts = []
        for row in chart["rows"]:
            vals = [
                f"{name} {fmt(v, chart)}" if v is not None else f"{name} not measured"
                for name, v in zip(chart["series"], row["values"], strict=True)
            ]
            parts.append(f"{row['label']}: {', '.join(vals)}")
        return "; ".join(parts) + "."
    return "; ".join(f"{row['label']}: {fmt(row['value'], chart)}" for row in chart["rows"]) + "."


def axis(chart: dict, top: float, bottom: float, t: dict) -> list[str]:
    out = []
    for tick in chart["ticks"]:
        x = scale(tick, chart)
        if tick:
            out.append(
                f'<line x1="{num(x)}" y1="{num(top)}" x2="{num(x)}" y2="{num(bottom)}" stroke="{t["grid"]}" stroke-width="1"/>'
            )
        label = f"{tick:g}" + ("×" if chart["unit"] == "x" else "")
        out.append(text(x, bottom + 14, label, t["text2"], 12, "middle"))
    out.append(
        f'<line x1="{num(X0)}" y1="{num(top)}" x2="{num(X0)}" y2="{num(bottom)}" stroke="{t["axis"]}" stroke-width="1"/>'
    )
    out.append(text((X0 + X1) / 2, bottom + 36, chart["axis"], t["text2"], 12, "middle"))
    return out


def grouped(chart: dict, t: dict) -> tuple[list[str], float]:
    bar_h, gap, group_gap = 12, 2, 16
    n = len(chart["series"])
    out = []
    # Legend: a swatch beside each series name, text in text colors.
    x = X0
    for i, name in enumerate(chart["series"]):
        out.append(
            f'<rect x="{num(x)}" y="74" width="12" height="12" rx="2" fill="{t["series"][i]}"/>'
        )
        out.append(text(x + 18, 80, name, t["text"], 13))
        x += 18 + len(name) * 13 * CHAR + 28
    top = 104
    group_h = n * bar_h + (n - 1) * gap
    y = top + 8
    for row in chart["rows"]:
        out.append(text(X0 - 12, y + group_h / 2, row["label"], t["text"], 13, "end"))
        for i, value in enumerate(row["values"]):
            by = y + i * (bar_h + gap)
            if value is None:
                out.append(text(X0 + 6, by + bar_h / 2, chart["missing_label"], t["text2"], 11))
                continue
            x1 = scale(value, chart)
            out.append(bar(X0, by, x1 - X0, bar_h, t["series"][i]))
            out.append(text(x1 + 6, by + bar_h / 2, fmt(value, chart), t["text"], 11))
        y += group_h + group_gap
    bottom = y - group_gap + 8
    ref = chart["reference"]
    rx = scale(ref["value"], chart)
    out.append(
        f'<line x1="{num(rx)}" y1="{num(top - 6)}" x2="{num(rx)}" y2="{num(bottom)}" stroke="{t["text2"]}" stroke-width="1"/>'
    )
    out.append(text(rx + 4, top - 2, ref["label"], t["text2"], 11))
    return axis(chart, top, bottom, t) + out, bottom


def single(chart: dict, t: dict) -> tuple[list[str], float]:
    bar_h, step = 16, 28
    top = 80
    out = []
    y = top + 8
    for row in chart["rows"]:
        fill = t["series"][0] if row["ours"] else t["neutral"]
        x1 = scale(row["value"], chart)
        out.append(text(X0 - 12, y + bar_h / 2, row["label"], t["text"], 13, "end"))
        out.append(bar(X0, y, x1 - X0, bar_h, fill))
        out.append(text(x1 + 6, y + bar_h / 2, fmt(row["value"], chart), t["text"], 12))
        y += step
    bottom = y - step + bar_h + 8
    return axis(chart, top, bottom, t) + out, bottom


def render(chart: dict, mode: str) -> str:
    t = THEMES[mode]
    body, bottom = (grouped if chart["kind"] == "grouped" else single)(chart, t)
    height = bottom + 36 + PAD
    head = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{num(height)}" '
        f'viewBox="0 0 {WIDTH} {num(height)}" role="img" aria-labelledby="title desc" '
        f'font-family="{FONT}">',
        f'<title id="title">{escape(chart["title"])}</title>',
        f'<desc id="desc">{escape(describe(chart))}</desc>',
        f'<rect width="{WIDTH}" height="{num(height)}" rx="6" fill="{t["surface"]}"/>',
        text(PAD, 30, chart["title"], t["text"], 16, weight=600),
        text(PAD, 52, chart["subtitle"], t["text2"], 13),
    ]
    return "\n".join([*head, *body, "</svg>"]) + "\n"


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    for chart in data["charts"]:
        for mode in THEMES:
            path = OUT / f"{chart['name']}-{mode}.svg"
            path.write_text(render(chart, mode), encoding="utf-8")
            print(path.relative_to(HERE.parent))


if __name__ == "__main__":
    main()
