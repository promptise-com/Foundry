#!/usr/bin/env python3
"""Draw the README illustrations: ``docs/assets/readme/*-{light,dark}.svg``.

    uv run --no-project --with fonttools --with pillow python scripts/build_readme_art.py

The art follows the promptise.com design system: one ink colour, hairlines,
square marks, Geist and Geist Mono. Text is converted to outlines, so the
images look the same whether or not a reader has the fonts installed.
Motion is SVG/SMIL, which GitHub renders inside a README image; readers who
ask for reduced motion get the still drawing.

The fonts are downloaded from Google Fonts into a cache on first run; they
are not committed.
"""

from __future__ import annotations

import base64
import io
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
HERE = ASSETS / "readme"
CACHE = Path.home() / ".cache" / "promptise-readme"
FONT_URLS = {
    "sans": "https://fonts.gstatic.com/s/geist/v5/gyBhhwUxId8gMGYQMKR3pzfaWI_RnOM4nQ.ttf",
    "mono": "https://fonts.gstatic.com/s/geistmono/v6/or3yQ6H-1_WfwkMZI_qYPLs1a-t7PU0AbeE9KJ5T.ttf",
}


# ── type ─────────────────────────────────────────────────────────────────────


def _n(v: float) -> str:
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


class Font:
    def __init__(self, path: Path) -> None:
        self.f = TTFont(path)
        self.glyphs = self.f.getGlyphSet()
        self.cmap = self.f.getBestCmap()
        self.upm = self.f["head"].unitsPerEm
        self.hmtx = self.f["hmtx"]

    def _glyph(self, ch: str) -> str:
        return self.cmap.get(ord(ch)) or self.cmap.get(ord("?"))

    def width(self, text: str, size: float, tracking: float = 0.0) -> float:
        s = size / self.upm
        return (
            sum(self.hmtx[self._glyph(c)][0] * s + tracking * size for c in text) - tracking * size
        )

    def outline(self, text: str, x: float, y: float, size: float, tracking: float = 0.0) -> str:
        s = size / self.upm
        parts = []
        for ch in text:
            g = self._glyph(ch)
            pen = SVGPathPen(self.glyphs, ntos=_n)
            self.glyphs[g].draw(TransformPen(pen, (s, 0, 0, -s, x, y)))
            cmd = pen.getCommands()
            if cmd:
                parts.append(cmd)
            x += self.hmtx[g][0] * s + tracking * size
        return "".join(parts)


def _fonts() -> dict[str, Font]:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, url in FONT_URLS.items():
        path = CACHE / f"{key}.ttf"
        if not path.exists():
            with urllib.request.urlopen(url) as r:  # noqa: S310 — fixed https URL
                path.write_bytes(r.read())
        out[key] = Font(path)
    return out


F = _fonts()
SANS, MONO = F["sans"], F["mono"]


def text(
    s: str,
    x: float,
    y: float,
    size: float,
    fill: str,
    *,
    font: Font = SANS,
    tracking: float = -0.03,
    anchor: str = "start",
    opacity: float = 1.0,
    upper: bool = False,
) -> str:
    if upper:
        s = s.upper()
    w = font.width(s, size, tracking)
    if anchor == "middle":
        x -= w / 2
    elif anchor == "end":
        x -= w
    op = f' fill-opacity="{opacity:g}"' if opacity < 1 else ""
    return f'<path fill="{fill}"{op} d="{font.outline(s, x, y, size, tracking)}"/>'


def mono(s: str, x: float, y: float, size: float, fill: str, **kw) -> str:
    kw.setdefault("tracking", -0.02)
    kw.setdefault("upper", True)
    return text(s, x, y, size, fill, font=MONO, **kw)


def wrap(
    s: str, size: float, max_w: float, *, font: Font = SANS, tracking: float = -0.03
) -> list[str]:
    lines, line = [], ""
    for word in s.split():
        trial = f"{line} {word}".strip()
        if line and font.width(trial, size, tracking) > max_w:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


# ── themes ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    ink: str
    muted: str
    faint: str
    line: str
    logo: str


LIGHT = Theme("light", "#ffffff", "#1a1a1a", "#555555", "#8f8f8f", "#e3e3e3", "logo-dark.png")
DARK = Theme("dark", "#0d1117", "#ededed", "#a9a9a9", "#727272", "#2b3139", "logo-light.png")

REDUCED = (
    "<style>"
    "@media (prefers-reduced-motion: reduce){.m{display:none}}"
    "@media (prefers-reduced-motion: no-preference){.s{display:none}}"
    "</style>"
)


def svg(w: int, h: int, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'role="img" aria-label="{title}"><title>{title}</title>{REDUCED}{body}</svg>\n'
    )


def save(name: str, theme: Theme, content: str) -> None:
    path = HERE / f"{name}-{theme.name}.svg"
    path.write_text(content, encoding="utf-8")
    print(f"{path.relative_to(ROOT)}  {len(content) / 1024:.0f} KB")


# ── hero: the foundation layer as a plane of square dots ────────────────────


def _smooth(a: float, b: float, v: float) -> float:
    k = min(1.0, max(0.0, (v - a) / (b - a)))
    return k * k * (3 - 2 * k)


def dot_field(w: float, h: float, x0: float, y0: float, color: str, *, clear_left: float) -> str:
    """The website hero's dot plane, frozen at one moment."""
    horizon = h * 0.40
    f = (h - horizon) * 1.1
    cam = 1.0
    z_near = (cam * f) / (h - horizon + 8)
    z_far = z_near * 6.5
    step = 46 / (h - horizon)
    turn = 0.42
    cos, sin = math.cos(turn), math.sin(turn)
    t = 8.0

    def height(x: float, z: float) -> float:
        return 0.034 * math.sin(2.1 * x + 0.5 * t) * math.cos(1.5 * z - 0.32 * t) + 0.02 * math.sin(
            1.1 * (x - z) + 0.75 * t
        )

    half_far = (w / 2 / f) * z_far
    corners = [
        (-half_far, z_far),
        (half_far, z_far),
        ((-w / 2 / f) * z_near, z_near),
        ((w / 2 / f) * z_near, z_near),
    ]
    uv = [(x * cos + z * sin, -x * sin + z * cos) for x, z in corners]
    u0 = math.floor(min(p[0] for p in uv) / step) * step
    u1 = max(p[0] for p in uv)
    v0 = math.floor(min(p[1] for p in uv) / step) * step
    v1 = max(p[1] for p in uv)
    buckets: dict[float, list[str]] = {}
    v = v0
    while v <= v1:
        u = u0
        while u <= u1:
            x = u * cos - v * sin
            z = u * sin + v * cos
            u += step
            if z < z_near or z > z_far:
                continue
            sx = w / 2 + (x * f) / z
            if sx < 0 or sx > w:
                continue
            mask = _smooth(w * clear_left, w * min(0.98, clear_left + 0.36), sx)
            fog = _smooth(z_far, z_near * 2.6, z)
            if mask * fog < 0.05:
                continue
            y = height(x, z)
            sy = horizon + ((cam - y) * f) / z
            if sy < 0 or sy > h:
                continue
            lift = 0.8 + max(-0.5, min(1.4, y * 14))
            a = min(0.9, 0.7 * fog * mask * lift)
            if a < 0.06:
                continue
            s = max(1.6, min(5.0, (4.6 * z_near) / z))
            key = round(a * 20) / 20
            buckets.setdefault(key, []).append(
                f"M{_n(x0 + sx - s / 2)} {_n(y0 + sy - s / 2)}h{_n(s)}v{_n(s)}h-{_n(s)}z"
            )
        v += step
    return "".join(
        f'<path fill="{color}" fill-opacity="{a:g}" d="{"".join(d)}"/>'
        for a, d in sorted(buckets.items())
    )


def _logo(theme: Theme, size: int) -> str:
    img = Image.open(ASSETS / theme.logo).convert("RGBA")
    img = img.crop(img.getbbox())
    img.thumbnail((size * 2, size * 2))
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def hero(theme: Theme) -> str:
    W, H = 1600, 780
    b = [
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" fill="{theme.bg}" stroke="{theme.line}"/>'
    ]
    b.append(dot_field(W - 2, H - 2, 1, 1, theme.ink, clear_left=0.46))
    # crosses on the plane, as on the website
    for cx, cy, r in ((1180, 470, 9), (1420, 560, 7), (1310, 680, 12), (1500, 420, 6)):
        b.append(
            f'<path stroke="{theme.ink}" stroke-opacity="0.55" d="M{cx - r} {cy}h{2 * r}M{cx} {cy - r}v{2 * r}"/>'
        )
    logo = _logo(theme, 44)
    b.append(f'<image x="96" y="84" width="44" height="44" href="data:image/png;base64,{logo}"/>')
    b.append(text("Promptise Foundry", 158, 118, 32, theme.ink, tracking=-0.035))
    b.append(mono("Open source · Python 3.10+ · Apache 2.0", 96, 230, 22, theme.faint))
    b.append(text("The foundation layer", 92, 336, 96, theme.ink, tracking=-0.055))
    b.append(text("for agentic intelligence.", 92, 438, 96, theme.ink, tracking=-0.055))
    para = (
        "An open-source framework for the whole agentic stack. Build agents that reason, give them "
        "safe access to your systems, and run them in production with full control."
    )
    for i, line in enumerate(wrap(para, 31, 860, tracking=-0.025)):
        b.append(text(line, 96, 514 + i * 44, 31, theme.muted, tracking=-0.025))
    b.append(
        f'<rect x="96.5" y="640.5" width="430" height="68" fill="{theme.bg}" stroke="{theme.ink}" stroke-opacity="0.35"/>'
    )
    b.append(mono("$", 122, 683, 25, theme.faint, upper=False))
    b.append(mono("pip install promptise", 156, 683, 25, theme.ink, upper=False))
    return svg(W, H, "".join(b), "Promptise Foundry: the foundation layer for agentic intelligence")


# ── the three parts ─────────────────────────────────────────────────────────

STROKE = 'fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"'
STROKE_BOLD = STROKE.replace('stroke-width="1"', 'stroke-width="1.5"')


def wire_branch() -> str:
    routes = ["M24 78H62L102 40L132 78", "M24 78H132", "M24 78H62L102 116L132 78"]
    out = [
        f"<g {STROKE}>"
        '<circle cx="24" cy="78" r="6"/><path d="M30 78h26"/><circle cx="62" cy="78" r="6"/>'
        '<path d="M67 74 L96 44M67 82 L96 112M68 78h28" opacity="0.7"/>'
        '<circle cx="102" cy="40" r="5"/><circle cx="102" cy="78" r="5"/><circle cx="102" cy="116" r="5"/>'
        '<path d="M107 40 L128 66M107 78h21M107 116 L128 90" opacity="0.45"/><circle cx="132" cy="78" r="6"/></g>'
    ]
    out.append('<g class="m">')
    for i, d in enumerate(routes):
        out.append(
            f'<path d="{d}" {STROKE_BOLD} opacity="0"><animate attributeName="opacity" '
            f'values="0;0.9;0.9;0;0" keyTimes="0;0.04;0.3;0.36;1" dur="4.5s" begin="{i * 1.5}s" '
            f'repeatCount="indefinite"/></path>'
            f'<circle r="3" fill="currentColor" opacity="0"><animateMotion path="{d}" dur="4.5s" '
            f'begin="{i * 1.5}s" repeatCount="indefinite" keyPoints="0;1;1" keyTimes="0;0.3;1" calcMode="linear"/>'
            f'<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;0.3;0.31;1" dur="4.5s" '
            f'begin="{i * 1.5}s" repeatCount="indefinite"/></circle>'
        )
    out.append("</g>")
    return "".join(out)


def wire_hub() -> str:
    spokes = [(112, 36), (136, 60), (136, 96), (112, 120), (40, 36), (16, 60), (16, 96), (40, 120)]
    out = [
        f'<g {STROKE}><ellipse cx="76" cy="78" rx="30" ry="15"/><path d="M46 78v18c0 8.3 13.4 15 30 15s30-6.7 30-15V78"/>'
    ]
    for x, y in spokes:
        out.append(f'<path d="M76 78 L{x} {y}" opacity="0.45"/><circle cx="{x}" cy="{y}" r="5"/>')
    out.append('</g><g class="m">')
    out.append(
        '<ellipse cx="76" cy="78" rx="8" ry="4" fill="none" stroke="currentColor">'
        '<animate attributeName="rx" values="8;30" dur="2.4s" repeatCount="indefinite"/>'
        '<animate attributeName="ry" values="4;15" dur="2.4s" repeatCount="indefinite"/>'
        '<animate attributeName="opacity" values="0.7;0" dur="2.4s" repeatCount="indefinite"/></ellipse>'
    )
    for i, (x, y) in enumerate(spokes):
        out.append(
            f'<circle r="2.5" fill="currentColor" opacity="0"><animateMotion path="M76 78 L{x} {y}" dur="4.8s" '
            f'begin="{i * 0.6:g}s" repeatCount="indefinite" keyPoints="0;1;1" keyTimes="0;0.125;1" calcMode="linear"/>'
            f'<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;0.125;0.126;1" dur="4.8s" '
            f'begin="{i * 0.6:g}s" repeatCount="indefinite"/></circle>'
            f'<circle cx="{x}" cy="{y}" r="5" fill="currentColor" opacity="0"><animate attributeName="opacity" '
            f'values="0;0;0.85;0;0" keyTimes="0;0.12;0.15;0.32;1" dur="4.8s" begin="{i * 0.6:g}s" '
            f'repeatCount="indefinite"/></circle>'
        )
    out.append("</g>")
    return "".join(out)


def wire_gate() -> str:
    return (
        f'<g {STROKE}><path d="M30 34v88M122 34v88"/><ellipse cx="30" cy="34" rx="10" ry="5"/>'
        '<ellipse cx="122" cy="34" rx="10" ry="5"/><path d="M58 96 L76 86 L94 96 L76 106 Z"/>'
        '<path d="M76 106v14" opacity="0.45"/><circle cx="76" cy="128" r="5"/>'
        '<path class="s" d="M30 62h92M30 70h92" opacity="0.5"/></g>'
        f'<g class="m"><path d="M30 62h92M30 70h92" {STROKE} opacity="0.5">'
        '<animateTransform attributeName="transform" type="translate" values="0 0;0 0;0 -14;0 -14;0 0;0 0" '
        'keyTimes="0;0.45;0.55;0.8;0.9;1" dur="5s" repeatCount="indefinite"/></path>'
        '<path d="M58 96 L76 86 L94 96 L76 106 Z" fill="currentColor" opacity="0"><animate attributeName="opacity" '
        'values="0;0;0.85;0.85;0;0" keyTimes="0;0.22;0.3;0.45;0.55;1" dur="5s" repeatCount="indefinite"/></path>'
        '<circle cx="76" cy="132" r="3" fill="currentColor" opacity="0">'
        '<animate attributeName="cy" values="132;96;96;26;26" keyTimes="0;0.2;0.55;0.8;1" dur="5s" repeatCount="indefinite"/>'
        '<animate attributeName="opacity" values="0;1;1;1;0;0" keyTimes="0;0.05;0.55;0.76;0.8;1" dur="5s" '
        'repeatCount="indefinite"/></circle></g>'
    )


LAYERS = [
    (
        "01",
        "Agent",
        "It thinks",
        wire_branch,
        "Agents that work through a task instead of guessing: they plan, use tools, check their own output "
        "and answer, with prompts managed like code.",
        ["The Promptise Agent", "Reasoning Engine", "Execution Engine", "Prompt & Context"],
    ),
    (
        "02",
        "Interface",
        "It acts",
        wire_hub,
        "Safe access to your systems. Build MCP servers with authentication and audit built in, connect to "
        "anyone else's, or turn the API you already have into one an agent can use.",
        ["MCP Server SDK", "MCP Client", "MCPcast"],
    ),
    (
        "03",
        "Harness",
        "It operates",
        wire_gate,
        "What keeps agents running and accountable: schedules and crash recovery so the work gets done, budgets "
        "and health checks so costs stay bounded, identity and audit so you can prove what happened.",
        ["Agent Runtime", "Governance & Controls", "Agent Identity", "Observability"],
    ),
]


def stack(theme: Theme) -> str:
    W, H = 1600, 880
    col = W / 3
    b = [
        f'<path stroke="{theme.line}" d="M0 0.5H{W}M0 {H - 0.5}H{W}M{col:.1f} 0V{H}M{2 * col:.1f} 0V{H}"/>'
    ]
    for i, (n, name, verb, art, definition, modules) in enumerate(LAYERS):
        x = i * col + 44
        b.append(mono(f"[{n}]", x, 66, 22, theme.faint))
        b.append(mono(verb, i * col + col - 44, 66, 22, theme.faint, anchor="end"))
        b.append(
            f'<svg x="{x - 8:.1f}" y="92" width="180" height="185" viewBox="0 0 152 156" color="{theme.ink}">{art()}</svg>'
        )
        b.append(text(name, x, 350, 58, theme.ink, tracking=-0.05))
        for k, line in enumerate(wrap(definition, 26, col - 88, tracking=-0.02)):
            b.append(text(line, x, 404 + k * 37, 26, theme.muted, tracking=-0.02))
        b.append(mono("Inside", x, 682, 20, theme.faint))
        for k, m in enumerate(modules):
            b.append(
                f'<rect x="{x:.1f}" y="{703 + k * 40}" width="7" height="7" fill="{theme.ink}" fill-opacity="0.7"/>'
            )
            b.append(mono(m, x + 22, 712 + k * 40, 22, theme.ink, upper=False))
    return svg(
        W, H, "".join(b), "Three parts of every agentic system: Agent, Interface and Harness"
    )


# ── one request, across the three layers ───────────────────────────────────

STEPS = [
    ("harness", "Agent Runtime", "A ticket wakes the agent"),
    ("harness", "Agent Identity", "It knows who it is"),
    ("harness", "Guardrails", "The input is checked"),
    ("agent", "Prompt & Context", "It gathers what it knows"),
    ("agent", "The Promptise Agent", "No shortcut this time"),
    ("agent", "Reasoning Engine", "It makes a plan"),
    ("interface", "MCP Client", "It reaches for a tool"),
    ("interface", "MCP Server SDK", "The server checks the caller"),
    ("harness", "Sandbox", "It double-checks the numbers"),
    ("interface", "MCPcast", "A tool generated from a spec"),
    ("interface", "Approval gate", "A person signs off"),
    ("agent", "Reasoning Engine", "It checks its own work"),
    ("agent", "Guardrails · Streaming", "The answer goes out"),
    ("harness", "Observability · Governance", "Everything is on the record"),
]


def _cubic(p0, p1, p2, p3, t):
    u = 1 - t
    return (
        u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
        u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
    )


def flow(theme: Theme) -> str:
    W, H = 1600, 700
    lane_y = {"interface": 104, "agent": 224, "harness": 344}
    names = {
        "interface": ("Interface", "It acts"),
        "agent": ("Agent", "It thinks"),
        "harness": ("Harness", "It operates"),
    }
    X0, X1 = 290, 1556
    stops = [
        (X0 + (X1 - X0) * k / (len(STEPS) - 1), lane_y[lane])
        for k, (lane, _, _) in enumerate(STEPS)
    ]
    b = []
    for lane, y in lane_y.items():
        b.append(
            f'<path stroke="{theme.ink}" stroke-opacity="0.12" stroke-dasharray="2 7" d="M220 {y}H{W - 20}"/>'
        )
        b.append(text(names[lane][0], 24, y + 4, 30, theme.ink, tracking=-0.04))
        b.append(mono(names[lane][1], 24, y + 34, 18, theme.faint))
    # the whole route, and its length at each stop
    d = [f"M{stops[0][0]:.1f} {stops[0][1]}"]
    lengths = [0.0]
    for a, c in zip(stops, stops[1:], strict=False):  # consecutive pairs
        if a[1] == c[1]:
            d.append(f"H{c[0]:.1f}")
            lengths.append(lengths[-1] + abs(c[0] - a[0]))
        else:
            mx = (a[0] + c[0]) / 2
            p1, p2 = (mx, a[1]), (mx, c[1])
            d.append(f"C{mx:.1f} {a[1]} {mx:.1f} {c[1]} {c[0]:.1f} {c[1]}")
            pts = [_cubic(a, p1, p2, c, i / 200) for i in range(201)]
            lengths.append(lengths[-1] + sum(math.dist(pts[i], pts[i + 1]) for i in range(200)))
    route = "".join(d)
    total = lengths[-1]
    frac = [ln / total for ln in lengths]
    b.append(f'<path d="{route}" fill="none" stroke="{theme.ink}" stroke-opacity="0.2"/>')

    n = len(STEPS)
    loop = n * 2.2
    times, points = [0.0], [0.0]
    for k in range(1, n):
        start = k / n
        times += [start, start + 0.32 / n]
        points += [frac[k - 1], frac[k]]
    times.append(1.0)
    points.append(1.0)
    kt = ";".join(f"{t:.4f}" for t in times)
    kp = ";".join(f"{p:.4f}" for p in points)
    offsets = ";".join(f"{total * (1 - p):.1f}" for p in points)
    b.append('<g class="m">')
    b.append(
        f'<path d="{route}" fill="none" stroke="{theme.ink}" stroke-width="1.8" stroke-dasharray="{total:.1f}" '
        f'stroke-dashoffset="{total:.1f}"><animate attributeName="stroke-dashoffset" values="{offsets}" '
        f'keyTimes="{kt}" dur="{loop:g}s" repeatCount="indefinite"/></path>'
        f'<circle r="5" fill="{theme.ink}"><animateMotion path="{route}" keyPoints="{kp}" keyTimes="{kt}" '
        f'calcMode="linear" dur="{loop:g}s" repeatCount="indefinite"/></circle>'
    )
    b.append("</g>")
    b.append(f'<path class="s" d="{route}" fill="none" stroke="{theme.ink}" stroke-width="1.8"/>')
    for k, (x, y) in enumerate(stops):
        b.append(
            f'<circle cx="{x:.1f}" cy="{y}" r="6.5" fill="{theme.bg}" stroke="{theme.ink}" stroke-opacity="0.55"/>'
        )
        arrive = (k / n) + (0.32 / n if k else 0.0)
        b.append(
            f'<circle class="m" cx="{x:.1f}" cy="{y}" r="6.5" fill="{theme.ink}" opacity="0"><animate attributeName="opacity" '
            f'values="0;0;1;1;0" keyTimes="0;{max(0.0001, arrive - 0.0001):.4f};{arrive + 0.0001:.4f};0.995;1" '
            f'dur="{loop:g}s" repeatCount="indefinite"/></circle>'
        )
        b.append(f'<circle class="s" cx="{x:.1f}" cy="{y}" r="6.5" fill="{theme.ink}"/>')
        b.append(mono(f"{k + 1:02d}", x, y + 40, 18, theme.faint, anchor="middle"))
    # the step being shown, as words underneath
    b.append(f'<path stroke="{theme.line}" d="M24 {H - 200}H{W - 24}"/>')
    for k, (_, where, title) in enumerate(STEPS):
        a, z = k / n, (k + 1) / n
        # step 1 is on screen from the first frame, so a still capture is never blank
        values, times = (
            ("1;1;0;0", f"0;{z - 0.004:.4f};{z:.4f};1")
            if k == 0
            else (
                "0;0;1;1;0;0",
                f"0;{a:.4f};{a + 0.004:.4f};{z - 0.004:.4f};{min(0.9999, z):.4f};1",
            )
        )
        b.append(
            f'<g class="m" opacity="{1 if k == 0 else 0}"><animate attributeName="opacity" values="{values}" '
            f'keyTimes="{times}" dur="{loop:g}s" repeatCount="indefinite"/>'
            + mono(f"Step {k + 1:02d} / {n} · {where}", 24, H - 130, 22, theme.faint)
            + text(title, 24, H - 58, 54, theme.ink, tracking=-0.045)
            + "</g>"
        )
    b.append(
        '<g class="s">'
        + mono("14 steps · 12 modules · 3 layers", 24, H - 130, 22, theme.faint)
        + text(
            "One customer request, from webhook to audit record.",
            24,
            H - 58,
            54,
            theme.ink,
            tracking=-0.045,
        )
        + "</g>"
    )
    return svg(
        W,
        H,
        "".join(b),
        "One customer request handled across the Interface, Agent and Harness layers",
    )


# ── MCPcast: a spec in, a reviewed server out ───────────────────────────────

OPS = [
    ("PUT", "/pet", "update_pet", True),
    ("POST", "/pet", "add_pet", True),
    ("GET", "/pet/findByStatus", "find_pets_by_status", False),
    ("GET", "/pet/findByTags", "find_pets_by_tags", False),
    ("GET", "/pet/{petId}", "get_pet_by_id", False),
    ("POST", "/pet/{petId}", "update_pet_with_form", True),
    ("DELETE", "/pet/{petId}", None, False),
    ("GET", "/store/inventory", "get_inventory", False),
    ("POST", "/store/order", "place_order", True),
    ("GET", "/store/order/{orderId}", "get_order_by_id", False),
    ("DELETE", "/store/order/{orderId}", None, False),
]


def mcpcast_art(theme: Theme) -> str:
    W, H = 1600, 860
    b = []
    loop = 9.0
    pw = 620  # panel width
    lx, rx, top, row = 24, W - 24 - pw, 150, 54
    fh = 20  # small labels
    # left: the spec
    b.append(
        f'<rect x="{lx}.5" y="40.5" width="{pw}" height="{H - 81}" fill="none" stroke="{theme.line}"/>'
    )
    b.append(mono("openapi.json", lx + 30, 92, 22, theme.ink))
    b.append(mono("11 operations", lx + pw - 30, 92, 22, theme.faint, anchor="end"))
    b.append(f'<path stroke="{theme.line}" d="M{lx} 118.5H{lx + pw}"/>')
    for i, (method, path, tool, gated) in enumerate(OPS):
        y = top + 30 + i * row
        b.append(mono(method, lx + 30, y, fh, theme.faint))
        b.append(
            mono(path, lx + 150, y, 23, theme.ink, upper=False, opacity=0.4 if tool is None else 1)
        )
        if tool is None:
            w = MONO.width(path, 23, -0.02)
            b.append(
                f'<path stroke="{theme.ink}" stroke-opacity="0.6" d="M{lx + 147} {y - 8}H{lx + 154 + w:.1f}"/>'
            )
            b.append(mono("not exposed", lx + pw - 30, y, 17, theme.faint, anchor="end"))
    t_end = 0.62
    b.append(
        f'<rect class="m" x="{lx + 1}" y="{top + 2}" width="{pw - 2}" height="{row - 8}" fill="{theme.ink}" fill-opacity="0.07">'
        f'<animate attributeName="y" values="{top + 2};{top + 2};{top + 2 + row * (len(OPS) - 1)};{top + 2 + row * (len(OPS) - 1)}" '
        f'keyTimes="0;0.04;{t_end};1" dur="{loop:g}s" repeatCount="indefinite"/>'
        f'<animate attributeName="opacity" values="0;1;1;0;0" keyTimes="0;0.04;{t_end};{t_end + 0.04};1" '
        f'dur="{loop:g}s" repeatCount="indefinite"/></rect>'
    )
    # middle: the command
    mx = W / 2
    b.append(
        f'<path stroke="{theme.ink}" stroke-opacity="0.5" d="M{lx + pw + 24} {H / 2}H{rx - 24}M{rx - 36} {H / 2 - 8}l12 8l-12 8"/>'
    )
    b.append(mono("promptise", mx, H / 2 - 50, 22, theme.ink, anchor="middle", upper=False))
    b.append(mono("mcpcast", mx, H / 2 - 20, 22, theme.ink, anchor="middle", upper=False))
    b.append(mono("--profile", mx, H / 2 + 44, 19, theme.faint, anchor="middle", upper=False))
    b.append(mono("standard", mx, H / 2 + 72, 19, theme.faint, anchor="middle", upper=False))
    # right: the generated server
    b.append(
        f'<rect x="{rx}.5" y="40.5" width="{pw}" height="{H - 81}" fill="none" stroke="{theme.ink}" stroke-opacity="0.55"/>'
    )
    b.append(mono("petstore-mcp", rx + 30, 92, 22, theme.ink))
    b.append(mono("9 tools", rx + pw - 30, 92, 22, theme.faint, anchor="end"))
    b.append(f'<path stroke="{theme.line}" d="M{rx} 118.5H{rx + pw}"/>')
    j = 0
    for i, (_, _, tool, gated) in enumerate(OPS):
        if tool is None:
            continue
        y = top + 30 + j * row
        b.append(mono(tool, rx + 30, y, 23, theme.ink, upper=False))
        cx = rx + pw - 38
        if gated:
            b.append(f'<path d="M{cx} {y - 18}l10 10l-10 10l-10 -10z" fill="{theme.ink}"/>')
            b.append(mono("approval", cx - 22, y, 17, theme.faint, anchor="end"))
        else:
            b.append(mono("read", rx + pw - 30, y, 17, theme.faint, anchor="end"))
        at = 0.04 + (t_end - 0.04) * i / (len(OPS) - 1)
        b.append(
            f'<rect class="m" x="{rx + 1}" y="{y - 34}" width="{pw - 2}" height="{row - 8}" fill="{theme.ink}" opacity="0">'
            f'<animate attributeName="opacity" values="0;0;0.09;0;0" keyTimes="0;{at:.3f};{at + 0.02:.3f};{at + 0.12:.3f};1" '
            f'dur="{loop:g}s" repeatCount="indefinite"/></rect>'
        )
        j += 1
    y = top + 30 + j * row + 30
    b.append(mono("4 behind approval · 2 destructive", rx + 30, y, 17, theme.faint))
    b.append(mono("operations left out by the profile", rx + 30, y + 28, 17, theme.faint))
    return svg(W, H, "".join(b), "MCPcast turns an OpenAPI spec into a reviewed MCP server")


def main() -> None:
    for theme in (LIGHT, DARK):
        save("hero", theme, hero(theme))
        save("stack", theme, stack(theme))
        save("flow", theme, flow(theme))
        save("mcpcast", theme, mcpcast_art(theme))


if __name__ == "__main__":
    main()
