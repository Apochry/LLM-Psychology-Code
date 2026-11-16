"""Inline-SVG helpers themed by the blog's .viz CSS classes (see src/styles/global.css)."""
from html import escape
FONT = 'font-family="Inter Variable, Inter, system-ui, sans-serif"'


class Plot:
    def __init__(self, x0, y0, w, h, xr, yr):
        self.x0, self.y0, self.w, self.h, self.xr, self.yr = x0, y0, w, h, xr, yr
    def X(self, v): return self.x0 + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w
    def Y(self, v): return self.y0 + self.h - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h


def svg(w, h, body, label):
    return (f'<svg class="viz" viewBox="0 0 {w} {h}" role="img" aria-label="{escape(label)}" {FONT} xmlns="http://www.w3.org/2000/svg">'
            f'<rect class="bg" width="{w}" height="{h}"/>' + "".join(body) + "</svg>")


def yaxis(p, ticks, fmt=lambda t: f"{t:g}", title=None):
    b = []
    for t in ticks:
        y = p.Y(t)
        b.append(f'<line class="grid" x1="{p.x0}" x2="{p.x0 + p.w}" y1="{y:.1f}" y2="{y:.1f}"/>')
        b.append(f'<text class="tick" x="{p.x0 - 6}" y="{y + 4:.1f}" text-anchor="end">{fmt(t)}</text>')
    if title:
        b.append(f'<text class="ptitle" x="{p.x0}" y="{p.y0 - 16}">{escape(title)}</text>')
    return b


def xaxis(p, ticks, fmt=lambda t: f"{t:g}", title=None, labels=None):
    b = [f'<line class="axis" x1="{p.x0}" x2="{p.x0 + p.w}" y1="{p.y0 + p.h}" y2="{p.y0 + p.h}"/>']
    for i, t in enumerate(ticks):
        b.append(f'<text class="tick" x="{p.X(t):.1f}" y="{p.y0 + p.h + 18}" text-anchor="middle">{escape(labels[i]) if labels else fmt(t)}</text>')
    if title:
        b.append(f'<text class="tick" x="{p.x0 + p.w / 2}" y="{p.y0 + p.h + 38}" text-anchor="middle">{escape(title)}</text>')
    return b


def dot(x, y, cls, tip, r=4.5):
    return f'<circle class="{cls} dot" cx="{x:.1f}" cy="{y:.1f}" r="{r}"><title>{escape(tip)}</title></circle>'


def line(pts, cls):
    return f'<polyline class="{cls} ln" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>'


def series(p, xs, ys, cls, name, fmt="{:.2f}", r=4):
    b = [line([(p.X(x), p.Y(y)) for x, y in zip(xs, ys)], cls)]
    b += [dot(p.X(x), p.Y(y), cls, f"{name}, {x:g}: " + fmt.format(y), r) for x, y in zip(xs, ys)]
    return b


def bar(p, xc, w, v, cls, tip, base=0.0, label=None):
    y0, y1 = p.Y(max(v, base)), p.Y(min(v, base))
    h = max(1.0, y1 - y0); r = min(4, h / 2)
    x = xc - w / 2
    d = f"M{x:.1f},{y1:.1f} V{y0 + r:.1f} Q{x:.1f},{y0:.1f} {x + r:.1f},{y0:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y0:.1f} {x + w:.1f},{y0 + r:.1f} V{y1:.1f} Z"
    b = [f'<path class="{cls} bar" d="{d}"><title>{escape(tip)}</title></path>']
    if label is not None:
        b.append(f'<text class="val" x="{xc:.1f}" y="{y0 - 6:.1f}" text-anchor="middle">{escape(label)}</text>')
    return b


def band(p, xs, lo, hi):
    pts = [(p.X(x), p.Y(v)) for x, v in zip(xs, hi)] + [(p.X(x), p.Y(v)) for x, v in zip(reversed(xs), reversed(lo))]
    return [f'<polygon class="ctrl-band" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>']


def legend(x, y, items):
    b, cx = [], x
    for cls, kind, text in items:
        if kind == "line":
            b.append(f'<line class="{cls} ln" x1="{cx}" x2="{cx + 18}" y1="{y - 4}" y2="{y - 4}"/><circle class="{cls} dot" cx="{cx + 9}" cy="{y - 4}" r="3.5"/>')
        elif kind == "bar":
            b.append(f'<rect class="{cls} bar" x="{cx + 3}" y="{y - 10}" width="12" height="12" rx="2"/>')
        elif kind == "dot":
            b.append(f'<circle class="{cls} dot" cx="{cx + 9}" cy="{y - 4}" r="4.5"/>')
        elif kind == "band":
            b.append(f'<rect class="ctrl-band" x="{cx + 1}" y="{y - 11}" width="16" height="14"/>')
        b.append(f'<text class="leg" x="{cx + 24}" y="{y}">{escape(text)}</text>')
        cx += 36 + 8.0 * len(text)
    return b


def text(x, y, s, cls="tick", anchor="start"):
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{escape(s)}</text>'
