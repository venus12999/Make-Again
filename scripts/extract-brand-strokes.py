#!/usr/bin/env python3
"""Generate skeletal centerline strokes for the Make Again brand word.

Instead of glyph outlines, each letter is stored as open pen strokes (the
writing path, in font units, y-up). The renderer thickens them in real time
with round caps/joins, so the visual weight is a single tunable parameter.

Metrics (advances, cap/x height, descender) follow Quicksand Light so the
word keeps its previous layout. The shared timeline (units per ms, letter
stagger) feeds both the native canvas animation and the HTML preview SVG.
"""
import json
import math
import sys

STROKE_SPEED = 4.2      # font units per millisecond
LETTER_STAGGER = 110    # ms between letters
MIN_STROKE_MS = 120
WEIGHT = 84             # rendering stroke width in units (0.084 em)
UPEM = 1000
ASCENT = 1000
DESCENT = -250

OUT = sys.argv[1] if len(sys.argv) > 1 else 'pages/login/brand-strokes.js'


def chain(*segments):
    """Concatenate strokes into one continuous pen path (skipping join duplicates)."""
    out = []
    for seg in segments:
        if out and abs(out[-2] - seg[0]) < 0.6 and abs(out[-1] - seg[1]) < 0.6:
            out.extend(seg[2:])
        else:
            out.extend(seg)
    return out


def line(x0, y0, x1, y1):
    return [x0, y0, x1, y1]


def quad(p0, c, p1, steps=28):
    pts = []
    for i in range(steps + 1):
        t = i / steps
        mt = 1 - t
        pts += [round(mt * mt * p0[0] + 2 * mt * t * c[0] + t * t * p1[0], 1),
                round(mt * mt * p0[1] + 2 * mt * t * c[1] + t * t * p1[1], 1)]
    return pts


def arc(cx, cy, rx, ry, a0, a1, steps=48):
    """Angles in degrees, y-up, counter-clockwise from a0 to a1."""
    pts = []
    for i in range(steps + 1):
        a = math.radians(a0 + (a1 - a0) * i / steps)
        pts += [round(cx + rx * math.cos(a), 1), round(cy + ry * math.sin(a), 1)]
    return pts


def circle(cx, cy, r, steps=24):
    return arc(cx, cy, r, r, 0, 360, steps)


# Each letter: strokes in handwriting order, coordinates relative to the
# glyph origin. Strokes that touch are chained into one continuous pen path so
# the renderer's round joins close every joint; bowl arcs follow the Quicksand
# centerline ellipses (outlines inset by half the stroke weight).
GLYPHS = {
    'M': lambda: [
        line(155, 0, 155, 700),                          # left stem up
        line(155, 700, 415, 60),                         # diagonal down to middle
        line(415, 60, 674, 700),                         # diagonal up to right
        line(674, 700, 674, 0),                          # right stem down
    ],
    'a': lambda: [
        # one continuous stroke: down the stem, sweep into the bowl, around and up
        chain(
            line(488, 505, 488, 25),
            quad((488, 25), (450, 30), (417.7, 124.9)),
            arc(284, 252, 172, 202, -39, -315, 64),
        ),
    ],
    'k': lambda: [
        line(140, 0, 140, 740),                          # stem up
        quad((140, 270), (310, 400), (480, 515)),        # arm, joins the stem
        quad((140, 270), (300, 150), (480, -10)),        # leg, joins the stem
    ],
    'e': lambda: [
        # one continuous stroke: bar, then the loop curling back underneath
        chain(
            line(292, 252, 452, 252),
            arc(262, 252, 190, 202, 0, 250, 64),
        ),
    ],
    'A': lambda: [
        line(80, 0, 325, 700),                           # left diagonal up
        line(325, 700, 570, 0),                          # right diagonal down
        line(150, 240, 500, 240),                        # crossbar, ends inside both diagonals
    ],
    'g': lambda: [
        # bowl with stem (like 'a'), then a separate hook off the stem bottom
        chain(
            line(560, 505, 560, 25),
            quad((560, 25), (500, 15), (446.4, 125.4)),
            arc(300, 252, 188, 202, -39, -315, 64),
        ),
        chain(
            quad((560, 25), (605, -80), (510, -175)),
            quad((510, -175), (455, -205), (365, -190)),
        ),
    ],
    'i': lambda: [
        line(103, 0, 103, 515),                          # stem up
        circle(103, 612, 44),                            # dot
    ],
    'n': lambda: [
        # one continuous stroke: up the stem and over the arch
        chain(
            line(150, 0, 150, 515),
            quad((150, 515), (430, 530), (510, 0)),
        ),
    ],
}

ADVANCES = {'M': 829, 'a': 599, 'k': 524, 'e': 556, 'A': 649, 'g': 619, 'i': 206, 'n': 572, ' ': 270}
WORD = 'Make Again'


def stroke_length(points):
    total = 0.0
    for i in range(0, len(points) - 2, 2):
        total += ((points[i + 2] - points[i]) ** 2 + (points[i + 3] - points[i + 1]) ** 2) ** 0.5
    return total


def main():
    letters = []
    x_cursor = 0
    for index, ch in enumerate(WORD):
        advance = ADVANCES[ch]
        letter_start = index * LETTER_STAGGER
        cursor = letter_start
        contours = []
        for points in GLYPHS.get(ch, lambda: [])():
            dur = max(MIN_STROKE_MS, stroke_length(points) / STROKE_SPEED)
            contours.append({'points': points, 'start': round(cursor), 'dur': round(dur)})
            cursor += dur
        letters.append({
            'char': ch,
            'x': x_cursor,
            'advance': advance,
            'contours': contours,
            'fillAt': round(cursor),
        })
        x_cursor += advance

    total = max(l['fillAt'] for l in letters) + 80
    data = {
        'upem': UPEM,
        'ascent': ASCENT,
        'descent': DESCENT,
        'advance': x_cursor,
        'weight': WEIGHT,
        'word': WORD,
        'letterStagger': LETTER_STAGGER,
        'fillFade': 0,
        'total': round(total),
        'letters': letters,
    }
    js = ('// Generated by scripts/extract-brand-strokes.py. Skeletal centerline strokes\n'
          '// (handwriting paths, font units, y-up) for "Make Again"; the renderer thickens\n'
          '// them in real time with round caps at `weight` units. Timeline in ms is shared\n'
          '// by pages/login/login.js (canvas) and scripts/build-html-preview.cjs (SVG).\n'
          'module.exports = ' + json.dumps(data, separators=(',', ':')) + '\n')
    with open(OUT, 'w') as fh:
        fh.write(js)
    count = sum(len(l['contours']) for l in letters)
    print('wrote', OUT, f'{len(js)} bytes, {count} strokes, total {round(total)} ms')


if __name__ == '__main__':
    main()
