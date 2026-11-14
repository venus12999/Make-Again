#!/usr/bin/env python3
"""Extract Caveat glyph outlines for the Make Again brand word.

Each glyph's contours are flattened to polylines (font units, y-up) and laid
out at their pen origins, including the negative overhangs of cursive joins.
A shared stroke-drawing timeline (units per ms) feeds the native canvas
animation and the HTML preview SVG so both draw the real letterforms
stroke by stroke, then fade the fill in.
"""
import json
import sys

from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import DecomposingRecordingPen

WORD = 'Make Again'
STROKE_SPEED = 5.2      # font units per millisecond
LETTER_STAGGER = 130    # ms between letters
FILL_GAP = 60           # ms between last outline and fill fade
FILL_FADE = 320         # ms fill fade duration
MIN_STROKE_MS = 110
CURVE_STEPS = 12

SRC = sys.argv[1] if len(sys.argv) > 1 else '/tmp/Caveat-Medium.ttf'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'pages/login/brand-strokes.js'


class FlatPen:
    """Outline pen that flattens curves into polylines."""

    def __init__(self):
        self.contours = []
        self.current = None
        self.start = None

    def _add(self, x, y):
        self.current.extend((round(x, 1), round(y, 1)))

    def moveTo(self, p):
        self.current = []
        self.start = p
        self._add(*p)
        self.contours.append(self.current)

    def lineTo(self, p):
        self._add(*p)

    def curveTo(self, *points):
        p0 = (self.current[-2], self.current[-1])
        p1, p2, p3 = points
        for i in range(1, CURVE_STEPS + 1):
            t = i / CURVE_STEPS
            mt = 1 - t
            x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
            y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
            self._add(x, y)

    def qCurveTo(self, *points):
        c = (self.current[-2], self.current[-1])
        pts = [p for p in points if p is not None]
        if not pts:
            return
        end = pts[-1]
        controls = pts[:-1]
        anchors = [c]
        for j in range(len(controls) - 1):
            a, b = controls[j], controls[j + 1]
            anchors.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
        anchors.append(end)
        for j, ctrl in enumerate(controls):
            self._quad(anchors[j], ctrl, anchors[j + 1])
        if not controls:
            self.lineTo(end)

    def _quad(self, p0, p1, p2):
        for i in range(1, CURVE_STEPS + 1):
            t = i / CURVE_STEPS
            mt = 1 - t
            x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0]
            y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]
            self._add(x, y)

    def closePath(self):
        if self.start:
            self._add(*self.start)
        self.current = None

    def endPath(self):
        self.current = None

    def addComponent(self, glyphName, transformation):
        raise AssertionError('DecomposingRecordingPen should have expanded components')


def contour_length(points):
    total = 0.0
    for i in range(0, len(points) - 2, 2):
        total += ((points[i + 2] - points[i]) ** 2 + (points[i + 3] - points[i + 1]) ** 2) ** 0.5
    return total


def main():
    font = TTFont(SRC)
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    hhea = font['head']
    upem = font['head'].unitsPerEm

    letters = []
    x_cursor = 0.0
    min_x = 0.0
    max_x = 0.0
    raw = []
    for ch in WORD:
        glyph_name = cmap[ord(ch)]
        recording = DecomposingRecordingPen(glyph_set)
        glyph_set[glyph_name].draw(recording)
        pen = FlatPen()
        recording.replay(pen)
        advance = font['hmtx'][glyph_name][0]
        raw.append((ch, x_cursor, advance, pen.contours))
        for contour in pen.contours:
            xs = contour[0::2]
            if xs:
                min_x = min(min_x, x_cursor + min(xs))
                max_x = max(max_x, x_cursor + max(xs))
        x_cursor += advance

    x_shift = -min_x
    total_advance = x_cursor
    ink_right = max_x + x_shift
    width = max(total_advance, ink_right) + 20

    for index, (ch, x, advance, contours) in enumerate(raw):
        letter_start = index * LETTER_STAGGER
        cursor = letter_start
        timed = []
        for points in contours:
            length = contour_length(points)
            dur = max(MIN_STROKE_MS, length / STROKE_SPEED)
            timed.append({'points': points, 'start': round(cursor), 'dur': round(dur)})
            cursor += dur
        letters.append({
            'char': ch,
            'x': round(x + x_shift, 1),
            'advance': advance,
            'contours': timed,
            'fillAt': round(cursor + FILL_GAP),
        })

    total = max(l['fillAt'] for l in letters) + FILL_FADE
    data = {
        'upem': upem,
        'ascent': font['hhea'].ascent,
        'descent': font['hhea'].descent,
        'advance': round(width, 1),
        'word': WORD,
        'letterStagger': LETTER_STAGGER,
        'fillFade': FILL_FADE,
        'total': round(total),
        'letters': letters,
    }
    js = ('// Generated by scripts/extract-caveat-outlines.py from Caveat Medium (SIL OFL 1.1).\n'
          '// Real glyph outlines as flattened polylines in font units (y-up) plus a shared\n'
          '// stroke-drawing timeline in milliseconds. Consumed by pages/login/login.js\n'
          '// (canvas) and scripts/build-html-preview.cjs (SVG preview).\n'
          'module.exports = ' + json.dumps(data, separators=(',', ':')) + '\n')
    with open(OUT, 'w') as fh:
        fh.write(js)
    count = sum(len(l['contours']) for l in letters)
    print('wrote', OUT, f'{len(js)} bytes, {count} contours, total {round(total)} ms, width {round(width)} units')


if __name__ == '__main__':
    main()
