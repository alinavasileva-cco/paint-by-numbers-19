#!/usr/bin/env python3
"""Build both printable plates from V9 and the hand-drawn face master."""

import argparse
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path

import cairosvg
import cv2
import fitz
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as PDFTrueType
from reportlab.pdfgen import canvas
from scipy.spatial import cKDTree

from face_transfer import PATCH, REFERENCE_SHA, transfer_face
from geometry import components, contacts, open_narrow_fields, region_rings, svg_paths, vectorize
from refine_art import EDIT_BOUNDS, ink_clip_path, ink_strokes, refine_artwork

HERE = Path(__file__).resolve().parent
V9 = HERE.parent / "face_v9_color_only"
PAGE = 400, 600
WIDTH_MM = 388
STROKE_MM = 0.2
FONT_PT = 5.3
FONT_MM = FONT_PT * 25.4 / 72
FONT_PATH = Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")
FONT_FAMILY = "Liberation Sans"
SAMPLES = 2
PAINT_NAMES = (
    "Глубокий синий", "Графитово-синий", "Сумеречный индиго",
    "Чернильный", "Тёмный сливовый", "Ежевичный", "Дымчато-сливовый",
    "Тёмный винный", "Бордовый", "Гранатовый", "Пыльно-бордовый",
    "Кирпично-красный", "Приглушённая терракота", "Красно-оранжевый",
    "Медно-персиковый", "Апельсиновый", "Янтарный", "Тёплый золотистый", "Светло-кремовый",
)
FAMILIES = (
    ("Синие", 1, 3), ("Чернильные и сливовые", 4, 7),
    ("Бордовые и терракотовые", 8, 15), ("Янтарные и золотистые", 16, 19),
)


def palette():
    with (HERE.parent / "review/palette.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = [(int(row[0]), row[1], tuple(map(int, row[2:])))
                for row in list(csv.reader(file))[1:]]
    if len(rows) != 19 or set(row[0] for row in rows) != set(range(1, 20)):
        raise ValueError("The V9 palette must contain all 19 original paint numbers")
    if len({row[1] for row in rows}) != 19:
        raise ValueError("The V9 palette contains repeated paints")
    return sorted(rows)


def artistic_master(rows):
    """Composite the local curves over the exact indexed companion to the V9 preview."""
    source = V9 / "portrait_v9_color.png"
    base = Image.open(source).convert("RGB")
    size = tuple(dimension * SAMPLES for dimension in base.size)
    rgb = np.array(base.resize(size, Image.Resampling.NEAREST))
    layer_png = cairosvg.svg2png(url=str(HERE / "face_master.svg"),
                               output_width=size[0], output_height=size[1])
    layer = np.asarray(Image.open(BytesIO(layer_png)).convert("RGBA"))
    mask = layer[:, :, 3] >= 128
    rgb[mask] = layer[mask, :3]

    pigments, inverse = np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True)
    tree = cKDTree(np.asarray([row[2] for row in rows]))
    _, nearest = tree.query(pigments)
    indexed = (nearest[inverse].reshape(rgb.shape[:2]) + 1).astype(np.uint8)
    return indexed, {
        "v9_indexed_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "v9_preview_sha256": hashlib.sha256((V9 / "portrait_v9_smooth_preview.png").read_bytes()).hexdigest(),
        "face_layer_bbox_native_px": [round(float(limit) / SAMPLES, 2) for limit in
                                      Image.fromarray(np.uint8(mask) * 255).getbbox()],
        "outside_face_master_changed_pixels": int(np.count_nonzero(
            (~mask) & np.any(rgb != np.asarray(base.resize(size, Image.Resampling.NEAREST)), axis=2))),
    }


def smooth_colors(indexed, ncolors):
    """Remove pixel-size serrations while retaining the hand-drawn facial planes."""
    winner = indexed.copy()
    best = np.full(indexed.shape, -1, dtype=np.float32)
    y = np.arange(indexed.shape[0])[:, None] / SAMPLES
    x = np.arange(indexed.shape[1])[None, :] / SAMPLES
    eye_and_profile = (445 < x) & (x < 693) & (325 < y) & (y < 550)
    for paint in range(1, ncolors + 1):
        mask = np.float32(indexed == paint)
        smooth = cv2.GaussianBlur(mask, (0, 0), 1.12 * SAMPLES)
        face = cv2.GaussianBlur(mask, (0, 0), 0.70 * SAMPLES)
        smooth[eye_and_profile] = face[eye_and_profile]
        better = smooth > best
        winner[better] = paint
        best[better] = smooth[better]
    return winner


def glyph_bounds():
    """Use the visible outline of each number in the same font the PDF embeds."""
    font_file = FONT_PATH
    if not font_file.exists():
        font_file = font_file.parent.parent / "liberation2" / font_file.name
    font = TTFont(font_file)
    units = font["head"].unitsPerEm
    cmap = font.getBestCmap()
    glyphs = font["glyf"]
    advances = font["hmtx"].metrics
    bounds = {}
    for paint in range(1, 20):
        left, bottom, right, top = float("inf"), float("inf"), 0, 0
        pen = 0
        for char in str(paint):
            name = cmap[ord(char)]
            glyph = glyphs[name]
            left, bottom = min(left, pen + glyph.xMin), min(bottom, glyph.yMin)
            right, top = max(right, pen + glyph.xMax), max(top, glyph.yMax)
            pen += advances[name][0]
        bounds[paint] = tuple(value * FONT_MM / units for value in (left, bottom, right, top))
    return bounds, font_file


def locate_numbers(ids, colors, boxes, unit, bounds, detail_spacing=False):
    geometry = []
    unfitted = []
    for region, (x, y, w, h, _) in enumerate(boxes[1:], start=1):
        paint = int(colors[region])
        left, bottom, right, top = bounds[paint]
        near_face = 445 < (x + w / 2) / SAMPLES < 693 and 325 < (y + h / 2) / SAMPLES < 550
        margin_mm = 0.33 if near_face else 0.44
        center_x, center_y = (x + w / 2) / SAMPLES, (y + h / 2) / SAMPLES
        eye_detail = 545 < center_x < 664 and 363 < center_y < 416 and w < 30 and h < 40
        if detail_spacing and eye_detail:
            margin_mm = 0.20
        half_width = int(np.ceil(((right - left) / 2 + margin_mm) / unit))
        half_height = int(np.ceil(((top - bottom) / 2 + margin_mm) / unit))
        if w < 2 * half_width + 1 or h < 2 * half_height + 1:
            unfitted.append(region)
            continue
        member = np.uint8(ids[y:y+h, x:x+w] == region)
        sample_space = cv2.erode(
            member, np.ones((2 * half_height + 1, 2 * half_width + 1), np.uint8),
            borderType=cv2.BORDER_CONSTANT, borderValue=0,
        )
        if not sample_space.any():
            unfitted.append(region)
            continue
        distance = cv2.distanceTransform(np.pad(member, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        clearance = np.rint(distance ** 2)
        clearance[sample_space == 0] = 0
        row, column = np.unravel_index(clearance.argmax(), clearance.shape)
        geometry.append((region, paint, float(x + column + 0.5), float(y + row + 0.5)))
    return geometry, unfitted


def fit_regions(indexed, rows, unit, bounds, detail_spacing=False):
    """Merge a field only when its printed number cannot fit."""
    rgb = np.float32([[[0, 0, 0]] + [list(row[2]) for row in rows]]) / 255
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[0]
    for iteration in range(20):
        ids, colors, boxes = components(indexed, len(rows))
        numbered, unfitted = locate_numbers(ids, colors, boxes, unit, bounds, detail_spacing)
        print(f"Pass {iteration + 1}: {len(numbered)} numbered, {len(unfitted)} need more space",
              flush=True)
        if not unfitted:
            return indexed, ids, colors, numbered
        neighbors = contacts(ids)
        assigned = colors.copy()
        needs_space = set(unfitted)
        for region in unfitted:
            area = boxes[region][-1]
            pigment = lab[colors[region]]
            options = [(other, shared) for other, shared in neighbors[region]
                       if other not in needs_space or (boxes[other][-1], other) > (area, region)]
            if not options:
                continue

            def score(contact):
                other, shared = contact
                difference = np.linalg.norm(pigment - lab[colors[other]])
                return np.log1p(shared) - .051 * difference + (1.20 if other not in needs_space else 0)

            target, _ = max(options, key=score)
            assigned[region] = colors[target]
        changed = assigned[ids]
        if np.array_equal(changed, indexed):
            raise ValueError("Some fields cannot be numbered or merged")
        indexed = changed
    raise ValueError("The numbering geometry did not converge")


def plate_svg(title, content):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE[0]}mm" height="{PAGE[1]}mm" '
            f'viewBox="0 0 {PAGE[0]} {PAGE[1]}">\n<title>{title}</title>\n'
            f'<rect width="{PAGE[0]}" height="{PAGE[1]}" fill="white"/>\n{content}\n</svg>\n')


def sheet_artifacts(destination, rows, source_data, indexed, ids, colors, numbered, unit, bounds,
                    arcs, rings, edges):
    height, width = indexed.shape
    origin = (PAGE[0] - width * unit) / 2, (PAGE[1] - height * unit) / 2
    outline, fills = svg_paths(arcs, rings, origin, unit)
    paint_hex = [""] + [row[1] for row in rows]
    boundary = (f'<g id="boundaries" fill="none" stroke="#46434A" stroke-width="{STROKE_MM}" '
                f'stroke-linecap="round" stroke-linejoin="round"><path d="{outline}"/></g>')
    labels = []
    segments = []
    for region, paint, column, row in numbered:
        x, y = origin[0] + column * unit, origin[1] + row * unit
        left, bottom, right, top = bounds[paint]
        labels.append(f'<text x="{x - (left + right) / 2:.4f}" y="{y + (top + bottom) / 2:.4f}" '
                      f'data-region="{region}">{paint}</text>')
        segments.append({"segment_id": region, "paint_id": paint,
                         "label_x_px": column, "label_y_px": row,
                         "bounds_mm": [round(x - (right - left) / 2, 4),
                                       round(y - (top - bottom) / 2, 4),
                                       round(x + (right - left) / 2, 4),
                                       round(y + (top - bottom) / 2, 4)],
                         "label_fits": True})
    label_group = (f'<g id="paint-numbers" font-family="{FONT_FAMILY}" '
                   f'font-size="{FONT_MM:.9f}" fill="#545159">\n' + "\n".join(labels) + "\n</g>")
    def color_path(region, path):
        pigment = paint_hex[int(colors[region])]
        return f'<path data-region="{region}" fill="{pigment}" stroke="{pigment}" d="{path}"/>'

    colored = "\n".join(color_path(region, path) for region, path in fills.items())
    # A narrow same-color stroke hides antialiasing seams along the shared curves.
    colored = ('<g id="paint-fills" fill-rule="evenodd" stroke-width=".14" stroke-linejoin="round">\n'
               + colored + "\n</g>")

    clip = ink_clip_path(arcs, rings, segments, origin, unit, SAMPLES)
    clip_definition = (f'<defs><clipPath id="art-ink-clip"><path clip-rule="evenodd" '
                       f'd="{clip}"/></clipPath></defs>')

    def details(monochrome):
        art_scale = unit * SAMPLES
        paths = "".join(f'<path stroke="{("#46434A" if monochrome else pigment)}" d="{path}"/>'
                        for path, pigment in ink_strokes())
        return (clip_definition + '<g id="open-details" clip-path="url(#art-ink-clip)">'
                f'<g transform="translate({origin[0]} {origin[1]}) scale({art_scale})" fill="none" '
                f'stroke-width="{STROKE_MM / art_scale}" stroke-linecap="round">{paths}</g></g>')

    line_art = boundary + "\n" + details(True)
    (destination / "numbered-outline.svg").write_text(
        plate_svg("Ночные фонари · схема по номерам", line_art + "\n" + label_group), encoding="utf-8")
    (destination / "color-map.svg").write_text(
        plate_svg("Ночные фонари · цветной образец", colored + "\n" + details(False)), encoding="utf-8")
    (destination / "_unlabelled.svg").write_text(
        plate_svg("Контуры для проверки", line_art), encoding="utf-8")
    audit = {**source_data, "image_px": [width, height], "print_size_mm": list(PAGE),
             "art_box_mm": [*origin, width * unit, height * unit],
             "line_width_mm": STROKE_MM, "font_size_pt": FONT_PT, "colors": len(rows),
             "regions": len(numbered), "unfitted_labels": [], "vector_arcs": len(arcs),
             "shared_grid_edges": edges, "segment_connectivity": 4,
             "segments": segments}
    (destination / "segments.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    Image.fromarray(indexed).save(destination / "labels.png")
    Image.fromarray(ids.astype(np.uint16)).save(destination / "_region_ids.png")
    return audit


def export_palette(destination, rows, font_file):
    pdfmetrics.registerFont(PDFTrueType("Palette", str(font_file)))
    swatches = canvas.Canvas(str(destination / "palette.pdf"), pagesize=A4, pageCompression=1)
    swatches.setTitle("Ночные фонари — палитра из 19 цветов")
    swatches.setAuthor("THE CASE")
    swatches.setFillColor(HexColor("#201B25"))
    swatches.setFont("Palette", 31)
    swatches.drawString(44, 776, "Ночные фонари")
    swatches.setFont("Palette", 12)
    swatches.setFillColor(HexColor("#615859"))
    swatches.drawString(45, 749, "Палитра для раскраски по номерам · 19 цветов")
    families = {paint: family for family, first, last in FAMILIES for paint in range(first, last + 1)}
    with (destination / "palette.csv").open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
        writer.writerow(("paint_id", "color_hex", "name_ru", "family_ru", "source_cluster"))
        for paint, color, _ in rows:
            writer.writerow((paint, color, PAINT_NAMES[paint - 1], families[paint], paint))
    for (family, first, last), (y, column_count) in zip(
            FAMILIES, ((701, 3), (560, 4), (420, 4), (169, 4))):
        swatches.setFillColor(HexColor("#201B25"))
        swatches.setFont("Palette", 13)
        swatches.drawString(45, y, family)
        y -= 19
        slot = 510 / column_count
        for index, paint in enumerate(range(first, last + 1)):
            column, row = index % column_count, index // column_count
            x = 45 + column * slot
            level = y - row * 100
            color = rows[paint - 1][1]
            swatches.setFillColor(HexColor(color))
            swatches.roundRect(x, level - 41, slot - 13, 41, 4, stroke=0, fill=1)
            swatches.setFillColor(HexColor("#201B25"))
            swatches.setFont("Palette", 10.3)
            swatches.drawString(x, level - 55, f"{paint:02d}   {color}")
            swatches.setFillColor(HexColor("#615859"))
            swatches.setFont("Palette", 8.6)
            swatches.drawString(x, level - 69, PAINT_NAMES[paint - 1])
    swatches.save()


def export_pdf(destination):
    import subprocess
    for source, stem in (("numbered-outline", "bw-numbered"), ("color-map", "color-map")):
        subprocess.run(("inkscape", str(destination / f"{source}.svg"),
                        f"--export-filename={destination / (stem + '.pdf')}"),
                       check=True, capture_output=True)
        with fitz.open(destination / f"{stem}.pdf") as document:
            document[0].get_pixmap(matrix=fitz.Matrix(1.25, 1.25)).save(destination / f"{stem}.png")
    with fitz.open() as combined:
        for filename in ("bw-numbered.pdf", "color-map.pdf"):
            with fitz.open(destination / filename) as document:
                combined.insert_pdf(document)
        combined.set_metadata({"title": "Ночные фонари — раскраска по номерам", "author": "THE CASE"})
        combined.save(destination / "print-plates.pdf", deflate=True)


def compile_fields(indexed, rows, unit, bounds, detail_spacing=False):
    for attempt in range(8):
        indexed, ids, colors, numbered = fit_regions(indexed, rows, unit, bounds, detail_spacing)
        arcs, edge_count = vectorize(ids, SAMPLES)
        rings = region_rings(arcs, len(numbered))
        repaired, defects = open_narrow_fields(
            indexed, ids, numbered, arcs, rings, unit, outer_clearance=0.19
        )
        print(f"Vector proof {attempt + 1}: {defects} narrow splits", flush=True)
        if not defects:
            return indexed, ids, colors, numbered, arcs, rings, edge_count
        if np.array_equal(indexed, repaired):
            raise ValueError("Some vector fields are too narrow to paint")
        indexed = repaired
    raise ValueError("Could not repair all narrow vector fields")


def retain_scene_numbers(numbered, prior, ids, colors, changed, patch=PATCH):
    """Reuse existing label positions away from the selected edit."""
    x0, y0, x1, y1 = patch
    existing = {}
    for _, paint, x, y in prior:
        column, row = int(x), int(y)
        inside = ((x0 - 6)*SAMPLES <= x < (x1 + 6)*SAMPLES
                  and (y0 - 6)*SAMPLES <= y < (y1 + 6)*SAMPLES)
        if inside and changed[row - 10:row + 11, column - 10:column + 11].any():
            continue
        region = int(ids[row, column])
        if colors[region] != paint:
            raise ValueError("The face transfer displaced a label in the scene")
        existing.setdefault(region, []).append((region, paint, x, y))
    return [existing[region][0] if len(existing.get(region, ())) == 1
            else (region, paint, x, y) for region, paint, x, y in numbered]


def refine_print(destination, rows, source_data, baseline, baseline_numbers, unit, bounds):
    artwork, anatomy = refine_artwork(baseline, rows, SAMPLES)
    indexed, ids, colors, numbered, arcs, rings, edge_count = compile_fields(
        artwork, rows, unit, bounds, detail_spacing=True
    )
    changed = indexed != baseline
    x0, y0, x1, y1 = EDIT_BOUNDS
    outside = changed.copy()
    outside[y0*SAMPLES:y1*SAMPLES, x0*SAMPLES:x1*SAMPLES] = False
    if outside.any():
        raise ValueError("The drawing changed outside the portrait and kimono")
    numbered = retain_scene_numbers(numbered, baseline_numbers, ids, colors, changed, EDIT_BOUNDS)
    source_data.update({
        "vector_topology_checked_mm": .19,
        "approved_face_v10_commit": REFERENCE_SHA,
        "art_refinement_bbox_native_px": list(EDIT_BOUNDS),
        "paint_pixels_changed_inside_art": int(np.count_nonzero(changed)),
        "paint_pixels_changed_outside_art": int(np.count_nonzero(outside)),
        **anatomy,
    })
    sheet_artifacts(destination, rows, source_data, indexed, ids, colors, numbered,
                    unit, bounds, arcs, rings, edge_count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows = palette()
    master, source_data = artistic_master(rows)
    unit = WIDTH_MM / master.shape[1]
    bounds, font_file = glyph_bounds()

    initial, _, _, prior, *_ = compile_fields(smooth_colors(master, len(rows)), rows, unit, bounds)
    face = transfer_face(initial, SAMPLES)
    baseline, base_ids, base_colors, before, *_ = compile_fields(face, rows, unit, bounds, detail_spacing=True)
    baseline_numbers = retain_scene_numbers(before, prior, base_ids, base_colors, baseline != initial)
    refine_print(destination, rows, source_data, baseline, baseline_numbers, unit, bounds)
    export_palette(destination, rows, font_file)
    export_pdf(destination)


if __name__ == "__main__":
    main()
