#!/usr/bin/env python3
"""Build Night Lanterns from the approved indexed illustration."""

import argparse
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import sys
from xml.etree import ElementTree as ET

import cairosvg
import cv2
import numpy as np
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "night_lanterns_v11"))

# Import the existing region fitting and tracing code. The artwork comes only
# from the approved first-stage master.
from night_lanterns_v11.build import FONT_FAMILY, FONT_MM, FONT_PT, PAGE, STROKE_MM, fit_regions, glyph_bounds
from geometry import open_narrow_fields, region_rings, svg_paths, vectorize

SOURCE_HASH = "159d218a4f53a5e6f7676e0c621681bed8e55b06bb7d14c3de2eeca480810c72"
SAMPLES = 2
PROOF_PX_PER_MM = 12
PORTRAIT = (194, 196, 707, 641)
FACE = (413, 394, 665, 601)
SILK = (137, 608, 739, 1144)
LANTERNS = ((0, 672, 165, 889), (698, 574, 853, 802), (711, 1027, 1003, 1403))
STEMS = {
    "scheme": "Nochnye_fonari_40x60_shema",
    "color": "Nochnye_fonari_40x60_cvet",
    "palette": "Nochnye_fonari_palitra_19",
}
# Put the small highlights in the one-digit part of the palette. Group each
# family together and keep the approved channel values verbatim.
SOURCE_ORDER = (14, 15, 16, 17, 18, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
BACKGROUND_PAINT = SOURCE_ORDER.index(1) + 1
PAINTS = (
    ("Медный", "Тёплые огни"),
    ("Огненно-оранжевый", "Тёплые огни"),
    ("Янтарный", "Тёплые огни"),
    ("Золотистый", "Тёплые огни"),
    ("Кремовый", "Тёплые огни"),
    ("Мягкая терракота", "Терракота и персик"),
    ("Персиковый", "Терракота и персик"),
    ("Светлый персик", "Терракота и персик"),
    ("Чернильный", "Чернила и индиго"),
    ("Ночной синий", "Чернила и индиго"),
    ("Глубокий индиго", "Чернила и индиго"),
    ("Сумеречный синий", "Чернила и индиго"),
    ("Дымчатый индиго", "Чернила и индиго"),
    ("Тёмный сливовый", "Слива и бордо"),
    ("Сливовый", "Слива и бордо"),
    ("Тёмный винный", "Слива и бордо"),
    ("Бордовый", "Слива и бордо"),
    ("Карминовый", "Слива и бордо"),
    ("Пыльно-розовый", "Слива и бордо"),
)


def read_master(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_HASH:
        raise ValueError("Use the approved indexed PNG from stage 1")
    with Image.open(path) as source:
        if source.mode != "P" or source.size != (4096, 6144):
            raise ValueError("The indexed master has an unexpected format")
        colors = np.uint8(source.getpalette()[: 19 * 3]).reshape(19, 3)
        if len({tuple(rgb) for rgb in colors}) != 19 or np.any(np.all(colors == 255, axis=1)):
            raise ValueError("The approved master needs 19 distinct pigments")
        lookup = np.empty(19, dtype=np.uint8)
        lookup[list(SOURCE_ORDER)] = np.arange(1, 20, dtype=np.uint8)
        pixels = lookup[np.asarray(source)[1::2, 1::2]]
    if set(np.unique(pixels)) != set(range(1, 20)):
        raise ValueError("A master swatch is missing from the artwork")
    rows = [
        (number, "#" + bytes(rgb).hex().upper(), tuple(int(value) for value in rgb))
        for number, rgb in enumerate(colors[list(SOURCE_ORDER)], 1)
    ]
    return pixels, rows


def within(x, y, box):
    x0, y0, x1, y1 = box
    return x0 <= x < x1 and y0 <= y < y1


def transfer_details(master, rows):
    """Widen the original highlights and flower centers for printed digits."""
    palette = {color: paint for paint, color, _ in rows}
    indexed = master.copy()
    design = ET.parse(HERE / "detail-fields.svg").getroot()
    for patch in design:
        layer = ET.Element(design.tag, design.attrib)
        layer.append(patch)
        png = cairosvg.svg2png(bytestring=ET.tostring(layer), output_width=master.shape[1],
                              output_height=master.shape[0])
        with Image.open(BytesIO(png)) as image:
            mask = np.asarray(image)[..., 3] >= 128
        indexed[mask] = palette[patch.attrib["fill"]]
    return indexed


def placement(x, y, _paint):
    """Keep facial and floral detail finer than distant reflections."""
    x, y = x / SAMPLES, y / SAMPLES
    if within(x, y, FACE):
        return 0.22, 0.0
    if within(x, y, PORTRAIT) or within(x, y, SILK) or any(within(x, y, box) for box in LANTERNS):
        return 0.30, 4.4
    return 0.32, 9.0


def contour_layer(outline):
    return (
        f'<g id="boundaries" fill="none" stroke="#323232" stroke-width="{STROKE_MM}" '
        f'stroke-linecap="round" stroke-linejoin="round"><path d="{outline}"/></g>'
    )


def scan_paper(svg):
    """Locate separate white interiors in the print-size stroke proof."""
    scale = PROOF_PX_PER_MM
    preview = cairosvg.svg2png(bytestring=svg.encode(), output_width=PAGE[0] * scale,
                              output_height=PAGE[1] * scale)
    with Image.open(BytesIO(preview)) as image:
        white = np.uint8(np.asarray(image.convert("L")) >= 190)
    return cv2.connectedComponentsWithStats(white, connectivity=8)


def close_print_pinholes(indexed, numbers, arcs, rings, unit):
    outline, _ = svg_paths(arcs, rings, (0, 0), unit)
    proof = make_svg("Контуры для проверки", contour_layer(outline), "#FFFFFF")
    count, interiors, stats, centers = scan_paper(proof)
    scale = PROOF_PX_PER_MM
    numbered = {int(interiors[int(y * unit * scale), int(x * unit * scale)]) for _, _, x, y in numbers}
    if 0 in numbered or len(numbered) != len(numbers):
        raise ValueError("The printed outlines displaced or combined labelled fields")
    pinholes = set(range(1, count)) - numbered
    edits = indexed.copy()
    for cell in sorted(pinholes):
        x, y, width, height, _ = stats[cell]
        if max(width, height) / scale > 1:
            raise ValueError("An unlabelled printed field needs an explicit repair")
        column, row = (int((value + .5) / (scale * unit)) for value in centers[cell])
        initial_radius = int(np.ceil((max(width, height) / (2 * scale) + .34) / unit))
        # The rounded contour may be displaced from the nearest raster edge.
        # Find that edge before adjusting the small junction.
        for radius in range(initial_radius, initial_radius + 5):
            y0, y1 = max(0, row - radius), min(row + radius + 1, indexed.shape[0])
            x0, x1 = max(0, column - radius), min(column + radius + 1, indexed.shape[1])
            patch = edits[y0:y1, x0:x1]
            mask = np.zeros(patch.shape, np.uint8)
            cv2.circle(mask, (column - x0, row - y0), radius, 1, thickness=-1)
            neighborhood = patch[mask > 0]
            if neighborhood.min() != neighborhood.max():
                patch[mask > 0] = int(np.bincount(neighborhood).argmax())
                break
        else:
            raise ValueError(f"Cannot find the associated contour at ({column}, {row})")
    return edits, len(pinholes)


def compile_fields(indexed, rows, bounds):
    unit = PAGE[0] / indexed.shape[1]
    for attempt in range(9):
        indexed, ids, colors, numbers = fit_regions(
            indexed, rows, unit, bounds, placement=placement, color_distance_weight=0.081,
        )
        arcs, source_edges = vectorize(
            ids, SAMPLES, detail_box=PORTRAIT, smoothing_scale=0.72, preserve_frame=True
        )
        rings = region_rings(arcs, len(numbers))
        for clearance in (STROKE_MM / 2, 0.15):
            repaired, defects = open_narrow_fields(
                indexed, ids, numbers, arcs, rings, unit, outer_clearance=clearance
            )
            if defects:
                break
        print(f"Vector check {attempt + 1}: {defects} split pockets", flush=True)
        if not defects:
            repaired, pinholes = close_print_pinholes(indexed, numbers, arcs, rings, unit)
            print(f"Print check: {pinholes} pinholes at contour junctions", flush=True)
            if not pinholes:
                return indexed, ids, colors, numbers, arcs, rings, source_edges
        if np.array_equal(indexed, repaired):
            raise ValueError("A contour still splits a field at print size")
        indexed = repaired
    raise ValueError("The printed field geometry did not converge")


def make_svg(title, content, background):
    width, height = PAGE
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<title>{title}</title>\n<rect width="{width}" height="{height}" fill="{background}"/>\n'
        f'{content}\n</svg>\n'
    )


def export_vectors(target, source, compiled, master, rows, bounds):
    indexed, ids, colors, numbers, arcs, rings, source_edges = compiled
    if set(np.unique(indexed)) != set(range(1, 20)):
        raise ValueError("The fitted map lost one or more approved pigments")
    width, height = indexed.shape[::-1]
    unit = PAGE[0] / width
    outline, fills = svg_paths(arcs, rings, (0, 0), unit)
    palette = {number: color for number, color, _ in rows}

    marks = []
    manifest = []
    for region, paint, column, row in numbers:
        x, y = column * unit, row * unit
        left, bottom, right, top = bounds[paint]
        rw, rh = (right - left) / 2, (top - bottom) / 2
        baseline = x - (left + right) / 2, y + (top + bottom) / 2
        marks.append(
            f'<text data-region="{region}" x="{baseline[0]:.5f}" y="{baseline[1]:.5f}">{paint}</text>'
        )
        manifest.append({
            "segment_id": region,
            "paint_id": paint,
            "label_x_px": column,
            "label_y_px": row,
            "bounds_mm": [round(k, 5) for k in (x - rw, y - rh, x + rw, y + rh)],
            "label_fits": True,
        })

    contours = contour_layer(outline)
    numbers_group = (
        f'<g id="paint-numbers" font-family="{FONT_FAMILY}" font-size="{FONT_MM:.9f}" fill="#242424">\n'
        + "\n".join(marks) + "\n</g>"
    )
    colored = '<g id="paint-fills" fill-rule="evenodd">\n' + "\n".join(
        f'<path data-region="{region}" fill="{palette[int(colors[region])]}" d="{path}"/>'
        for region, path in fills.items()
    ) + "\n</g>"
    svg_files = {
        "numbered-outline.svg": make_svg("Ночные фонари · схема по номерам", contours + "\n" + numbers_group, "#FFFFFF"),
        "color-map.svg": make_svg("Ночные фонари · цветной образец", colored, palette[BACKGROUND_PAINT]),
        "_contour-proof.svg": make_svg("Контуры для проверки", contours, "#FFFFFF"),
    }
    for filename, content in svg_files.items():
        (target / filename).write_text(content, encoding="utf-8")

    data = {
        "source": source.name, "source_sha256": SOURCE_HASH,
        "image_px": [width, height], "print_size_mm": list(PAGE),
        "art_box_mm": [0, 0, *PAGE],
        "line_width_mm": STROKE_MM, "font_size_pt": FONT_PT,
        "colors": len(rows), "regions": len(numbers), "connectivity": 4,
        "shared_arcs": len(arcs), "source_edges": source_edges,
        "changed_working_pixels": int(np.count_nonzero(master != indexed)),
        "working_pixels_total": indexed.size, "unfitted_labels": [], "segments": manifest,
    }
    (target / "segments.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Image.fromarray(indexed).save(target / "labels.png")
    Image.fromarray(ids.astype(np.uint16)).save(target / "region-ids.png")


def export_palette(target, rows, font_file):
    with (target / "palette.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, delimiter=";", lineterminator="\n")
        writer.writerow(("paint_id", "color_hex", "name_ru", "family_ru", "source_cluster"))
        for (paint, color, _), (name, family) in zip(rows, PAINTS):
            writer.writerow((paint, color, name, family, SOURCE_ORDER[paint - 1]))

    pdfmetrics.registerFont(TTFont("Palette regular", str(font_file)))
    page = canvas.Canvas(str(target / (STEMS["palette"] + ".pdf")), pagesize=A4, pageCompression=1)
    page.setTitle("Ночные фонари · Палитра из 19 красок")
    ink, muted = HexColor("#1A111F"), HexColor("#66556B")
    page.setFont("Palette regular", 30)
    page.setFillColor(ink)
    page.drawString(43, 773, "Ночные фонари")
    page.setFont("Palette regular", 12)
    page.setFillColor(muted)
    page.drawString(44, 746, "Палитра раскраски по номерам · 19 красок")

    top = 689
    families = dict.fromkeys(family for _, family in PAINTS)
    for family in families:
        page.setFillColor(ink)
        page.setFont("Palette regular", 12)
        page.drawString(44, top, family)
        subset = [((paint, color), name) for (paint, color, _), (name, group) in zip(rows, PAINTS)
                  if group == family]
        slot = 510 / len(subset)
        for column, ((paint, color), name) in enumerate(subset):
            x, y = 44 + column * slot, top - 66
            page.setFillColor(HexColor(color))
            page.roundRect(x, y, slot - 9, 49, 3.5, stroke=0, fill=1)
            page.setFillColor(ink)
            page.setFont("Palette regular", 10.2)
            page.drawString(x, y - 16, f"{paint}   {color}")
            page.setFillColor(muted)
            page.setFont("Palette regular", 8.1)
            page.drawString(x, y - 29, name)
        top -= 146
    page.showPage()
    page.save()


def pdf_path(page, data):
    """Pass the exact shared lines and cubics through to the PDF."""
    tokens = iter(re.findall(r"[MLCZ]|-?\d+(?:\.\d+)?", data))
    path = page.beginPath()

    def point():
        return float(next(tokens)) * mm, (PAGE[1] - float(next(tokens))) * mm

    for cmd in tokens:
        if cmd == "M":
            path.moveTo(*point())
        elif cmd == "L":
            path.lineTo(*point())
        elif cmd == "C":
            path.curveTo(*point(), *point(), *point())
        elif cmd == "Z":
            path.close()
        else:
            raise ValueError(f"Unexpected path command {cmd!r}")
    return path


def export_pdfs(target, font_file):
    ns = {"s": "http://www.w3.org/2000/svg"}
    pdfmetrics.registerFont(TTFont("Numbering", str(font_file)))
    for vector, key in (("numbered-outline", "scheme"), ("color-map", "color")):
        root = ET.parse(target / (vector + ".svg"))
        page = canvas.Canvas(str(target / (STEMS[key] + ".pdf")),
                             pagesize=tuple(k * mm for k in PAGE), pageCompression=1)
        page.setTitle(root.find("s:title", ns).text)
        page.setFillColor(HexColor(root.find("s:rect", ns).get("fill")))
        page.rect(0, 0, PAGE[0] * mm, PAGE[1] * mm, stroke=0, fill=1)
        if key == "scheme":
            boundaries = root.find("s:g[@id='boundaries']", ns)
            page.setStrokeColor(HexColor(boundaries.get("stroke")))
            page.setLineWidth(STROKE_MM * mm)
            page.setLineCap(1)
            page.setLineJoin(1)
            page.drawPath(pdf_path(page, boundaries.find("s:path", ns).get("d")), stroke=1, fill=0)
            labels = root.find("s:g[@id='paint-numbers']", ns)
            page.setFillColor(HexColor(labels.get("fill")))
            page.setFont("Numbering", FONT_PT)
            for item in labels:
                x, y = (float(item.get(axis)) for axis in ("x", "y"))
                page.drawString(x * mm, (PAGE[1] - y) * mm, item.text)
        else:
            for item in root.find("s:g[@id='paint-fills']", ns):
                page.setFillColor(HexColor(item.get("fill")))
                page.drawPath(pdf_path(page, item.get("d")), stroke=0, fill=1, fillMode=0)
        page.showPage()
        page.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("master", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    master, rows = read_master(args.master)
    bounds, font_file = glyph_bounds()
    compiled = compile_fields(transfer_details(master, rows), rows, bounds)
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)
    export_vectors(target, args.master, compiled, master, rows, bounds)
    export_palette(target, rows, font_file)
    export_pdfs(target, font_file)


if __name__ == "__main__":
    main()
