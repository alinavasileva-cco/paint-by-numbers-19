#!/usr/bin/env python3
"""Build the colored artwork for approval and its reusable field geometry."""

import argparse
import csv
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import cairosvg
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "night_lanterns_v11"))
from build import SAMPLES, STROKE_MM, compile_fields, glyph_bounds
from geometry import svg_paths

CROP = (16, 12, 1069, 1332)
ART_WIDTH_MM = 388
NS = "http://www.w3.org/2000/svg"


def paints():
    with (HERE / "palette.csv").open(encoding="utf-8", newline="") as source:
        return [(int(row["paint_id"]), row["color_hex"], tuple(bytes.fromhex(row["color_hex"][1:])))
                for row in csv.DictReader(source)]


def lab(pixels):
    return cv2.cvtColor(np.float32(pixels) / 255, cv2.COLOR_RGB2Lab)


def png_array(svg, shape):
    content = cairosvg.svg2png(bytestring=svg, output_width=shape[1], output_height=shape[0])
    return np.asarray(Image.open(BytesIO(content)).convert("RGBA"))


def selected_path(root, name, shape):
    layer = ET.Element(f"{{{NS}}}svg", root.attrib)
    layer.append(deepcopy(root.find(f".//*[@id='{name}']")))
    return png_array(ET.tostring(layer), shape)[..., 3] >= 128


def check_far_eye(profile, eye):
    if np.count_nonzero(eye & ~profile):
        raise ValueError("The far eye leaves the drawn facial profile")


def artist_map(source, palette):
    """Reduce the scenery into planes and draw the focal details as curves."""
    image = Image.open(source).convert("RGB")
    if image.size != (1084, 1451):
        raise ValueError("Use the attached scene-and-palette reference")
    rgb = np.asarray(image.crop(CROP))
    blurred = cv2.bilateralFilter(rgb, 9, 25, 4)
    tree = cKDTree(lab(np.array([[row[2] for row in palette]], np.uint8))[0])
    _, colors = tree.query(lab(blurred).reshape(-1, 3), workers=2)
    indexed = (colors.reshape(rgb.shape[:2]) + 1).astype(np.uint8)

    height, width = indexed.shape
    y = np.arange(height)[:, None] + CROP[1]
    x = np.arange(width)[None, :] + CROP[0]
    portrait = (310 < x) & (x < 730) & (246 < y) & (y < 1001)
    votes = np.full(indexed.shape, -1., np.float32)
    smoothed = indexed.copy()
    for paint, *_ in palette:
        one = np.float32(indexed == paint)
        broad = cv2.GaussianBlur(one, (0, 0), 1.23)
        fine = cv2.GaussianBlur(one, (0, 0), .74)
        score = np.where(portrait, fine, broad)
        replace = score > votes
        smoothed[replace] = paint
        votes[replace] = score[replace]

    base = cv2.resize(smoothed, (width * SAMPLES, height * SAMPLES), interpolation=cv2.INTER_NEAREST)
    master = ET.parse(HERE / "artwork.svg").getroot()
    shaped = png_array(ET.tostring(master), base.shape)
    mask = shaped[..., 3] >= 128
    pixels, inverse = np.unique(shaped[mask, :3], axis=0, return_inverse=True)
    near_color = tree.query(lab(pixels[None])[0])[1]
    base[mask] = np.uint8(near_color[inverse] + 1)

    face = selected_path(master, "portrait-profile", base.shape)
    far_eye = selected_path(master, "far-eye", base.shape)
    check_far_eye(face, far_eye)
    return base, face, master


def color_svg(fills, colors, palette, master, w_mm, h_mm):
    lookup = [""] + [row[1] for row in palette]
    shapes = []
    for region, path in fills.items():
        color = lookup[colors[region]]
        shapes.append(f'<path data-region="{region}" fill="{color}" stroke="{color}" d="{path}"/>')
    ink = master.find(f".//{{{NS}}}g[@id='open-ink']")
    strokes = "".join(f'<path stroke="{p.attrib["stroke"]}" d="{p.attrib["d"]}"/>' for p in ink)
    scale = w_mm / (CROP[2] - CROP[0])
    tx, ty = -CROP[0] * scale, -CROP[1] * scale
    return (f'<svg xmlns="{NS}" width="{w_mm}mm" height="{h_mm:.5f}mm" viewBox="0 0 {w_mm} {h_mm:.5f}">\n'
            '<title>Ночь тысячи фонарей</title>\n'
            f'<rect width="{w_mm}" height="{h_mm:.5f}" fill="{lookup[3]}"/>\n'
            '<g id="paint-fields" fill-rule="evenodd" stroke-width=".20" stroke-linejoin="round">\n'
            + "\n".join(shapes) + '\n</g>\n'
            f'<g id="open-ink" transform="translate({tx:.7f} {ty:.7f}) scale({scale:.9f})" '
            f'stroke-width="{STROKE_MM / scale:.9f}" stroke-linecap="round" fill="none">\n'
            f'{strokes}\n</g>\n</svg>\n')


def write_indexed(path, labels, palette):
    image = Image.fromarray(labels.astype(np.uint8), "P")
    image.putpalette([0, 0, 0] + [channel for _, _, rgb in palette for channel in rgb])
    image.save(path, optimize=True)


def export_color(source, output):
    output.mkdir(parents=True, exist_ok=True)
    palette = paints()
    if len(palette) != 19 or len({row[1] for row in palette}) != 19:
        raise ValueError("Expected exactly 19 distinct paints")
    artist, face, master = artist_map(source, palette)
    unit = ART_WIDTH_MM / artist.shape[1]
    bounds, _ = glyph_bounds()
    indexed, ids, colors, positions, arcs, rings, edges = compile_fields(
        artist, palette, unit, bounds, detail_spacing=False
    )
    if set(np.unique(indexed)) != set(range(1, 20)):
        raise ValueError("One of the 19 palette colors is absent")

    eye_regions = []
    for x, y, paint in ((634, 431, 1), (636, 449, 16), (640, 434, 19)):
        row, col = (y - CROP[1]) * SAMPLES, (x - CROP[0]) * SAMPLES
        region = int(ids[row, col])
        if colors[region] != paint:
            raise ValueError(f"The far eye lost paint {paint} during field fitting")
        eye_regions.append(region)
    check_far_eye(face, np.isin(ids, eye_regions))

    _, fills = svg_paths(arcs, rings, (0, 0), unit)
    if fills.keys() != {position[0] for position in positions}:
        raise ValueError("Traced field IDs differ from the verified label positions")
    svg = color_svg(fills, colors, palette, master, ART_WIDTH_MM, artist.shape[0] * unit)
    (output / "color-master.svg").write_text(svg, encoding="utf-8")
    write_indexed(output / "color-indexed.png", indexed, palette)
    smooth = png_array(svg.encode(), artist.shape)
    if np.any(smooth[..., 3] != 255):
        raise ValueError("The color export has transparent gaps")
    Image.fromarray(smooth[..., :3]).save(output / "color-master.png")
    data = {
        "stage": "COLOR_FOR_APPROVAL",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "paint_count": len(palette),
        "field_count": len(positions),
        "fields_with_room_at_5_3_pt": len(positions),
        "art_width_mm_for_fit_check": ART_WIDTH_MM,
        "trace_checked_with_clearance_mm": 0.19,
        "shared_grid_edges": edges,
        "far_eye_within_drawn_face": True,
        "far_eye_paints_retained": [1, 16, 19],
    }
    (output / "audit.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2))
    return indexed, ids, face


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export_color(args.reference, args.output)
