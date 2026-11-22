#!/usr/bin/env python3
"""Check the numbered SVG, both printed plates, and the palette against the map."""

import argparse
from collections import Counter
import csv
from io import BytesIO
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import cairosvg
import cv2
import fitz
import numpy as np
from PIL import Image

from face_transfer import OFFSET, SCALE

SVG = {"s": "http://www.w3.org/2000/svg"}
POINTS_PER_MM = 72 / 25.4


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_palette(folder):
    with (folder / "palette.csv").open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source, delimiter=";"))
    palette = {int(row["paint_id"]): row["color_hex"] for row in rows}
    check(list(palette) == list(range(1, 20)), "Palette numbers are missing or out of order")
    check(len(set(palette.values())) == 19, "Palette contains repeated paint values")
    check("#FFFFFF" not in palette.values(), "White must remain unpainted")
    return palette


def check_geometry(folder, data, palette):
    labels = np.asarray(Image.open(folder / "labels.png"))
    region_ids = np.asarray(Image.open(folder / "_region_ids.png"))
    segments = data["segments"]
    check(labels.shape[::-1] == tuple(data["image_px"]), "Label-map dimensions differ")
    check(set(np.unique(labels)) == set(palette), "Some paints are missing from the map")
    check(all(segment["label_fits"] for segment in segments) and not data["unfitted_labels"],
          "The map has an unplaced number")
    count = sum(cv2.connectedComponents(np.uint8(labels == paint), connectivity=4)[0] - 1
                for paint in palette)
    check(count == len(segments) == data["regions"], "The map does not have one record per region")
    for segment in segments:
        x, y = int(segment["label_x_px"]), int(segment["label_y_px"])
        check(region_ids[y, x] == segment["segment_id"], "A number is in the wrong field")
        check(labels[y, x] == segment["paint_id"], "A number points to the wrong paint")
    samples = labels.shape[1] / 1122
    face_paints = ((524, 812, 19), (742, 752, 19), (747, 844, 17),
                   (725, 808, 16), (501, 869, 16), (723, 1072, 16))
    for x, y, paint in face_paints:
        row = round((OFFSET[1] + SCALE * y) * samples)
        col = round((OFFSET[0] + SCALE * x) * samples)
        check(labels[row, col] == paint, f"The approved V10 face paint moved at {x}, {y}")
    check(data["paint_pixels_changed_outside_face"] == 0, "The poster changed beyond the face")


def check_printed_boundaries(folder, data):
    """Compare disconnected paper on the vector proof with each full number footprint."""
    scale = 12
    w, h = data["print_size_mm"]
    proof = cairosvg.svg2png(url=str(folder / "_unlabelled.svg"),
                             output_width=w * scale, output_height=h * scale)
    gray = np.asarray(Image.open(BytesIO(proof)).convert("L"))
    _, connected, stats, _ = cv2.connectedComponentsWithStats(
        np.uint8(gray >= 165), connectivity=8
    )
    outside = int(connected[0, 0])
    occupied = set()
    for segment in data["segments"]:
        x0, y0, x1, y1 = np.array(segment["bounds_mm"]) * scale
        rect = connected[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))]
        field = int(rect[rect.shape[0] // 2, rect.shape[1] // 2])
        check(field not in (0, outside), "A number intersects a line or falls outside the artwork")
        check(field not in occupied, "Multiple numbers share one printed field")
        check(np.all(rect == field), "The printed contour crosses a number footprint")
        occupied.add(field)
    extras = [int(area) for field, area in enumerate(stats[:, -1])
              if field not in (0, outside) and field not in occupied]
    check(not extras, "A printed field still needs a number")
    return len(occupied)


def check_pdf(folder, data, palette):
    outline = ET.parse(folder / "numbered-outline.svg")
    color = ET.parse(folder / "color-map.svg")
    labels = outline.findall(".//s:g[@id='paint-numbers']/s:text", SVG)
    expected = [(segment["segment_id"], segment["paint_id"]) for segment in data["segments"]]
    check([(int(mark.attrib["data-region"]), int(mark.text)) for mark in labels] == expected,
          "SVG numbers differ from the region map")
    vector_fills = {int(path.attrib["data-region"]): path.attrib["fill"] for path in
                    color.findall(".//s:g[@id='paint-fills']/s:path", SVG)}
    check(vector_fills == {region: palette[paint] for region, paint in expected},
          "The color SVG applies a different paint to a numbered region")
    expected_words = Counter(str(paint) for _, paint in expected)
    with fitz.open(folder / "print-plates.pdf") as document:
        check(len(document) == 2, "The print PDF must contain two pages")
        for page in document:
            for measured, desired in zip((page.rect.width, page.rect.height), data["print_size_mm"]):
                check(abs(measured / POINTS_PER_MM - desired) < 0.01, "Unexpected PDF trim size")
            check(not page.get_images(), "The print PDF must retain vector artwork")
        page = document[0]
        words = Counter(word[4] for word in page.get_text("words"))
        check(words == expected_words, "The PDF drops, joins or changes some numbers")
        traces = page.get_texttrace()
        check(all(abs(span["size"] - 5.3) < 0.002 for span in traces), "Incorrect printed font size")
        printed = [char for span in traces for char in span["chars"]]
        check("".join(chr(char[0]) for char in printed) == "".join(mark.text for mark in labels),
              "The PDF changes the order or content of the numbers")
        index = 0
        for mark in labels:
            baseline = printed[index][2]
            svg_position = (float(mark.attrib["x"]), float(mark.attrib["y"]))
            check(all(abs(measured / POINTS_PER_MM - expected_mm) < 0.012 for measured, expected_mm in
                      zip(baseline, svg_position)), "The PDF displaced a printed number")
            index += len(mark.text)
        strokes = [draw["width"] / POINTS_PER_MM for draw in page.get_drawings()
                   if draw.get("width") is not None]
        check(bool(strokes) and max(abs(width - 0.2) for width in strokes) < 0.001,
              "The PDF changed the contour thickness")

        page = document[1]
        check(not page.get_text().strip(), "The clean color plate should contain only the painting")
        fills = [draw["fill"] for draw in page.get_drawings() if draw.get("fill") is not None]
        colors = [tuple(round(channel * 255) for channel in fill) for fill in fills]
        wanted = [(255, 255, 255)] + [tuple(bytes.fromhex(palette[paint][1:]))
                                      for _, paint in expected]
        check(colors == wanted, "The exported PDF changes the vector pigments")
        proof = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        for segment in data["segments"]:
            x0, y0, x1, y1 = segment["bounds_mm"]
            center = ((x0 + x1) * POINTS_PER_MM, (y0 + y1) * POINTS_PER_MM)
            sample = proof.pixel(*(int(value) for value in center))
            pigment = palette[segment["paint_id"]]
            target = tuple(bytes.fromhex(pigment[1:]))
            # MuPDF truncates PDF device-RGB floats at the 8-bit raster boundary.
            check(max(abs(actual - expected) for actual, expected in zip(sample, target)) <= 1,
                  "The color PDF differs at a numbered field")
    with fitz.open(folder / "palette.pdf") as document:
        check(len(document) == 1, "The palette must be a separate one-page PDF")
        content = document[0].get_text()
        check(all(f"{paint:02d}   {value}" in content for paint, value in palette.items()),
              "The PDF palette is missing a visible number or color")
        check(document[0].get_pixmap(matrix=fitz.Matrix(.3, .3)).height > 0,
              "The palette PDF cannot render")


def validate(folder):
    data = json.loads((folder / "segments.json").read_text(encoding="utf-8"))
    palette = read_palette(folder)
    check_geometry(folder, data, palette)
    checked = check_printed_boundaries(folder, data)
    check_pdf(folder, data, palette)
    result = {
        "regions_total": data["regions"], "regions_numbered": checked,
        "regions_without_number": data["regions"] - checked, "colors_total": len(palette),
        "white_counted_as_paint": False, "font_size_pt": data["font_size_pt"],
        "line_width_mm": data["line_width_mm"], "print_size_mm": data["print_size_mm"],
        "vector_interiors_verified_with_clearance_mm": data["vector_topology_checked_mm"],
        "printed_proof_dpi": 304.8,
        "all_color_pdf_samples_match": True,
        "approved_face_v10_commit": data["approved_face_v10_commit"],
        "face_transfer_bbox_native_px": data["face_transfer_bbox_native_px"],
        "paint_pixels_changed_outside_face": data["paint_pixels_changed_outside_face"],
        "under_far_eye_paint": 17,
        "v9_indexed_sha256": data["v9_indexed_sha256"],
        "v9_preview_sha256": data["v9_preview_sha256"],
    }
    (folder / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(validate(parser.parse_args().output), ensure_ascii=False, indent=2))
