#!/usr/bin/env python3
"""Check the actual printed plates and their shared region map."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import fitz
import numpy as np
from PIL import Image

from build import (BACKGROUND_PAINT, FONT_PT, PAGE, PROOF_PX_PER_MM, SAMPLES,
                   SOURCE_HASH, STEMS, STROKE_MM, read_master, scan_paper)

POINTS_PER_MM = 72 / 25.4
SVG = {"s": "http://www.w3.org/2000/svg"}


def check(condition, description):
    if not condition:
        raise ValueError(description)


def read_map(folder, master):
    data = json.loads((folder / "segments.json").read_text(encoding="utf-8"))
    with (folder / "palette.csv").open(encoding="utf-8-sig", newline="") as source:
        paints = {int(row["paint_id"]): row["color_hex"] for row in csv.DictReader(source, delimiter=";")}
    master_pixels, approved_paints = read_master(master)
    check(paints == {paint: color for paint, color, _ in approved_paints}, "The palette changed")
    check(data["source_sha256"] == SOURCE_HASH, "Wrong master")
    check(data["print_size_mm"] == list(PAGE) and data["art_box_mm"] == [0, 0, *PAGE], "Wrong art box")

    with Image.open(folder / "labels.png") as image:
        labels = np.asarray(image).copy()
    with Image.open(folder / "region-ids.png") as image:
        ids = np.asarray(image).copy()
    check(labels.shape == ids.shape == master_pixels.shape, "Wrong map dimensions")
    segments = data["segments"]
    expected_ids = list(range(1, len(segments) + 1))
    check(data["regions"] == len(segments) and not data["unfitted_labels"], "Incomplete region map")
    check(np.unique(ids).tolist() == expected_ids, "An area is missing a region ID")
    check([region["segment_id"] for region in segments] == expected_ids, "Duplicate or missing region records")
    check(set(np.unique(labels)) == set(paints), "All 19 colors must occur on the sheet")

    regions_by_color = np.array([0] + [region["paint_id"] for region in segments], dtype=np.uint8)
    check(np.array_equal(regions_by_color[ids], labels), "A region contains two paint numbers")
    for paint in paints:
        count, connected = cv2.connectedComponents(np.uint8(labels == paint), connectivity=4)
        parts = np.unique(ids[labels == paint])
        check(count == len(parts) + 1, f"Paint {paint} has a mislabeled component")
        for region in (record for record in segments if record["paint_id"] == paint):
            x, y = int(region["label_x_px"]), int(region["label_y_px"])
            check(ids[y, x] == region["segment_id"] and region["label_fits"], "Incorrect region number location")
            connected_index = connected[y, x]
            check(connected_index > 0, "A label is outside its paint component")
            connected[y, x] = -connected_index
        connected_labels = np.unique(connected[connected < 0])
        check(len(connected_labels) == len(parts), "At least one connected component has no number")

    for name, x, y, paint in (
        ("near-eye glint", 575, 452, 5),
        ("near-iris light", 567, 472, 2),
        ("far-iris light", 635, 450, 2),
        ("far-eye light", 643, 435, 5),
        ("left flower center", 348, 866, 2),
        ("sleeve flower center", 481, 998, 2),
    ):
        check(int(labels[y * SAMPLES, x * SAMPLES]) == paint, f"Missing painted detail: {name}")

    return data, paints


def printed_fields(folder, data):
    """Flood-fill the white paper inside the actual vector stroke at print scale."""
    scale = PROOF_PX_PER_MM
    proof = (folder / "_contour-proof.svg").read_text(encoding="utf-8")
    count, cells, stats, centers = scan_paper(proof)
    seen = set()
    for record in data["segments"]:
        x0, y0, x1, y1 = np.asarray(record["bounds_mm"]) * scale
        internal = cells[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))]
        ink_clearance = int(internal[internal.shape[0] // 2, internal.shape[1] // 2])
        check(ink_clearance > 0 and np.all(internal == ink_clearance),
              f"Contour crosses the number in region {record['segment_id']}")
        check(ink_clearance not in seen, f"More than one label in printed region {record['segment_id']}")
        seen.add(ink_clearance)
    extras = set(range(1, count)) - seen
    details = [(stats[cell][-1].item(), np.round(centers[cell] / scale, 3).tolist()) for cell in sorted(extras)]
    check(not extras, f"Printed cells without a label (proof pixels, center mm): {details[:16]}")
    return count - 1


def path_edges(drawings):
    """Normalize the direction of each line/cubic in the exported PDF."""
    edges = Counter()
    for path in drawings:
        for operation, *points in path["items"]:
            check(operation in ("l", "c"), f"Unexpected contour operator {operation!r}")
            branch = tuple((round(point.x, 3), round(point.y, 3)) for point in points)
            if branch[0] != branch[-1] or operation == "c":
                edges[(operation, min(branch, branch[::-1]))] += 1
    return edges


def frame_edge(edge, rect):
    points = edge[1]
    for axis, extent in enumerate((rect.width, rect.height)):
        for edge_coordinate in (0, round(extent, 3)):
            if all(point[axis] == edge_coordinate for point in points):
                return True
    return False


def check_page(page):
    check(abs(page.rect.width / POINTS_PER_MM - PAGE[0]) < 0.01, "Incorrect PDF width")
    check(abs(page.rect.height / POINTS_PER_MM - PAGE[1]) < 0.01, "Incorrect PDF height")
    check(not page.get_images(), "Print PDF unexpectedly contains raster artwork")


def check_pdfs(folder, data, paints):
    svg = ET.parse(folder / "numbered-outline.svg")
    marks = svg.findall(".//s:g[@id='paint-numbers']/s:text", SVG)
    expected = [(r["segment_id"], r["paint_id"]) for r in data["segments"]]
    check([(int(mark.attrib["data-region"]), int(mark.text)) for mark in marks] == expected, "Wrong SVG labels")
    color_svg = ET.parse(folder / "color-map.svg")
    fills = color_svg.findall(".//s:g[@id='paint-fills']/s:path", SVG)
    check([(int(path.attrib["data-region"]), path.attrib["fill"]) for path in fills] ==
          [(region, paints[paint]) for region, paint in expected], "Wrong SVG fill")

    with fitz.open(folder / (STEMS["scheme"] + ".pdf")) as document:
        check(len(document) == 1, "The scheme PDF must have one page")
        page = document[0]
        check_page(page)
        spans = page.get_texttrace()
        check(spans and all(abs(span["size"] - FONT_PT) < 0.002 for span in spans), "Wrong printed label size")
        check(len({span["font"] for span in spans}) == 1, "The print numbers use different faces")
        check(any(document.extract_font(font[0])[3] for font in page.get_fonts()), "The number font is not embedded")
        printed = [char for span in spans for char in span["chars"]]
        check("".join(chr(char[0]) for char in printed) == "".join(mark.text for mark in marks),
              "The print numbers differ from the SVG")
        index = 0
        for mark in marks:
            baseline = printed[index][2]
            exact = tuple(float(mark.attrib[axis]) for axis in ("x", "y"))
            check(max(abs(a / POINTS_PER_MM - b) for a, b in zip(baseline, exact)) < 0.008,
                  "A number moved during PDF export")
            index += len(mark.text)
        drawing = [path for path in page.get_drawings() if path.get("width") is not None]
        check(drawing and all(abs(path["width"] / POINTS_PER_MM - STROKE_MM) < 0.001 for path in drawing),
              "Wrong thickness of printed contours")
        bw_edges = path_edges(drawing)
        check(set(bw_edges.values()) == {1}, "The scheme contains doubled boundaries")

    with fitz.open(folder / (STEMS["color"] + ".pdf")) as document:
        check(len(document) == 1, "The color PDF must have one page")
        page = document[0]
        check_page(page)
        check(not page.get_text().strip(), "The color plate contains unwanted text")
        vector = page.get_drawings()
        rgb_fills = [tuple(round(value * 255) for value in path["fill"]) for path in vector]
        target_rgb = [tuple(bytes.fromhex(paints[BACKGROUND_PAINT][1:]))]
        target_rgb.extend(tuple(bytes.fromhex(paints[paint][1:])) for _, paint in expected)
        check(rgb_fills == target_rgb, "The color PDF uses incorrect pigments")
        color_edges = path_edges(vector[1:])
        check(color_edges.keys() == bw_edges.keys(), "Some printed color boundaries differ from the scheme")
        expected_edge_counts = {edge: 1 if frame_edge(edge, page.rect) else 2 for edge in bw_edges}
        check(color_edges == expected_edge_counts, "The vector fields do not share every interior boundary")
        raster = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        for record in data["segments"]:
            x0, y0, x1, y1 = record["bounds_mm"]
            sample = raster.pixel(int((x0 + x1) * POINTS_PER_MM), int((y0 + y1) * POINTS_PER_MM))
            target = bytes.fromhex(paints[record["paint_id"]][1:])
            check(max(abs(a - b) for a, b in zip(sample, target)) <= 1, "Wrong color underneath a label")

    with fitz.open(folder / (STEMS["palette"] + ".pdf")) as document:
        check(len(document) == 1, "The palette must have one page")
        page = document[0]
        extracted = page.get_text()
        check(all(f"{paint}   {hexcode}" in extracted for paint, hexcode in paints.items()), "Missing swatch legend")
        swatches = [tuple(round(rgb * 255) for rgb in item["fill"]) for item in page.get_drawings()]
        check(swatches == [tuple(bytes.fromhex(color[1:])) for color in paints.values()], "Wrong palette swatches")

    return len(bw_edges)


def validate(folder, master):
    data, paints = read_map(folder, master)
    labels = printed_fields(folder, data)
    checked_edges = check_pdfs(folder, data, paints)
    result = {
        "source_sha256": data["source_sha256"],
        "print_size_mm": list(PAGE),
        "colors": len(paints),
        "regions_total": data["regions"],
        "regions_numbered": labels,
        "regions_without_number": data["regions"] - labels,
        "font_size_pt": FONT_PT,
        "line_width_mm": STROKE_MM,
        "proof_resolution_px_per_mm": PROOF_PX_PER_MM,
        "shared_pdf_edges_checked": checked_edges,
        "color_match_at_all_number_locations": True,
    }
    (folder / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("master", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.output, args.master), ensure_ascii=False, indent=2))
