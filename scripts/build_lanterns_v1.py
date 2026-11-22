#!/usr/bin/env python3
"""Build the approved lantern scene as a 19-colour paint-by-numbers sheet.

Source pixels are never regenerated. The workflow downloads the approved PNG,
verifies its SHA-256, then this script derives a deterministic field map.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import cairosvg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "lanterns_hachette_19_v1"
SOURCE = OUT / "source_exact_approved.png"

EXPECTED_SHA256 = "af8f41ccffe0fa9232205e8ea0df0b0e88aa48176ddfbca6332b2d06857d5036"

# Exact artwork bounds inside the approved poster image.
CROP = (15, 11, 1070, 1334)  # x0, y0, x1, y1

# Exact centres of the 19 swatches embedded in the approved image.
SWATCH_X = [296, 339, 382, 425, 468, 511, 555, 598, 641, 684,
            728, 771, 814, 856, 898, 938, 977, 1015, 1053]
SWATCH_Y = 1375

PAGE_W_MM = 400.0
PAGE_H_MM = 600.0
STROKE_MM = 0.2
LABEL_PT = 5.3
FONT_MM = LABEL_PT * 25.4 / 72.0


def majority_filter(indexed: np.ndarray, ncolors: int = 19, k: int = 5) -> np.ndarray:
    counts = np.stack([
        cv2.boxFilter(
            (indexed == c).astype(np.uint8), cv2.CV_16U,
            (k, k), normalize=False, borderType=cv2.BORDER_REFLECT
        )
        for c in range(ncolors)
    ])
    return np.argmax(counts, axis=0).astype(np.uint8)


def component_stats(indexed: np.ndarray, ncolors: int = 19):
    ids = np.zeros(indexed.shape, np.int32)
    colors = [-1]
    sizes = [0]
    for c in range(ncolors):
        n, labs, stats, _ = cv2.connectedComponentsWithStats(
            (indexed == c).astype(np.uint8), connectivity=8
        )
        offset = len(sizes) - 1
        present = labs > 0
        ids[present] = labs[present] + offset
        colors.extend([c] * (n - 1))
        sizes.extend(stats[1:, cv2.CC_STAT_AREA].astype(int).tolist())
    return ids, np.asarray(colors, np.int16), np.asarray(sizes)


def border_contacts(ids: np.ndarray):
    pairlists = []
    for a, b in ((ids[:, :-1], ids[:, 1:]), (ids[:-1, :], ids[1:, :])):
        mask = a != b
        aa, bb = a[mask], b[mask]
        pairlists.append(
            np.stack((np.minimum(aa, bb), np.maximum(aa, bb)), axis=1)
        )
    pairs, counts = np.unique(
        np.concatenate(pairlists), axis=0, return_counts=True
    )
    contacts = defaultdict(list)
    for (a, b), count in zip(pairs, counts):
        contacts[int(a)].append((int(b), int(count)))
        contacts[int(b)].append((int(a), int(count)))
    return contacts


def merge_small(indexed: np.ndarray, min_area: int = 80, rounds: int = 5):
    work = indexed.copy()
    for _ in range(rounds):
        ids, colors, sizes = component_stats(work)
        small = np.flatnonzero(
            (sizes < min_area) & (np.arange(len(sizes)) > 0)
        )
        if not len(small):
            break
        contacts = border_contacts(ids)
        remap = colors.copy()
        changed = 0
        for rid in small:
            neighbours = contacts.get(int(rid), ())
            if not neighbours:
                continue
            best = max(neighbours, key=lambda item: (item[1], sizes[item[0]]))[0]
            remap[rid] = colors[best]
            changed += 1
        if not changed:
            break
        work = remap[ids].astype(np.uint8)
    return work


def clearance_and_positions(work: np.ndarray):
    ids, colors, sizes = component_stats(work)
    max_clear = np.zeros(len(sizes), dtype=np.float32)
    positions = [(0, 0)] * len(sizes)

    for c in range(19):
        mask = (work == c).astype(np.uint8)
        n, labs, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            continue
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        for local_id in range(1, n):
            ys, xs = np.where(labs == local_id)
            if not len(xs):
                continue
            local_values = dist[ys, xs]
            k = int(np.argmax(local_values))
            rid = int(ids[ys[k], xs[k]])
            max_clear[rid] = float(local_values[k])
            positions[rid] = (int(xs[k]), int(ys[k]))

    return ids, colors, sizes, max_clear, positions


def min_label_radius(number: int, font_px: float) -> float:
    text = str(number)
    estimated_width = (0.62 * len(text) + 0.30) * font_px
    return max(font_px * 0.69, estimated_width * 0.60)


def merge_until_labels_fit(
    indexed: np.ndarray,
    font_px: float,
    min_area: int = 50,
    rounds: int = 12,
):
    work = indexed.copy()
    history = []

    for _ in range(rounds):
        ids, colors, sizes, clearance, _ = clearance_and_positions(work)
        minimum = np.zeros(len(sizes), dtype=np.float32)
        for rid in range(1, len(sizes)):
            minimum[rid] = min_label_radius(int(colors[rid]) + 1, font_px)

        bad = np.flatnonzero(
            (np.arange(len(sizes)) > 0)
            & ((clearance < minimum) | (sizes < min_area))
        )
        history.append({"regions": int(len(sizes) - 1), "unfit": int(len(bad))})
        if not len(bad):
            return work, history

        contacts = border_contacts(ids)
        bad_set = set(map(int, bad))
        remap = colors.copy()
        changed = 0

        for rid in bad:
            neighbours = contacts.get(int(rid), ())
            if not neighbours:
                continue
            best = max(
                neighbours,
                key=lambda item: (
                    item[0] not in bad_set,
                    item[1],
                    sizes[item[0]],
                ),
            )[0]
            remap[rid] = colors[best]
            changed += 1

        if not changed:
            break
        work = remap[ids].astype(np.uint8)

    return work, history


def trace_shared_interfaces(indexed: np.ndarray, px_to_mm: float, xoff: float, yoff: float):
    h, w = indexed.shape
    stride = w + 1
    adjacency = defaultdict(list)

    horizontal = np.argwhere(indexed[1:, :] != indexed[:-1, :])
    vertical = np.argwhere(indexed[:, 1:] != indexed[:, :-1])

    for y, x in horizontal:
        a = int((y + 1) * stride + x)
        b = a + 1
        adjacency[a].append(b)
        adjacency[b].append(a)

    for y, x in vertical:
        a = int(y * stride + x + 1)
        b = a + stride
        adjacency[a].append(b)
        adjacency[b].append(a)

    visited = set()
    polylines = []

    def walk(first: int, second: int):
        chain = [first]
        a, b = first, second
        while True:
            edge = (min(a, b), max(a, b))
            if edge in visited:
                break
            visited.add(edge)
            chain.append(b)
            if len(adjacency[b]) != 2 or b == first:
                break
            neighbours = adjacency[b]
            nxt = neighbours[0] if neighbours[0] != a else neighbours[1]
            if (min(b, nxt), max(b, nxt)) in visited:
                break
            a, b = b, nxt
        return chain

    for start, neighbours in adjacency.items():
        if len(neighbours) == 2:
            continue
        for nxt in neighbours:
            if (min(start, nxt), max(start, nxt)) not in visited:
                polylines.append(walk(start, nxt))

    for start, neighbours in adjacency.items():
        for nxt in neighbours:
            if (min(start, nxt), max(start, nxt)) not in visited:
                polylines.append(walk(start, nxt))

    paths = []
    for nodes in polylines:
        if len(nodes) < 2:
            continue
        points = np.asarray(
            [(node % stride, node // stride) for node in nodes],
            dtype=np.float32,
        )
        closed = len(nodes) > 3 and nodes[0] == nodes[-1]
        simplified = cv2.approxPolyDP(
            points.reshape(-1, 1, 2), 1.0, closed
        ).reshape(-1, 2)
        if len(simplified) < 2:
            continue

        p0 = simplified[0]
        path = f"M{xoff+p0[0]*px_to_mm:.3f},{yoff+p0[1]*px_to_mm:.3f}"
        for x, y in simplified[1:]:
            path += f"L{xoff+x*px_to_mm:.3f},{yoff+y*px_to_mm:.3f}"
        if closed:
            path += "Z"
        paths.append(path)

    return " ".join(paths), len(paths)


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)

    raw = SOURCE.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != EXPECTED_SHA256:
        raise RuntimeError(
            f"Source SHA mismatch: expected {EXPECTED_SHA256}, got {actual_sha}"
        )

    full = np.asarray(Image.open(SOURCE).convert("RGB"))
    if full.shape[:2] != (1451, 1084):
        raise RuntimeError(f"Unexpected source dimensions: {full.shape[1]}x{full.shape[0]}")

    x0, y0, x1, y1 = CROP
    art = full[y0:y1, x0:x1].copy()
    Image.fromarray(art).save(OUT / "artwork_crop_reference.png")

    palette = np.asarray(
        [full[SWATCH_Y, x] for x in SWATCH_X],
        dtype=np.uint8,
    )

    # Critical: int32 prevents overflow while squaring Lab distances.
    lab_art = cv2.cvtColor(art, cv2.COLOR_RGB2LAB).astype(np.int32)
    lab_palette = cv2.cvtColor(
        palette[None, :, :], cv2.COLOR_RGB2LAB
    )[0].astype(np.int32)

    flat = lab_art.reshape(-1, 3)
    indexed_flat = np.empty(len(flat), dtype=np.uint8)
    chunk = 200_000
    for start in range(0, len(flat), chunk):
        sample = flat[start:start + chunk]
        diff = sample[:, None, :] - lab_palette[None, :, :]
        distance = (diff * diff).sum(axis=2)
        indexed_flat[start:start + chunk] = np.argmin(distance, axis=1)

    indexed = indexed_flat.reshape(art.shape[:2])

    px_to_mm = PAGE_W_MM / art.shape[1]
    font_px = FONT_MM / px_to_mm

    work = majority_filter(indexed, k=5)
    work = merge_small(work, min_area=80, rounds=5)
    work, merge_history = merge_until_labels_fit(
        work, font_px=font_px, min_area=50, rounds=12
    )

    Image.fromarray(palette[work]).save(OUT / "color_19_segmentation.png")

    ids, colors, sizes, clearances, positions = clearance_and_positions(work)
    region_count = int(len(sizes) - 1)

    failures = []
    for rid in range(1, len(sizes)):
        number = int(colors[rid]) + 1
        if clearances[rid] < min_label_radius(number, font_px):
            failures.append(rid)

    if failures:
        raise RuntimeError(
            f"{len(failures)} regions cannot fit a {LABEL_PT} pt label"
        )

    h, w = work.shape
    drawing_h = h * px_to_mm
    xoff = 0.0
    yoff = (PAGE_H_MM - drawing_h) / 2.0

    path, segment_count = trace_shared_interfaces(
        work, px_to_mm, xoff, yoff
    )

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{PAGE_W_MM:g}mm" height="{PAGE_H_MM:g}mm" '
        f'viewBox="0 0 {PAGE_W_MM:g} {PAGE_H_MM:g}">\n',
        f'<rect width="{PAGE_W_MM:g}" height="{PAGE_H_MM:g}" fill="white"/>\n',
        f'<g fill="none" stroke="#161616" stroke-width="{STROKE_MM}" '
        f'stroke-linecap="round" stroke-linejoin="round">\n',
        f'<rect x="{xoff:.3f}" y="{yoff:.3f}" '
        f'width="{PAGE_W_MM:.3f}" height="{drawing_h:.3f}"/>\n',
        f'<path d="{path}"/>\n',
        '</g>\n',
        f'<g fill="#111" font-family="DejaVu Sans, Arial, sans-serif" '
        f'font-size="{FONT_MM:.5f}" font-weight="400" '
        f'text-anchor="middle" dominant-baseline="central">\n',
    ]

    for rid in range(1, len(sizes)):
        number = int(colors[rid]) + 1
        x, y = positions[rid]
        svg.append(
            f'<text x="{xoff+(x+0.5)*px_to_mm:.3f}" '
            f'y="{yoff+(y+0.5)*px_to_mm:.3f}">{number}</text>\n'
        )

    svg.append('</g>\n</svg>\n')
    svg_text = "".join(svg)

    svg_path = OUT / "lineart_numbered_5.3pt_0.2mm.svg"
    svg_path.write_text(svg_text, encoding="utf-8")

    cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"),
        write_to=str(OUT / "lineart_numbered_preview.png"),
        output_width=2400,
        output_height=3600,
    )
    cairosvg.svg2pdf(
        bytestring=svg_text.encode("utf-8"),
        write_to=str(OUT / "lineart_numbered_5.3pt_0.2mm.pdf"),
    )

    with (OUT / "palette_19.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["Номер", "HEX", "R", "G", "B"])
        for i, (r, g, b) in enumerate(palette, start=1):
            r, g, b = int(r), int(g), int(b)
            writer.writerow([i, f"#{r:02X}{g:02X}{b:02X}", r, g, b])

    audit = {
        "source_exact_approved_sha256": actual_sha,
        "source_exact_preserved": True,
        "source_dimensions_px": [1084, 1451],
        "artwork_crop_px": list(CROP),
        "artwork_dimensions_px": [w, h],
        "paint_colors": 19,
        "white_counted_as_paint": False,
        "palette_source": "19 swatches embedded in approved image",
        "palette_order": "approved source order: dark/cool -> warm/light",
        "line_width_mm": STROKE_MM,
        "font_size_pt": LABEL_PT,
        "regions_total": region_count,
        "regions_numbered": region_count,
        "regions_without_number": len(failures),
        "all_regions_fit_5_3pt": not failures,
        "quantization_distance_dtype": "int32",
        "outline_paths": int(segment_count),
        "simplification": {
            "majority_filter": 5,
            "small_region_merge_px": 80,
            "label_fit_min_area_px": 50,
            "history": merge_history,
        },
    }
    (OUT / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = f"""# Ночные фонари — раскраска по номерам

Это версия, построенная **строго из утверждённого изображения** без
перегенерации исходной композиции.

- Красок: **19**
- Белый: **не считается краской**
- Линия: **0,2 мм**
- Номера: **5,3 pt**
- Областей: **{region_count}**
- Пронумеровано: **{region_count}**
- Областей без номера: **0**
- SHA-256 исходника: `{actual_sha}`

## Файлы

- `source_exact_approved.png` — утверждённый исходник, байт-в-байт.
- `lineart_numbered_5.3pt_0.2mm.pdf` — печатная ЧБ-схема.
- `lineart_numbered_5.3pt_0.2mm.svg` — векторная ЧБ-схема.
- `lineart_numbered_preview.png` — контрольное превью.
- `color_19_segmentation.png` — контрольная раскраска по тем же областям.
- `palette_19.csv` — 19 оттенков из исходного изображения.
- `audit.json` — автоматическая проверка ограничений.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    build()
