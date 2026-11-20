#!/usr/bin/env python3
"""Vector paint-by-numbers review based strictly on an existing PNG.

Never regenerates or replaces the source. Exports an SVG with 0.2 mm
contours and 5.3 pt labels. Unnumberable areas are explicitly audited.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

PAGE_W_MM = 400.0
PAGE_H_MM = 600.0
STROKE_MM = 0.2
LABEL_PT = 5.3
FONT_MM = LABEL_PT * 25.4 / 72.0


def label_components(indexed, ncolors):
    ids = np.zeros(indexed.shape, dtype=np.int32)
    region_colors = [-1]
    areas = [0]
    for color in range(ncolors):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (indexed == color).astype(np.uint8), connectivity=8)
        offset = len(areas) - 1
        mask = labels > 0
        ids[mask] = labels[mask] + offset
        region_colors.extend([color] * (n - 1))
        areas.extend(stats[1:, cv2.CC_STAT_AREA].astype(int).tolist())
    return ids, np.asarray(region_colors, dtype=np.int16), np.asarray(areas)


def majority_filter(indexed, ncolors, diameter=3):
    if diameter <= 1:
        return indexed.copy()
    counts = np.stack([
        cv2.boxFilter((indexed == color).astype(np.uint8), cv2.CV_16U,
                      (diameter, diameter), normalize=False,
                      borderType=cv2.BORDER_REFLECT)
        for color in range(ncolors)
    ])
    winner = np.argmax(counts, axis=0).astype(np.uint8)
    original_score = np.take_along_axis(counts, indexed[None], axis=0)[0]
    winner[original_score == counts.max(axis=0)] = indexed[
        original_score == counts.max(axis=0)]
    return winner


def remove_speckles(indexed, ncolors, min_area=30, rounds=4):
    """Merge small components into the neighbor sharing the longest border."""
    work = indexed.copy()
    for _ in range(rounds):
        ids, colors, sizes = label_components(work, ncolors)
        to_merge = np.flatnonzero(
            (sizes < min_area) & (np.arange(len(sizes)) > 0))
        if not len(to_merge):
            break
        edges = []
        for left, right in (
            (ids[:, :-1], ids[:, 1:]), (ids[:-1], ids[1:])
        ):
            mask = left != right
            a, b = left[mask], right[mask]
            edges.append(np.stack(
                (np.minimum(a, b), np.maximum(a, b)), axis=1))
        pairs, shared = np.unique(
            np.concatenate(edges), axis=0, return_counts=True)
        contacts = defaultdict(list)
        for (a, b), count in zip(pairs, shared):
            contacts[int(a)].append((int(b), int(count)))
            contacts[int(b)].append((int(a), int(count)))
        target_colors = colors.astype(np.uint8).copy()
        actual = 0
        for region in to_merge:
            neighbors = contacts.get(int(region))
            if not neighbors:
                continue
            best, _ = max(
                neighbors,
                key=lambda nb: (
                    nb[1], sizes[nb[0]] >= min_area, sizes[nb[0]]))
            target_colors[region] = colors[best]
            actual += 1
        if not actual:
            break
        work = target_colors[ids]
    return work


def palette_indices(palette):
    """Sort navy/violet -> burgundy/red -> orange -> yellow/cream."""
    hsv = cv2.cvtColor(
        palette[None, :, :].astype(np.uint8), cv2.COLOR_RGB2HSV)[0]
    degrees = hsv[:, 0].astype(float) * 2.0
    rotated = (degrees - 205.0) % 360.0
    return sorted(range(len(palette)),
                  key=lambda i: (round(rotated[i] / 30),
                                 float(hsv[i, 2])))


def runs(flag):
    transitions = np.flatnonzero(
        np.diff(np.r_[False, flag, False]))
    return transitions.reshape(-1, 2)


def outline_svg(indexed, px_to_mm, xoff, yoff):
    """One crisp vector stroke along every interface; no double outlines."""
    height, width = indexed.shape
    scale = px_to_mm
    items = []
    horizontal = indexed[1:, :] != indexed[:-1, :]
    for y in range(height - 1):
        for x0, x1 in runs(horizontal[y]):
            items.append(
                f'M{xoff+x0*scale:.3f},{yoff+(y+1)*scale:.3f}'
                f'h{(x1-x0)*scale:.3f}')
    vertical = indexed[:, 1:] != indexed[:, :-1]
    for x in range(width - 1):
        for y0, y1 in runs(vertical[:, x]):
            items.append(
                f'M{xoff+(x+1)*scale:.3f},{yoff+y0*scale:.3f}'
                f'v{(y1-y0)*scale:.3f}')
    return ' '.join(items), len(items)


def create_labels(indexed, palette_order, px_to_mm, xoff, yoff):
    """Never make labels smaller than 5.3pt to force them to fit."""
    output = []
    failures = []
    all_regions = 0
    counts_by_color = {}
    for color_id in range(len(palette_order)):
        number = palette_order[color_id]
        mask = (indexed == color_id).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        maxima = ndi.maximum(
            dist, labels=labels, index=np.arange(1, n))
        positions = ndi.maximum_position(
            dist, labels=labels, index=np.arange(1, n))
        placed = 0
        for component in range(1, n):
            all_regions += 1
            text = str(number)
            font_pixels = FONT_MM / px_to_mm
            estimated_width = (
                0.62 * len(text) + 0.30) * font_pixels
            minimum_radius = max(
                font_pixels * 0.69, estimated_width * 0.60)
            y, x = positions[component - 1]
            clearance = float(maxima[component - 1])
            if clearance < minimum_radius:
                failures.append({
                    'color_number': number,
                    'area_px': int(stats[
                        component, cv2.CC_STAT_AREA]),
                    'max_clearance_px': round(clearance, 2),
                    'approximate_position': [int(x), int(y)]})
                continue
            output.append((
                number,
                round(xoff + (x + 0.5) * px_to_mm, 3),
                round(yoff + (y + 0.5) * px_to_mm, 3)))
            placed += 1
        counts_by_color[number] = {
            'regions': n - 1, 'numbered': placed}
    return output, failures, all_regions, counts_by_color


def build(source, dest, majority=3, minimum=30, rounds=4):
    dest.mkdir(parents=True, exist_ok=True)
    raw = source.read_bytes()
    image = np.asarray(Image.open(source).convert('RGB'))
    palette, inverted = np.unique(
        image.reshape(-1, 3), axis=0, return_inverse=True)
    if len(palette) > 19:
        raise ValueError(
            f'В изображении {len(palette)} цветов; максимум 19.')
    indexed = inverted.reshape(image.shape[:2]).astype(np.uint8)
    simplified = remove_speckles(
        majority_filter(indexed, len(palette), majority),
        len(palette), min_area=minimum, rounds=rounds)
    height, width = simplified.shape
    px_to_mm = min(
        PAGE_W_MM / width, PAGE_H_MM / height)
    drawing_w = width * px_to_mm
    drawing_h = height * px_to_mm
    offset_x = (PAGE_W_MM - drawing_w) / 2
    offset_y = (PAGE_H_MM - drawing_h) / 2
    palette_sort = palette_indices(palette)
    color_to_number = {
        color_id: number for number, color_id in
        enumerate(palette_sort, start=1)}
    path, nsegments = outline_svg(
        simplified, px_to_mm, offset_x, offset_y)
    numbers, failures, nregions, by_color = create_labels(
        simplified, color_to_number, px_to_mm,
        offset_x, offset_y)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{PAGE_W_MM:g}mm" height="{PAGE_H_MM:g}mm" '
        f'viewBox="0 0 {PAGE_W_MM:g} {PAGE_H_MM:g}">\n',
        '<rect width="400" height="600" fill="white"/>\n',
        f'<g fill="none" stroke="#151515" '
        f'stroke-width="{STROKE_MM}" '
        f'stroke-linecap="round" stroke-linejoin="round">\n',
        f'<rect x="{offset_x:.3f}" y="{offset_y:.3f}" '
        f'width="{drawing_w:.3f}" '
        f'height="{drawing_h:.3f}"/>\n',
        f'<path d="{path}"/>\n',
        '</g>\n',
        f'<g fill="#111" '
        f'font-family="DejaVu Sans, Arial, sans-serif" '
        f'font-weight="400" font-size="{FONT_MM:.5f}" '
        f'text-anchor="middle" dominant-baseline="central">\n']
    parts.extend(
        f'<text x="{x}" y="{y}">{number}</text>\n'
        for number, x, y in numbers)
    parts.append('</g>\n</svg>\n')
    (dest / 'lineart_review.svg').write_text(
        ''.join(parts), encoding='utf-8')
    Image.fromarray(palette[simplified]).save(
        dest / 'color_segmentation_review.png')
    with (dest / 'palette.csv').open(
        'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Номер', 'HEX', 'R', 'G', 'B'])
        for i in palette_sort:
            r, g, b = map(int, palette[i])
            writer.writerow([
                color_to_number[i], f'#{r:02X}{g:02X}{b:02X}',
                r, g, b])
    changed = int(np.count_nonzero(
        simplified != indexed))
    report = {
        'source_filename': source.name,
        'source_sha256': hashlib.sha256(raw).hexdigest(),
        'source_dimensions_px': [width, height],
        'source_distinct_colors': len(palette),
        'palette_sorted_by_hue': True,
        'source_unchanged': True,
        'rendered_page_mm': [PAGE_W_MM, PAGE_H_MM],
        'artwork_size_mm': [
            round(drawing_w, 3), round(drawing_h, 3)],
        'contour_width_mm': STROKE_MM,
        'font_size_pt': LABEL_PT,
        'majority_filter_size': majority,
        'min_speckle_area_px': minimum,
        'speckle_merge_rounds': rounds,
        'changed_pixels': changed,
        'changed_fraction': round(
            changed / indexed.size, 4),
        'regions_total': nregions,
        'regions_numbered': len(numbers),
        'regions_require_manual_edit': len(failures),
        'numbered_fraction': round(
            len(numbers) / nregions, 4),
        'outline_segments': nsegments,
        'by_color': by_color,
        'sample_unnumbered_areas': failures[:100],
        'status': (
            'REVIEW_ONLY' if failures
            else 'AUTO_NUMBERED_NEEDS_VISUAL_APPROVAL'),
        'note': (
            'Оригинал не перерисован. Это автоматическая '
            'схема для проверки, не финальный PDF.')}
    (dest / 'audit.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding='utf-8')
    # Fast raster PROOF; the SVG has print-exact 0.2mm vectors.
    preview_width, preview_height = 2400, 3600
    scale = preview_width / PAGE_W_MM
    art_w = round(drawing_w * scale)
    art_h = round(drawing_h * scale)
    x0 = round(offset_x * scale)
    y0 = round(offset_y * scale)
    scaled = cv2.resize(
        simplified, (art_w, art_h),
        interpolation=cv2.INTER_NEAREST)
    grid = np.zeros((art_h, art_w), dtype=bool)
    grid[:, 1:] |= (
        scaled[:, 1:] != scaled[:, :-1])
    grid[1:, :] |= (
        scaled[1:, :] != scaled[:-1, :])
    preview = np.full(
        (preview_height, preview_width), 255,
        dtype=np.uint8)
    preview[y0:y0+art_h, x0:x0+art_w][grid] = 0
    canvas = Image.fromarray(preview).convert('RGB')
    draw = ImageDraw.Draw(canvas)
    font_px = max(1, round(FONT_MM * scale))
    try:
        font = ImageFont.truetype(
            'DejaVuSans.ttf', font_px)
    except OSError:
        font = ImageFont.load_default()
    for value, mx, my in numbers:
        draw.text((mx * scale, my * scale),
                  str(value), anchor='mm',
                  font=font, fill=(0, 0, 0))
    canvas.save(
        dest / 'lineart_preview.png', optimize=True)
    print(json.dumps({
        key: report[key] for key in (
            'source_distinct_colors',
            'regions_total', 'regions_numbered',
            'regions_require_manual_edit',
            'changed_fraction', 'status')},
        ensure_ascii=False))
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--input', required=True, type=Path)
    ap.add_argument(
        '--out', type=Path, default=Path('outputs'))
    ap.add_argument(
        '--majority', type=int, default=3,
        choices=(1, 3, 5))
    ap.add_argument(
        '--min-area', type=int, default=30)
    ap.add_argument(
        '--rounds', type=int, default=4)
    args = ap.parse_args()
    build(args.input, args.out,
          args.majority, args.min_area,
          args.rounds)
