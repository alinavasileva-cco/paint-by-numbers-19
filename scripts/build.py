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



def label_with_centers(indexed, ncolors):
    """Track the centroid of each full-color region (8-neighbor model)."""
    ids = np.zeros(indexed.shape, dtype=np.int32)
    region_colors = [-1]
    sizes = [0]
    centers = [(0.0, 0.0)]
    for color in range(ncolors):
        n, labs, stats, cent = cv2.connectedComponentsWithStats(
            (indexed == color).astype(np.uint8), connectivity=8)
        offset = len(sizes) - 1
        present = labs > 0
        ids[present] = labs[present] + offset
        region_colors.extend([color] * (n - 1))
        sizes.extend(stats[1:, cv2.CC_STAT_AREA].astype(int).tolist())
        centers.extend(cent[1:].tolist())
    return (ids, np.asarray(region_colors, dtype=np.int16),
            np.asarray(sizes), np.asarray(centers))


def simplify_horizon(indexed, ncolors, global_min=40,
                     horizon_min=70, rounds=3):
    """More cleanup along skyline; preserve the girl's face/kimono detail.

    Thresholds are in original-image pixels (1122x1402 reference).
    Each merged pixel is reassigned to one of the SAME input colors.
    """
    work = indexed.copy()
    height, width = work.shape
    for _ in range(rounds):
        ids, colors, sizes, centers = label_with_centers(work, ncolors)
        xs, ys = centers[:, 0], centers[:, 1]
        far_distance = (ys < height * (230 / 1402))
        side_sky = ((ys < height * (380 / 1402)) &
                    ((xs < width * (265 / 1122)) |
                     (xs > width * (790 / 1122))))
        threshold = np.where(
            far_distance | side_sky, horizon_min, global_min)
        tiny = np.flatnonzero(
            (sizes < threshold) & (np.arange(len(sizes)) > 0))
        if not len(tiny):
            break
        edges = []
        for a, b in ((ids[:, :-1], ids[:, 1:]),
                     (ids[:-1, :], ids[1:, :])):
            changed = a != b
            low, high = a[changed], b[changed]
            edges.append(np.stack(
                (np.minimum(low, high), np.maximum(low, high)), axis=1))
        pairs, shared = np.unique(
            np.concatenate(edges), axis=0, return_counts=True)
        contacts = defaultdict(list)
        for (a, b), length in zip(pairs, shared):
            contacts[int(a)].append((int(b), int(length)))
            contacts[int(b)].append((int(a), int(length)))
        assigned = colors.astype(np.uint8).copy()
        for region in tiny:
            adjacent = contacts.get(int(region), ())
            if not adjacent:
                continue
            best, _ = max(
                adjacent,
                key=lambda item: (
                    item[1],
                    sizes[item[0]] >= threshold[item[0]],
                    sizes[item[0]]))
            assigned[region] = colors[best]
        work = assigned[ids]
    return work



def art_directed_1500(base, original, palette):
    """Refine a 2014-region review into ~1500 paintable regions.

    The horizon and far water are simplified more strongly than the girl's
    eyes and the blossom in her hair. All colors are original palette entries.
    """
    work = base.copy()
    h, w = work.shape
    ncolors = len(palette)
    warm = np.asarray([
        int(r) > 140 and int(r) > int(g) * .9
        and int(r) > int(b) * 1.25
        for r, g, b in palette
    ], dtype=bool)
    for _ in range(4):
        ids, colors, sizes, centers = label_with_centers(work, ncolors)
        xs, ys = centers[:, 0], centers[:, 1]
        skyline = (ys < h * (240 / 1402)) | (
            (ys < h * (495 / 1402)) &
            ((xs < w * (270 / 1122)) | (xs > w * (800 / 1122))))
        threshold = np.where(skyline, 336, 143)
        # Retain small luminous windows and individual floating lanterns.
        warm_area = (ys < h * (485 / 1402)) & warm[colors.clip(0)]
        threshold[warm_area] = np.minimum(threshold[warm_area], 18)
        small = np.flatnonzero(
            (sizes < threshold) & (np.arange(len(sizes)) > 0))
        if not len(small):
            break
        edge_pairs = []
        for left, right in (
            (ids[:, :-1], ids[:, 1:]), (ids[:-1], ids[1:])
        ):
            differs = left != right
            aa, bb = left[differs], right[differs]
            edge_pairs.append(np.stack(
                (np.minimum(aa, bb), np.maximum(aa, bb)), axis=1))
        pairs, shared = np.unique(
            np.concatenate(edge_pairs), axis=0, return_counts=True)
        adjacent = defaultdict(list)
        for (a, b), length in zip(pairs, shared):
            adjacent[int(a)].append((int(b), int(length)))
            adjacent[int(b)].append((int(a), int(length)))
        reassigned = colors.astype(np.uint8).copy()
        for region in small:
            neighbors = adjacent.get(int(region))
            if not neighbors:
                continue
            best, _ = max(
                neighbors,
                key=lambda pair: (
                    pair[1],
                    sizes[pair[0]] >= threshold[pair[0]],
                    sizes[pair[0]]))
            reassigned[region] = colors[best]
        work = reassigned[ids]

    # The previous review lost the iris shape. Reintroduce the real
    # detail from the uploaded original, after only a gentle cleanup.
    fine = remove_speckles(
        majority_filter(original, ncolors, 3), ncolors,
        min_area=3, rounds=3)
    protected = np.zeros(work.shape, dtype=np.uint8)
    def protected_ellipse(cx, cy, rx, ry):
        cv2.ellipse(
            protected,
            (round(cx * w / 1122), round(cy * h / 1402)),
            (round(rx * w / 1122), round(ry * h / 1402)),
            0, 0, 360, 255, -1)
    protected_ellipse(580, 401, 38, 33)  # near eye
    protected_ellipse(643, 386, 22, 30)  # far eye
    protected_ellipse(361, 390, 38, 43)  # sakura hair ornament
    work[protected > 0] = fine[protected > 0]
    # Protect the structural petals and slender branches of the kimono.
    flower_detail = remove_speckles(
        majority_filter(original, ncolors, 3), ncolors,
        min_area=8, rounds=3)
    petal_mask = np.zeros(work.shape, dtype=np.uint8)
    cv2.ellipse(petal_mask,
        (round(542 * w / 1122), round(660 * h / 1402)),
        (round(75 * w / 1122), round(80 * h / 1402)),
        0, 0, 360, 255, -1)
    cv2.ellipse(petal_mask,
        (round(387 * w / 1122), round(851 * h / 1402)),
        (round(65 * w / 1122), round(96 * h / 1402)),
        0, 0, 360, 255, -1)
    work[petal_mask > 0] = flower_detail[petal_mask > 0]
    return work


def trace_smooth_shared_interfaces(indexed, px_to_mm, xoff, yoff):
    """Trace color boundaries ONCE as shared vector paths.

    No repeated staircase segments. Junctions are pinned, and 1 px
    contour simplification straightens roofs/lantern edges without
    changing the per-pixel color assignment map.
    """
    height, width = indexed.shape
    stride = width + 1
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
    def walk(first, second):
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
            neighbor = adjacency[b]
            next_node = neighbor[0] if neighbor[0] != a else neighbor[1]
            if (min(b, next_node), max(b, next_node)) in visited:
                break
            a, b = b, next_node
        return chain
    for start, neighbors in adjacency.items():
        if len(neighbors) == 2:
            continue
        for next_node in neighbors:
            if (min(start, next_node), max(start, next_node)) not in visited:
                polylines.append(walk(start, next_node))
    for start, neighbors in adjacency.items():
        for next_node in neighbors:
            if (min(start, next_node), max(start, next_node)) not in visited:
                polylines.append(walk(start, next_node))
    paths = []
    for nodes in polylines:
        if len(nodes) < 2:
            continue
        points = np.asarray(
            [(i % stride, i // stride) for i in nodes],
            dtype=np.float32)
        closed = len(nodes) > 3 and nodes[0] == nodes[-1]
        simplified = cv2.approxPolyDP(
            points.reshape(-1, 1, 2), 1.0, closed).reshape(-1, 2)
        if len(simplified) < 2:
            continue
        points_mm = [
            (xoff + float(x) * px_to_mm, yoff + float(y) * px_to_mm)
            for x, y in simplified]
        path = f'M{points_mm[0][0]:.3f},{points_mm[0][1]:.3f}'
        path += ''.join(
            f'L{x:.3f},{y:.3f}' for x, y in points_mm[1:])
        if closed:
            path += 'Z'
        paths.append(path)
    return ' '.join(paths), len(paths)


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
    simplified = simplify_horizon(simplified, len(palette))
    simplified = art_directed_1500(simplified, indexed, palette)
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
    path, nsegments = trace_smooth_shared_interfaces(
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
    # Exact pixel-level assignment map: ZERO unassigned pixels.
    # This is also a machine-readable reference when a 5.3pt number
    # cannot physically fit inside a narrow print area.
    region_ids, region_colors, region_areas = label_components(
        simplified, len(palette))
    uncovered_pixels = int(np.count_nonzero(region_ids == 0))
    if uncovered_pixels:
        raise AssertionError(
            f'{uncovered_pixels} pixels have no region/color assignment')
    if len(region_colors) > 65536:
        raise ValueError('Too many regions for 16-bit region-id map')
    Image.fromarray(region_ids.astype(np.uint16)).save(
        dest / 'region_ids_16bit.png')
    with (dest / 'region_color_numbers.csv').open(
            'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'region_id', 'color_number', 'hex', 'pixel_area'])
        for rid in range(1, len(region_colors)):
            original_color_id = int(region_colors[rid])
            r, g, b = map(int, palette[original_color_id])
            writer.writerow([
                rid, color_to_number[original_color_id],
                f'#{r:02X}{g:02X}{b:02X}', int(region_areas[rid])])

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
        'horizon_simplification': {
            'global_min_area_px': 40,
            'horizon_min_area_px': 70,
            'rounds': 3,
            'horizon_top_y_at_reference': 230,
            'side_sky_to_y_at_reference': 380
        },
        'every_pixel_assigned_to_one_of_19_colors': uncovered_pixels == 0,
        'pixels_without_color_assignment': uncovered_pixels,
        'art_directed_pass': {
            'goal_regions': 1500,
            'horizon_threshold_px': 336,
            'global_threshold_px': 143,
            'rounds': 4,
            'protected_landmarks': ['near_eye', 'far_eye', 'sakura_hairpin', 'kimono_shoulder', 'kimono_sleeve'],
            'small_warm_lights_preserved': True,
            'line_style': 'shared_contour_vector_simplification_1px'
        },
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
    # Raster preview is rendered FROM the vector so both look identical.
    import cairosvg
    cairosvg.svg2png(
        url=str(dest / 'lineart_review.svg'),
        write_to=str(dest / 'lineart_preview.png'),
        output_width=2400, output_height=3600)
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
