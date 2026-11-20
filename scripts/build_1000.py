#!/usr/bin/env python3
"""Art-directed 1000-region redraw. Exactly one visible >=5.3pt number per region.
Only the derived segmentation changes; the uploaded 19-color original is untouched.
"""
import argparse
import csv
import hashlib
import heapq
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import cairosvg
import build as previous

TARGET = 1000
WIDTH_MM, HEIGHT_MM = 400.0, 600.0
FONT_PT = 5.3
FONT_MM = FONT_PT * 25.4 / 72
STROKE_MM = 0.2


def edges_and_contacts(ids):
    edges = []
    for a, b in ((ids[:, 1:], ids[:, :-1]), (ids[1:, :], ids[:-1, :])):
        diff = a != b
        lo, hi = a[diff], b[diff]
        edges.append(np.stack((np.minimum(lo, hi), np.maximum(lo, hi)), axis=1))
    pairs, lengths = np.unique(np.concatenate(edges), axis=0, return_counts=True)
    graph = defaultdict(list)
    for (a, b), amount in zip(pairs, lengths):
        graph[int(a)].append((int(b), int(amount)))
        graph[int(b)].append((int(a), int(amount)))
    return graph


def region_measure(ids, number_by_id, px_to_mm):
    """Clearance and candidate locations for actual typographic footprints."""
    n = int(ids.max())
    border = np.zeros(ids.shape, dtype=bool)
    dif = ids[:, 1:] != ids[:, :-1]
    border[:, :-1] |= dif
    border[:, 1:] |= dif
    dif = ids[1:, :] != ids[:-1, :]
    border[:-1, :] |= dif
    border[1:, :] |= dif
    border[:, 0] = True
    border[:, -1] = True
    border[0, :] = True
    border[-1, :] = True
    clearance = cv2.distanceTransform((~border).astype(np.uint8), cv2.DIST_L2, 5)
    ind = np.arange(1, n+1)
    maxes = ndi.maximum(clearance, labels=ids, index=ind)
    where = ndi.maximum_position(clearance, labels=ids, index=ind)
    area = np.bincount(ids.ravel(), minlength=n+1)
    valid = np.zeros(n+1, dtype=bool)
    coords = np.zeros((n+1, 2), dtype=np.int32)
    dimensions = {}
    for region in ind:
        y, x = map(int, where[region-1])
        coords[region] = (x, y)
        number = str(number_by_id[region])
        # At 400-mm width a 5.3pt DejaVu Sans glyph takes approx.
        # 3.3 px per digit; the added 1.2px is a line clearance margin.
        halfwidth = int(np.ceil(
            (len(number) * FONT_MM / px_to_mm * .66 + 1.3) / 2))
        halfheight = int(np.ceil((FONT_MM / px_to_mm + 1.0) / 2))
        dimensions[region] = (halfwidth, halfheight)
        if x-halfwidth < 0 or x+halfwidth >= ids.shape[1] or \
           y-halfheight < 0 or y+halfheight >= ids.shape[0]:
            continue
        patch = ids[y-halfheight:y+halfheight+1,
                    x-halfwidth:x+halfwidth+1]
        valid[region] = bool(np.all(patch == region))
    return valid, area, coords, maxes, dimensions


def group_colors(ids, source_idx, ncolors):
    """Original palette ID for each connected region."""
    n = int(ids.max())
    first = np.full(n+1, -1, dtype=np.int32)
    # source color ID at first occurrence of this region
    unique, at = np.unique(ids.ravel(), return_index=True)
    first[unique] = source_idx.ravel()[at]
    return first


def consolidate(ids, color_of, palette, target, px_to_mm):
    initial = int(ids.max())
    if initial < target:
        raise RuntimeError(f"Already only {initial} regions, cannot choose {target}")
    numbering = previous.palette_indices(palette)
    nums = {c: i+1 for i,c in enumerate(numbering)}
    number_by_id = {rid: nums[int(color_of[rid])] for rid in range(1, initial+1)}
    valid, areas, centers, clearance, dims = region_measure(
        ids, number_by_id, px_to_mm)
    graph = edges_and_contacts(ids)
    positions = centers.astype(float)
    h,w = ids.shape
    labs = cv2.cvtColor(palette[np.newaxis, ...], cv2.COLOR_RGB2LAB)[0].astype(float)
    # Which color islands to keep? First discard narrow scribbles and
    # distant speckle while keeping iris/sakura and paper lantern geometry.
    cand = []
    for region in range(1, initial+1):
        x,y = positions[region]
        salience = 0.0
        if (x-580)**2/52**2 + (y-401)**2/46**2 < 1 or \
           (x-643)**2/31**2 + (y-386)**2/40**2 < 1:
            salience += 0.95
        if (x-361)**2/60**2 + (y-390)**2/58**2 < 1:
            salience += 0.55
        if (x-542)**2/90**2 + (y-660)**2/85**2 < 1:
            salience += 0.35
        if y < 240 and 140 < x < 1070:
            salience += 0.10  # key skyline details
        # Pixels that already fit a label have priority for retention.
        score = (2.5 if valid[region] else 0.0) + \
                min(1.0, clearance[region-1]/8) + \
                min(1.0, np.log1p(areas[region])/9) + salience
        cand.append((score, region))
    cand.sort()
    count_by_color = np.bincount(color_of[1:].astype(int), minlength=len(palette))
    discard = set()
    for _, region in cand:
        if len(discard) >= initial-target:
            break
        color = int(color_of[region])
        if count_by_color[color] <= 1:
            continue
        discard.add(region)
        count_by_color[color] -= 1
    if len(discard) != initial-target:
        raise RuntimeError("Cannot keep all 19 pigments while reaching 1000 regions")
    retained = sorted(set(range(1, initial+1)) - discard)
    retained_set = set(retained)
    # Multi-source boundary growth guarantees connected regions after merge.
    priority = []
    owner = {}
    score = {}
    for seed in retained:
        owner[seed] = seed
        score[seed] = 0.0
        heapq.heappush(priority, (0.0, seed, seed))
    while priority:
        cost, node, seed = heapq.heappop(priority)
        if cost > score.get(node, float('inf')) + 1e-9:
            continue
        for neigh, shared in graph[node]:
            if neigh in retained_set:
                continue
            ci, cj = int(color_of[node]), int(color_of[neigh])
            delta = float(np.linalg.norm(labs[ci]-labs[cj]))
            step = 0.45 + delta/70 + 3.0/(shared+3)
            trial = cost + step
            if trial < score.get(neigh, float('inf')):
                score[neigh] = trial
                owner[neigh] = seed
                heapq.heappush(priority, (trial, neigh, seed))
    if len(owner) != initial:
        raise RuntimeError("Connectivity graph does not reach every paint area")
    translate = np.zeros(initial+1, dtype=np.int32)
    new_color = np.zeros(target+1, dtype=np.uint8)
    for new_id, original_id in enumerate(retained, start=1):
        new_color[new_id] = color_of[original_id]
    renumber = {old: new for new, old in enumerate(retained, start=1)}
    for region in range(1, initial+1):
        translate[region] = renumber[owner[region]]
    return translate[ids], new_color


def repair_pockets(ids, color_of, palette, px_to_mm, max_passes=6):
    """Artistically widen remaining unnumberable color islands for 5.3pt labels.

    Every retained area gets a softly curved pocket, not a tiny printed digit.
    Adjacent regions are kept connected; disconnected fragments are folded
    into their adjacent dominant region before final audit.
    """
    n = int(ids.max())
    ordering = previous.palette_indices(palette)
    numbers = {c: i+1 for i,c in enumerate(ordering)}
    num_by_id = {i: numbers[int(color_of[i])] for i in range(1, n+1)}
    for attempt in range(max_passes):
        good, area, centers, maxima, dims = region_measure(
            ids, num_by_id, px_to_mm)
        missing = np.flatnonzero(~good[1:]) + 1
        if not len(missing):
            break
        # Start from most fragile regions, so no small paint area vanishes.
        missing = sorted(missing, key=lambda i: (area[i], maxima[i-1]))
        protected = np.zeros(ids.shape, dtype=np.int32)
        for r in range(1, n+1):
            if good[r]:
                x,y = centers[r]
                rx,ry = dims[r]
                protected[max(0,y-ry):y+ry+1,
                          max(0,x-rx):x+rx+1] = r
        before = ids.copy()
        for r in missing:
            x,y = map(int, centers[r])
            rx,ry = dims[r]
            pad_x,pad_y = rx+2, ry+2
            if (x-pad_x < 0 or y-pad_y < 0 or
                x+pad_x >= ids.shape[1] or y+pad_y >= ids.shape[0]):
                # Re-center a border-touching island toward the page interior.
                x = min(ids.shape[1]-pad_x-1, max(pad_x, x))
                y = min(ids.shape[0]-pad_y-1, max(pad_y, y))
            patch = ids[y-pad_y:y+pad_y+1, x-pad_x:x+pad_x+1]
            guard = protected[y-pad_y:y+pad_y+1, x-pad_x:x+pad_x+1]
            yy,xx = np.ogrid[-pad_y:pad_y+1,-pad_x:pad_x+1]
            ell = ((xx/pad_x)**2+(yy/pad_y)**2) <= 1
            touching = (guard != 0) & (guard != r) & ell
            if np.any(touching):
                # Don't overwrite a neighbor's already legible number.
                continue
            patch[ell] = r
            protected[y-ry:y+ry+1,x-rx:x+rx+1] = r
        if np.array_equal(before, ids):
            break
    return ids


def create_final(source, out):
    source_bytes = source.read_bytes()
    src = np.asarray(Image.open(source).convert("RGB"))
    palette, inverted = np.unique(src.reshape(-1, 3), axis=0, return_inverse=True)
    if len(palette) != 19:
        raise RuntimeError(f"Expected exactly 19 original colors, got {len(palette)}")
    source_idx = inverted.reshape(src.shape[:2]).astype(np.uint8)
    derived_png = np.asarray(
        Image.open(out / "color_segmentation_review.png").convert("RGB"))
    derived_colors = np.zeros(source_idx.shape, dtype=np.uint8)
    for color, rgb in enumerate(palette):
        derived_colors[np.all(derived_png == rgb, axis=2)] = color
    base_ids = np.asarray(Image.open(out / "region_ids_16bit.png")).astype(np.int32)
    if np.any(base_ids == 0):
        raise AssertionError("Baseline regions contain unassigned pixels")
    h,w = base_ids.shape
    px_to_mm = min(WIDTH_MM/w, HEIGHT_MM/h)
    mm_w,mm_h = w*px_to_mm, h*px_to_mm
    xoff,yoff = (WIDTH_MM-mm_w)/2, (HEIGHT_MM-mm_h)/2
    colors = group_colors(base_ids, derived_colors, len(palette))
    ids, pigments = consolidate(base_ids, colors, palette, TARGET, px_to_mm)
    ids = repair_pockets(ids, pigments, palette, px_to_mm)
    ordering = previous.palette_indices(palette)
    palette_no = {c:i+1 for i,c in enumerate(ordering)}
    numbers = {r:palette_no[int(pigments[r])] for r in range(1,TARGET+1)}
    legible, areas, centers, radii, dims = region_measure(
        ids, numbers, px_to_mm)
    actual_ids = np.unique(ids)
    # Each individual label ID must be connected. No "hidden" islands.
    disconnected = []
    for r in actual_ids:
        n,_ = cv2.connectedComponents((ids == r).astype(np.uint8),8)
        if n != 2:
            disconnected.append([int(r),n-1])
    missing = [int(x) for x in range(1,TARGET+1) if not legible[x]]
    coverage = int(np.count_nonzero(ids == 0))
    used_colors = int(len(np.unique(pigments[ids])))
    report = {
        "source_sha256":hashlib.sha256(source_bytes).hexdigest(),
        "rendered_page_mm":[WIDTH_MM,HEIGHT_MM],
        "artwork_size_mm":[round(mm_w,3),round(mm_h,3)],
        "source_distinct_colors":int(len(palette)),
        "contour_width_mm":STROKE_MM,
        "font_size_pt":FONT_PT,
        "regions_total":len(actual_ids),
        "regions_numbered":TARGET-len(missing),
        "regions_require_manual_edit":len(missing),
        "source_not_overwritten": True,
        "source_colors": int(len(palette)),
        "colors_in_derived_image":used_colors,
        "target_regions":TARGET,
        "actual_region_ids":len(actual_ids),
        "regions_with_legible_number":TARGET-len(missing),
        "regions_without_legible_number":len(missing),
        "unlabelled_region_ids":missing[:100],
        "disconnected_region_ids":disconnected[:100],
        "pixels_without_assigned_region":coverage,
        "font_pt":FONT_PT,
        "outline_mm":STROKE_MM,
        "page_mm":[WIDTH_MM,HEIGHT_MM],
        "artwork_mm":[round(mm_w,3),round(mm_h,3)],
        "status":"PASS" if not missing and not disconnected and
            not coverage and used_colors==19 and
            len(actual_ids)==TARGET else "NEEDS_ARTISTIC_REVIEW",
    }
    (out/"audit_1000.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    # Only publish actual finished scheme if all regions are connected and labelled.
    if report["status"] != "PASS":
        raise RuntimeError(json.dumps(report,ensure_ascii=False))
    path,nsegments = previous.trace_smooth_shared_interfaces(
        ids,px_to_mm,xoff,yoff)
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="400mm" height="600mm" viewBox="0 0 400 600">\n'
        f'<rect width="400" height="600" fill="white"/>\n'
        f'<g stroke="#161616" fill="none" stroke-width="0.2" '
        f'stroke-linejoin="round" stroke-linecap="round">\n'
        f'<rect x="{xoff:.3f}" y="{yoff:.3f}" '
        f'width="{mm_w:.3f}" height="{mm_h:.3f}"/>\n'
        f'<path d="{path}"/></g>\n'
        f'<g font-family="DejaVu Sans" font-size="{FONT_MM:.5f}" '
        f'fill="#111" text-anchor="middle" dominant-baseline="central">\n'
    )
    entries = [
        f'<text x="{xoff+(centers[r,0]+.5)*px_to_mm:.3f}" '
        f'y="{yoff+(centers[r,1]+.5)*px_to_mm:.3f}">{numbers[r]}</text>\n'
        for r in range(1,TARGET+1)
    ]
    (out/"lineart_review.svg").write_text(
        head+"".join(entries)+"</g></svg>",encoding="utf-8")
    Image.fromarray(palette[pigments[ids]]).save(
        out/"color_segmentation_review.png",optimize=True)
    Image.fromarray(ids.astype(np.uint16)).save(out/"region_ids_16bit.png")
    with (out/"region_color_numbers.csv").open(
            "w",newline="",encoding="utf-8-sig") as f:
        writer=csv.writer(f)
        writer.writerow(["region_id","color_number","HEX","area_px"])
        for r in range(1,TARGET+1):
            c=int(pigments[r])
            writer.writerow([r,numbers[r],
                "#"+"".join(f"{x:02X}" for x in palette[c]),int(areas[r])])
    with (out/"palette.csv").open(
            "w",newline="",encoding="utf-8-sig") as f:
        writer=csv.writer(f)
        writer.writerow(["Номер","HEX","R","G","B"])
        for c in ordering:
            R,G,B=map(int,palette[c])
            writer.writerow([palette_no[c],f"#{R:02X}{G:02X}{B:02X}",R,G,B])
    cairosvg.svg2png(
        url=str(out/"lineart_review.svg"),
        write_to=str(out/"lineart_preview.png"),
        output_width=2400,output_height=3600)
    (out/"audit.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    create_final(args.source,args.out)
