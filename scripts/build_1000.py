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
        # The face, eyes and hair ornament are illustration landmarks.
        # Never inflate tiny unpaintable facial flecks into artificial dots:
        # merge them with the closest matching parent color instead.
        on_face_or_hair = (
            ((x-552)/170)**2 + ((y-394)/172)**2 < 1.0 or
            ((x-357)/75)**2 + ((y-387)/70)**2 < 1.0
        )
        if on_face_or_hair:
            salience += 1.5 if valid[region] else -20.0
        on_kimono_flower = (
            ((x-542)/105)**2 + ((y-660)/100)**2 < 1.0 or
            ((x-385)/88)**2 + ((y-851)/112)**2 < 1.0
        )
        if on_kimono_flower:
            salience += 0.5 if valid[region] else -7.0
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


def repair_pockets(ids, color_of, palette, px_to_mm):
    """Reserve a non-overlapping 5.3pt label footprint for ALL 1000 areas.

    Reconstruct a compact smooth label pocket *before* editing neighbors.
    A region's reserved footprint can never be erased by a later edit.
    """
    n = int(ids.max())
    original = ids.copy()
    ordering = previous.palette_indices(palette)
    numbers = {c: i+1 for i,c in enumerate(ordering)}
    num_by_id = {i: numbers[int(color_of[i])] for i in range(1,n+1)}
    already_legible, area, positions, radii, dimensions = region_measure(
        ids, num_by_id, px_to_mm)
    reserved = np.zeros(ids.shape, dtype=bool)
    centers = np.zeros((n+1,2), dtype=np.int32)
    dims = {int(k):v for k,v in dimensions.items()}
    h,w = ids.shape

    def candidate_positions(x,y):
        yield int(x),int(y)
        # Displacement is restricted to the neighbourhood of the existing
        # paint region; curving its edge creates space for a readable number.
        for radius in (6,12,19,27,37,49,64,82,108,142,178):
            for angle in range(16):
                theta = 2*np.pi*angle/16
                yield int(round(x+radius*np.cos(theta))),\
                      int(round(y+radius*np.sin(theta)))

    # Thin and tiny components get their label space before large easy ones.
    ordering_ids = sorted(range(1,n+1),
                          key=lambda r: (not already_legible[r], area[r], radii[r-1]))
    for region in ordering_ids:
        x0,y0 = map(int,positions[region])
        rx,ry = dims[region]
        if already_legible[region]:
            # This location already fits the real-size digit in the original
            # area. Do NOT move a valid label without changing its contour.
            cx,cy = x0,y0
            if reserved[cy-ry:cy+ry+1,cx-rx:cx+rx+1].any():
                raise RuntimeError(
                    f"Previously legible slots collide at {region}")
            centers[region] = (cx,cy)
            reserved[cy-ry:cy+ry+1,cx-rx:cx+rx+1] = True
            continue
        pad_x,pad_y = rx+2,ry+2
        picked = None
        best_score = -1e12
        for cx,cy in candidate_positions(x0,y0):
            if (cx-pad_x < 0 or cy-pad_y < 0 or
                cx+pad_x >= w or cy+pad_y >= h):
                continue
            occupied = reserved[cy-pad_y:cy+pad_y+1,
                                cx-pad_x:cx+pad_x+1]
            if occupied.any():
                continue
            patch = original[cy-pad_y:cy+pad_y+1,
                             cx-pad_x:cx+pad_x+1]
            overlap = np.count_nonzero(patch == region)
            shift = np.hypot(cx-x0,cy-y0)
            # The slot must meaningfully connect to the existing area
            # wherever possible (not a detached floating number bubble).
            score = overlap*4 - shift*.55
            if patch[pad_y,pad_x] == region:
                score += 45
            if score > best_score:
                best_score = score
                picked = (cx,cy)
                if np.all(patch == region):
                    break
        if picked is None:
            raise RuntimeError(
                f"No space for a nonoverlapping label slot in region {region}")
        cx,cy = picked
        centers[region] = (cx,cy)
        reserved[cy-pad_y:cy+pad_y+1,
                 cx-pad_x:cx+pad_x+1] = True

    # All slots are disjoint. Only now alter the geometry of painted regions.
    for r in range(1,n+1):
        if already_legible[r]:
            continue   # Keep the original contour; only redesign narrow shapes.
        cx,cy=map(int,centers[r])
        rx,ry=dims[r]
        pad_x,pad_y=rx+2,ry+2
        yy,xx=np.ogrid[-pad_y:pad_y+1,-pad_x:pad_x+1]
        ellipse=(xx/(rx+2.5))**2+(yy/(ry+2.5))**2 <= 1
        # The glyph footprint is a rectangle; corners must also remain
        # inside the same paintable area after smoothing.
        footprint=(abs(xx)<=rx)&(abs(yy)<=ry)
        brush=ellipse|footprint
        subsection=ids[cy-pad_y:cy+pad_y+1,cx-pad_x:cx+pad_x+1]
        subsection[brush]=r

    # A label pocket can cross another region's thin bridge. Reassign
    # resulting *orphan* fragments, retaining the component with its
    # own number position. No region may be represented by two islands.
    for pass_index in range(8):
        detached = 0
        boxes=ndi.find_objects(ids)
        for r in range(1,n+1):
            box=boxes[r-1]
            if box is None:
                raise RuntimeError(f"Seeded region {r} disappeared")
            sub=ids[box]
            count, cc=cv2.connectedComponents(
                (sub==r).astype(np.uint8),connectivity=8)
            if count<=2:
                continue
            x,y=map(int,centers[r])
            key=int(cc[y-box[0].start,x-box[1].start])
            if key == 0:
                raise RuntimeError(f"Label seed lost for {r}")
            for component in range(1,count):
                if component==key:
                    continue
                orphan=(cc==component)
                outline=cv2.dilate(orphan.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),
                    iterations=1).astype(bool)
                neighbors=sub[outline & (~orphan)]
                neighbors=neighbors[(neighbors!=r)&(neighbors!=0)]
                if len(neighbors)==0:
                    continue
                ids_unique, frequencies=np.unique(
                    neighbors,return_counts=True)
                pigment=int(color_of[r])
                # Prefer a long shared boundary, then a nearby original hue.
                colors=np.asarray([
                    color_of[int(z)] for z in ids_unique],dtype=int)
                rgb=np.asarray(palette).astype(float)
                distances=np.linalg.norm(rgb[colors]-rgb[pigment],axis=1)
                weight=frequencies/(1+distances/80)
                chosen=int(ids_unique[np.argmax(weight)])
                sub[orphan]=chosen
                detached+=1
        if detached == 0:
            break

    return ids, centers, dims


def check_number_footprints(ids,centers,dims):
    """A number is legible only if its entire 5.3pt bounding box fits."""
    passed = np.zeros(int(ids.max())+1,dtype=bool)
    for r in range(1,len(passed)):
        x,y=map(int,centers[r])
        rx,ry=dims[r]
        if x-rx < 0 or y-ry < 0 or \
           x+rx >= ids.shape[1] or y+ry >= ids.shape[0]:
            continue
        patch=ids[y-ry:y+ry+1,x-rx:x+rx+1]
        passed[r]=bool(np.all(patch==r))
    return passed


def restore_eyes_from_reference(indexed):
    """Replace illegible eye micro-islands with two clearly readable irises.

    Only palette ID 16 (#F8A54E) is used; the surrounding facial silhouette,
    hair, nose and garment stay based on the uploaded source artwork.
    Eyelids, eye corners and mouth are printed as open detail strokes,
    not falsely numbered paint areas.
    """
    rebuilt=indexed.copy()
    # Two iris silhouettes in three-quarter view, consistent gaze.
    cv2.ellipse(rebuilt, (581,405), (11,10), -7, 0, 360, 16,-1)
    cv2.ellipse(rebuilt, (647,385), (7,11), 8, 0, 360, 16,-1)
    return rebuilt


def facial_ink_path(px_to_mm,xoff,yoff):
    """Hand-placed non-enclosing 0.2 mm strokes along the ORIGINAL gaze."""
    def p(x,y):
        return f"{xoff+x*px_to_mm:.3f},{yoff+y*px_to_mm:.3f}"
    # Both eyes are anime-like, but the distant eye is more foreshortened.
    lines=[
        f"M{p(551,401)} Q{p(576,379)} {p(602,393)}",
        f"M{p(551,401)} Q{p(580,424)} {p(601,405)}",
        f"M{p(636,387)} Q{p(646,369)} {p(654,380)}",
        f"M{p(640,398)} Q{p(649,402)} {p(655,392)}",
        f"M{p(619,460)} Q{p(626,462)} {p(632,457)}",
    ]
    return " ".join(lines)


def colored_proof_with_ink(image):
    """Preview the PRINTED artistic ink lines without inventing a 20th paint."""
    preview=np.ascontiguousarray(image.copy())
    dark=(24,16,27)
    def quad(a,b,c,n=25):
        t=np.linspace(0,1,n)
        p=(1-t)[:,None]**2*np.asarray(a)+\
          2*((1-t)*t)[:,None]*np.asarray(b)+\
          t[:,None]**2*np.asarray(c)
        return np.round(p).astype(np.int32).reshape(-1,1,2)
    detail=[
        ((551,401),(576,379),(602,393),3),
        ((552,400),(575,420),(601,405),2),
        ((636,387),(646,369),(654,380),3),
        ((639,395),(648,407),(655,394),2),
        ((619,460),(626,462),(632,457),2)
    ]
    for a,b,c,width in detail:
        cv2.polylines(preview,[quad(a,b,c)],False,dark,width,cv2.LINE_AA)
    cv2.ellipse(preview,(582,405),(3,5),0,0,360,dark,-1,cv2.LINE_AA)
    cv2.ellipse(preview,(648,386),(2,4),0,0,360,dark,-1,cv2.LINE_AA)
    return preview


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
    # Artistic revision: keep readable eyes instead of 100 decorative dots.
    derived_colors = restore_eyes_from_reference(derived_colors)
    base_ids, _, _ = previous.label_components(derived_colors,len(palette))
    if np.any(base_ids == 0):
        raise AssertionError("Baseline regions contain unassigned pixels")
    h,w = base_ids.shape
    px_to_mm = min(WIDTH_MM/w, HEIGHT_MM/h)
    mm_w,mm_h = w*px_to_mm, h*px_to_mm
    xoff,yoff = (WIDTH_MM-mm_w)/2, (HEIGHT_MM-mm_h)/2
    colors = group_colors(base_ids, derived_colors, len(palette))
    ids, pigments = consolidate(base_ids, colors, palette, TARGET, px_to_mm)
    ids, seed_centers, footprint_dims = repair_pockets(
        ids, pigments, palette, px_to_mm)
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
    seed_valid=check_number_footprints(
        ids,seed_centers,footprint_dims)
    missing = [int(x) for x in range(1,TARGET+1) if not seed_valid[x]]
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
        f'<path d="{path}"/>'
        f'<path d="{facial_ink_path(px_to_mm,xoff,yoff)}" '
        f'stroke-width="0.2"/>'
        f'<circle cx="{xoff+582*px_to_mm:.3f}" '
        f'cy="{yoff+405*px_to_mm:.3f}" r="0.65" fill="#161616" stroke="none"/>'
        f'<circle cx="{xoff+648*px_to_mm:.3f}" '
        f'cy="{yoff+386*px_to_mm:.3f}" r="0.48" fill="#161616" stroke="none"/>'
        f'</g>\n'
        f'<g font-family="DejaVu Sans" font-size="{FONT_MM:.5f}" '
        f'fill="#111" text-anchor="middle" dominant-baseline="central">\n'
    )
    entries = [
        f'<text x="{xoff+(seed_centers[r,0]+.5)*px_to_mm:.3f}" '
        f'y="{yoff+(seed_centers[r,1]+.5)*px_to_mm:.3f}">{numbers[r]}</text>\n'
        for r in range(1,TARGET+1)
    ]
    (out/"lineart_review.svg").write_text(
        head+"".join(entries)+"</g></svg>",encoding="utf-8")
    colored = palette[pigments[ids]]
    Image.fromarray(colored).save(
        out/"color_segmentation_review.png",optimize=True)
    Image.fromarray(colored_proof_with_ink(colored)).save(
        out/"color_artist_preview.png",optimize=True)
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
