#!/usr/bin/env python3
"""Rework the confirmed 1501-region version toward 1300 labelled regions.
Preserve the source-based foreground; preferentially remove background flecks.
No generated illustration or synthetic cartoon eyes are used.
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
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import cairosvg
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build as previous

TARGET = 1300
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
    # The target requires eliminating only 201 of the original 1501 cells.
    # Prioritize distant skyline and water flecks, not figure/clothing details.
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
        if y < 540 and ((y < 300) or not (275 < x < 690)):
            salience -= 18.0  # distant skyline, windows, water / lanterns
        if y > 550:
            salience += 12.0  # foreground water, kimono and close lanterns
        if 270 < x < 695 and 195 < y < 1040:
            salience += 20.0  # girl / clothing protected as first priority
        # The face, eyes and hair ornament are illustration landmarks.
        # Never inflate tiny unpaintable facial flecks into artificial dots:
        # merge them with the closest matching parent color instead.
        on_face_or_hair = (
            ((x-552)/170)**2 + ((y-394)/172)**2 < 1.0 or
            ((x-357)/75)**2 + ((y-387)/70)**2 < 1.0
        )
        if on_face_or_hair:
            salience += 24.0 if valid[region] else 9.0
        on_kimono_flower = (
            ((x-542)/105)**2 + ((y-660)/100)**2 < 1.0 or
            ((x-385)/88)**2 + ((y-851)/112)**2 < 1.0
        )
        if on_kimono_flower:
            salience += 18.0 if valid[region] else 9.0
        # Pixels that already fit a label have priority for retention.
        score = (2.5 if valid[region] else 0.0) + \
                min(1.0, clearance[region-1]/8) + \
                min(1.0, np.log1p(areas[region])/9) + salience
        # A 1px fleck must never survive as a numbered paint zone.
        # Remove tiny fragments wherever they occur, including the face;
        # keep the larger original silhouettes rather than inflating dots.
        if areas[region] < 20:
            score -= 100.0 - float(areas[region])
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
        raise RuntimeError("Cannot keep all 19 pigments while reaching 1300 regions")
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



def _number_stencil(number, px_to_mm):
    """Rasterize the ACTUAL 5.3pt letterforms and a one-pixel safety halo.

    A stencil follows printed ink rather than inventing a large circular
    paint island. SVG uses the exact 5.3pt font size at export.
    """
    pixel_size=max(6,int(np.ceil(FONT_MM/px_to_mm)))
    font=ImageFont.truetype("DejaVuSans.ttf",pixel_size)
    bbox=font.getbbox(str(number))
    sw,sh=bbox[2]-bbox[0],bbox[3]-bbox[1]
    canvas=Image.new("L",(sw+4,sh+4),0)
    dr=ImageDraw.Draw(canvas)
    dr.text((2-bbox[0],2-bbox[1]),str(number),font=font,fill=255)
    ink=(np.asarray(canvas)>40).astype(np.uint8)
    ink=cv2.dilate(ink,np.ones((3,3),np.uint8),iterations=1).astype(bool)
    return ink


def repair_pockets(ids, color_of, palette, px_to_mm):
    """Fit real-sized printed numerals without creating circular label patches.

    Reserves label positions before any pixel reassignment; gives priority
    to nearby original-colored pixels and avoids protected drawing areas.
    """
    source_ids=ids.copy()
    n=int(ids.max())
    order=previous.palette_indices(palette)
    numbers={c:i+1 for i,c in enumerate(order)}
    by_region={r:numbers[int(color_of[r])] for r in range(1,n+1)}
    initial,areas,positions,_,_=region_measure(ids,by_region,px_to_mm)
    stencils={r:_number_stencil(by_region[r],px_to_mm) for r in range(1,n+1)}
    reserved=np.zeros(ids.shape,bool)
    centers=np.zeros((n+1,2),np.int32)
    orientations=np.zeros(n+1,np.int16)
    h,w=ids.shape
    ys,xs=np.ogrid[:h,:w]
    protect=(((xs-494)/252)**2+((ys-620)/455)**2<=1)
    protect|=((xs>535)&(xs<675)&(ys>355)&(ys<485))

    def at(r,cx,cy):
        glyph=stencils[r]
        hh,ww=glyph.shape
        x0=cx-ww//2;y0=cy-hh//2
        if x0<2 or y0<2 or x0+ww>=w-2 or y0+hh>=h-2:
            return None
        return x0,y0,glyph

    def samples(x,y):
        yield x,y
        for radius in (5,9,14,21,29,41,56,78,105):
            for j in range(20):
                ang=j*np.pi/10
                yield round(x+radius*np.cos(ang)),round(y+radius*np.sin(ang))

    region_boxes=ndi.find_objects(source_ids)
    for r in sorted(range(1,n+1),key=lambda z:(initial[z],areas[z])):
        x,y=map(int,positions[r])
        candidate=None
        best=-1e50
        def candidates():
            yield from samples(x,y)
            if areas[r] < 2500 and region_boxes[r-1] is not None:
                box=region_boxes[r-1]
                ry,rx=np.nonzero(source_ids[box]==r)
                for yi,xi in zip(ry,rx):
                    yield int(xi+box[1].start),int(yi+box[0].start)
        original_glyph=stencils[r].copy()
        chosen_angle=0
        for angle,quarters in ((0,0),(-90,1),(90,3)):
            stencils[r]=np.rot90(original_glyph,quarters)
            for cx,cy in candidates():
                v=at(r,cx,cy)
                if v is None: continue
                x0,y0,glyph=v
                hh,ww=glyph.shape
                roi=source_ids[y0:y0+hh,x0:x0+ww]
                if not np.any(roi==r):continue
                rr=reserved[y0:y0+hh,x0:x0+ww]
                # Separate the real numeral strokes, not whole empty text boxes.
                if np.any(rr[glyph]):continue
                part=roi[glyph]
                original_count=np.count_nonzero(part==r)
                protected_changes=int(np.count_nonzero(
                    (part!=r)&protect[y0:y0+hh,x0:x0+ww][glyph]))
                required_changes=int(np.count_nonzero(part!=r))
                moved=np.hypot(cx-x,cy-y)
                score=original_count*8-required_changes*3-protected_changes*8-moved*.18
                if roi[hh//2,ww//2]==r:score+=15
                if score>best:
                    best=score
                    candidate=(cx,cy,x0,y0,glyph)
                if required_changes==0:
                    break
            if candidate is not None:
                chosen_angle=angle
                break
        if candidate is None:
            raise RuntimeError(f"No viable label stencil for cell {r}, area={areas[r]}, center={x},{y}")
        cx,cy,x0,y0,glyph=candidate
        centers[r]=(cx,cy)
        orientations[r]=chosen_angle
        hh,ww=glyph.shape
        reserved[y0:y0+hh,x0:x0+ww] |= glyph

    # Reassign ONLY actual glyph/halo pixels, not their rectangular bounding
    # boxes or an enlarged ellipse around every small source fragment.
    for r in range(1,n+1):
        cx,cy=map(int,centers[r])
        glyph=stencils[r]
        hh,ww=glyph.shape
        x0=cx-ww//2;y0=cy-hh//2
        patch=ids[y0:y0+hh,x0:x0+ww]
        patch[glyph]=r

    # Large original art fragments cannot silently be recolored when a
    # small ink halo grazes a narrow bridge. Only tiny orphan fragments
    # can be merged; larger topology errors cause candidate rejection.
    for _ in range(4):
        detached=0
        boxes=ndi.find_objects(ids)
        for r in range(1,n+1):
            box=boxes[r-1]
            if box is None:continue
            sub=ids[box]
            count,labels=cv2.connectedComponents(
                (sub==r).astype(np.uint8),connectivity=8)
            if count<=2:continue
            cx,cy=map(int,centers[r])
            main=int(labels[cy-box[0].start,cx-box[1].start])
            if main==0:continue
            for k in range(1,count):
                if k==main:continue
                orphan=(labels==k)
                size=int(orphan.sum())
                if size>25:continue
                rim=cv2.dilate(orphan.astype(np.uint8),
                              np.ones((3,3),np.uint8),iterations=1).astype(bool)
                other=sub[rim & ~orphan]
                other=other[(other!=r)&(other>0)]
                if not other.size:continue
                vals,nums=np.unique(other,return_counts=True)
                target=int(vals[np.argmax(nums)])
                sub[orphan]=target
                detached+=1
        if not detached:break

    return ids,centers,stencils,orientations


def check_number_footprints(ids,centers,stencils):
    passed=np.zeros(int(ids.max())+1,bool)
    for r in range(1,len(passed)):
        cx,cy=map(int,centers[r])
        glyph=stencils[r]
        hh,ww=glyph.shape
        x0=cx-ww//2;y0=cy-hh//2
        if x0<0 or y0<0 or x0+ww>ids.shape[1] or y0+hh>ids.shape[0]:
            continue
        passed[r]=bool(np.all(ids[y0:y0+hh,x0:x0+ww][glyph]==r))
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
        cv2.polylines(preview,[quad(a,b,c)],False,dark,width,cv2.LINE_8)
    cv2.ellipse(preview,(582,405),(3,5),0,0,360,dark,-1,cv2.LINE_8)
    cv2.ellipse(preview,(648,386),(2,4),0,0,360,dark,-1,cv2.LINE_8)
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
    pixel_has_color = np.zeros(source_idx.shape, dtype=bool)
    for color, rgb in enumerate(palette):
        matching = np.all(derived_png == rgb, axis=2)
        derived_colors[matching] = color
        pixel_has_color |= matching
    # Do not silently interpret an unknown RGB value as palette index zero.
    missing_pigment_pixels = int(np.count_nonzero(~pixel_has_color))
    if missing_pigment_pixels:
        raise RuntimeError(
            f"{missing_pigment_pixels} pixels have no original palette color")
    # Do not redraw the eyes. Retain the existing source-based shapes.
    base_ids, _, _ = previous.label_components(derived_colors,len(palette))
    if np.any(base_ids == 0):
        raise AssertionError("Baseline regions contain unassigned pixels")
    h,w = base_ids.shape
    px_to_mm = min(WIDTH_MM/w, HEIGHT_MM/h)
    mm_w,mm_h = w*px_to_mm, h*px_to_mm
    xoff,yoff = (WIDTH_MM-mm_w)/2, (HEIGHT_MM-mm_h)/2
    colors = group_colors(base_ids, derived_colors, len(palette))
    ids, pigments = consolidate(base_ids, colors, palette, TARGET, px_to_mm)
    merged_ids=ids.copy()
    ids, seed_centers, footprint_dims, orientations = repair_pockets(
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
    # Compare the result to the protected drawing zones of the previous
    # 1501-region version, which is the required artistic reference.
    ys,xs=np.ogrid[:h,:w]
    protected_girl=(((xs-494)/252)**2+((ys-620)/455)**2<=1)
    protected_eyes=((xs>535)&(xs<675)&(ys>355)&(ys<485))
    protected_ink=np.logical_or(protected_girl,protected_eyes)
    protected_delta=np.count_nonzero(
        np.logical_and(pigments[ids]!=derived_colors,protected_ink))
    protected_delta_before_pockets=int(np.count_nonzero(
        np.logical_and(pigments[merged_ids]!=derived_colors,protected_ink)))
    regions_altered_by_pockets=int(np.count_nonzero(
        np.logical_and(merged_ids!=ids,protected_ink)))
    total_protected=int(protected_ink.sum())
    print("ART AUDIT: foreground changed after merge",protected_delta_before_pockets,
          "after number pockets",protected_delta,
          "pixels reassigned by pockets",regions_altered_by_pockets,flush=True)
    report = {
        "source_sha256":hashlib.sha256(source_bytes).hexdigest(),
        "reference_regions":int(base_ids.max()),
        "protected_pixels_total":total_protected,
        "protected_pixels_changed":int(protected_delta),
        "protected_changed_before_number_fit":protected_delta_before_pockets,
        "pocket_reassigned_protected_pixels":regions_altered_by_pockets,
        "protected_changed_fraction":round(protected_delta/max(1,total_protected),6),
        "rendered_page_mm":[WIDTH_MM,HEIGHT_MM],
        "artwork_size_mm":[round(mm_w,3),round(mm_h,3)],
        "source_distinct_colors":int(len(palette)),
        "contour_width_mm":STROKE_MM,
        "font_size_pt":FONT_PT,
        "regions_total":len(actual_ids),
        "regions_numbered":TARGET-len(missing),
        "regions_without_number":len(missing),
        "all_regions_numbered":bool(
            not missing and len(actual_ids)==TARGET and
            (TARGET-len(missing))==len(actual_ids)),
        "regions_require_manual_edit":len(missing),
        "source_not_overwritten": True,
        "source_colors": int(len(palette)),
        "rotated_labels":int(np.count_nonzero(orientations)),
        "colors_in_derived_image":used_colors,
        "target_regions":TARGET,
        "actual_region_ids":len(actual_ids),
        "regions_with_legible_number":TARGET-len(missing),
        "regions_without_legible_number":len(missing),
        "label_fit_passed_regions":int(np.count_nonzero(seed_valid[1:])),
        "label_fit_failed_regions":int(np.count_nonzero(~seed_valid[1:])),
        "unlabelled_region_ids":missing[:100],
        "disconnected_region_ids":disconnected[:100],
        "pixels_without_assigned_region":coverage,
        "pixels_without_color_assignment":missing_pigment_pixels+coverage,
        "font_pt":FONT_PT,
        "outline_mm":STROKE_MM,
        "page_mm":[WIDTH_MM,HEIGHT_MM],
        "artwork_mm":[round(mm_w,3),round(mm_h,3)],
        "status":"PASS" if (
            not missing and not disconnected and not coverage and
            not missing_pigment_pixels and used_colors==19 and
            len(actual_ids)==TARGET and
            (TARGET-len(missing))==len(actual_ids) and
            protected_delta / max(1,total_protected) < 0.03
        ) else "NEEDS_ARTISTIC_REVIEW",
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
        # No synthetic facial linework or added pupil circles.
        f'</g>\n'
        f'<g font-family="DejaVu Sans" font-size="{FONT_MM:.5f}" '
        f'fill="#111" text-anchor="middle" dominant-baseline="central">\n'
    )
    entries = []
    for r in range(1,TARGET+1):
        x=xoff+(seed_centers[r,0]+.5)*px_to_mm
        y=yoff+(seed_centers[r,1]+.5)*px_to_mm
        angle=int(orientations[r])
        rotate=f' transform="rotate({angle} {x:.3f} {y:.3f})"' if angle else ''
        entries.append(
            f'<text x="{x:.3f}" y="{y:.3f}"{rotate}>{numbers[r]}</text>\\n')
    (out/"lineart_review.svg").write_text(
        head+"".join(entries)+"</g></svg>",encoding="utf-8")
    colored = palette[pigments[ids]]
    Image.fromarray(colored).save(
        out/"color_segmentation_review.png",optimize=True)
    artist_image = colored.copy()
    artist_color_count = len(np.unique(artist_image.reshape(-1,3),axis=0))
    if artist_color_count > 19:
        raise AssertionError(f"Artist preview has {artist_color_count} colors (maximum 19)")
    Image.fromarray(artist_image).save(
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
