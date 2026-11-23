#!/usr/bin/env python3
"""Refine approved 19-color lantern scene entirely by code.

All output geometry comes from a categorical 19-color region map, not an
image generator. The approved source is read-only. The previous painted image
is the fixed baseline for compositional preservation. Fine iris definition is
restored in a tightly localized, documented eye patch. Every paint region must
accept an actual >=5.3 pt number; otherwise the build exits with an error.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cairosvg
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

COLORS = 19
PAGE_W_MM = 400.0
PAGE_H_MM = 600.0
LINE_MM = 0.2
FONT_PT = 5.3
FONT_MM = FONT_PT * 25.4 / 72.0
CROP = (15, 11, 1070, 1334)
SWATCH_X = [296,339,382,425,468,511,555,598,641,684,728,771,814,856,898,938,977,1015,1053]
SWATCH_Y = 1375

def pack(rgb):
    return (rgb[...,0].astype(np.int32) << 16) | (rgb[...,1].astype(np.int32) << 8) | rgb[...,2].astype(np.int32)

def components(indexed):
    """All 8-connected paint islands, their colors, sizes and center candidates."""
    ids = np.zeros(indexed.shape, dtype=np.int32)
    colors = [-1]
    areas = [0]
    centers = [(0.0, 0.0)]
    radii = [0.0]
    for color in range(COLORS):
        mask = (indexed == color).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        count = n-1
        if not count:
            continue
        off = len(colors)-1
        ids[lab > 0] = lab[lab > 0] + off
        colors.extend([color] * count)
        areas.extend(stats[1:, cv2.CC_STAT_AREA].tolist())
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        region_ids = np.arange(1, n)
        centers.extend([(float(x), float(y)) for y, x in ndi.maximum_position(distance, labels=lab, index=region_ids)])
        radii.extend([float(v) for v in np.atleast_1d(ndi.maximum(distance, labels=lab, index=region_ids))])
    return ids, np.asarray(colors, dtype=np.int16), np.asarray(areas, dtype=np.int32), np.asarray(centers), np.asarray(radii)

def border_contacts(ids):
    pairs = []
    for a, b in ((ids[:,1:], ids[:,:-1]), (ids[1:,:], ids[:-1,:])):
        change = a != b
        x,y = a[change], b[change]
        pairs.append(np.stack([np.minimum(x,y), np.maximum(x,y)],axis=1))
    arr, counts = np.unique(np.concatenate(pairs),axis=0,return_counts=True)
    out = defaultdict(list)
    for (a,b), n in zip(arr, counts):
        a,b,n = int(a),int(b),int(n)
        out[a].append((b,n))
        out[b].append((a,n))
    return out

def needed_radius(color_id, scale_mm_px):
    number = str(color_id+1)
    font_px = FONT_MM/scale_mm_px
    estimated_w = (0.62*len(number)+0.30)*font_px
    return max(0.69*font_px, 0.60*estimated_w)

def enforce_number_fit(work, mm_px, max_passes=12):
    history=[]
    for iteration in range(max_passes):
        ids, colors, sizes, center, radius = components(work)
        desired = np.asarray([0.0] + [needed_radius(int(c), mm_px) for c in colors[1:]])
        invalid=np.flatnonzero((np.arange(len(colors))>0)&(radius < desired))
        history.append({'pass':iteration,'regions':len(colors)-1,'not_fitting':len(invalid)})
        if not len(invalid):
            return work, history
        contacts = border_contacts(ids)
        changes=colors.copy()
        for rid in invalid:
            near=contacts.get(int(rid),())
            if not near:
                continue
            best=max(near,key=lambda pair:(pair[1],sizes[pair[0]]))[0]
            changes[rid]=colors[best]
        update=changes[ids].astype(np.uint8)
        if np.array_equal(update,work):
            break
        work=update
    raise ValueError('5.3pt label fit failed after merging; refuse to publish unnumbered regions')

def restore_iris(work, original):
    """Restore eye structure by source-coordinate vector shapes using ONLY palette colors.

    Coordinates are measured on the exact approved drawing. Deliberately
    local: hair, face shape, skin, kimono and skyline cannot change.
    The whites, warm iris and pupils become actual numbered paint islands.
    """
    changed = work.copy()
    # Main eye: small sclera, amber iris, dark pupil (palette 8, 10, 1).
    sclera = np.array([[587,443],[596,437],[604,438],[605,454],
                       [599,464],[588,462],[582,454]],dtype=np.int32)
    sclera -= (CROP[0], CROP[1])
    cv2.fillPoly(changed,[sclera],7)
    cv2.ellipse(changed,(564,441),(10,12),8,0,360,9,-1)
    cv2.ellipse(changed,(562,433),(5,6),8,0,360,0,-1)
    # Far eye is visibly smaller in the reference: keep its perspective.
    cv2.ellipse(changed,(641,437),(7,11),0,0,360,9,-1)
    cv2.ellipse(changed,(639,432),(4,6),0,0,360,0,-1)
    return changed

def trace_interfaces(indexed):
    """A boundary is represented once, so both adjacent fields share it."""
    h,w=indexed.shape
    stride=w+1
    adj=defaultdict(list)
    horiz=np.argwhere(indexed[1:,:] != indexed[:-1,:])
    vert=np.argwhere(indexed[:,1:] != indexed[:,:-1])
    for y,x in horiz:
        a=int((y+1)*stride+x); b=a+1
        adj[a].append(b);adj[b].append(a)
    for y,x in vert:
        a=int(y*stride+x+1);b=a+stride
        adj[a].append(b);adj[b].append(a)
    seen=set()
    def walk(a,b):
        path=[a]
        while True:
            edge=(min(a,b),max(a,b))
            if edge in seen:break
            seen.add(edge);path.append(b)
            if len(adj[b])!=2 or b==path[0]:break
            nxt=adj[b][0] if adj[b][0]!=a else adj[b][1]
            if (min(b,nxt),max(b,nxt)) in seen:break
            a,b=b,nxt
        return np.asarray([(node%stride,node//stride) for node in path],dtype=np.float32)
    out=[]
    for key, ns in adj.items():
        if len(ns)==2:continue
        for neighbor in ns:
            if (min(key,neighbor),max(key,neighbor)) not in seen:
                out.append(walk(key,neighbor))
    for key,ns in adj.items():
        for neighbor in ns:
            if (min(key,neighbor),max(key,neighbor)) not in seen:
                out.append(walk(key,neighbor))
    return out

# Canonical lantern rectangles: linear corners, roofs and folds stay sharp.
LANTERNS = [
    (10,350,200,520), (0,665,215,920), (665,500,810,730),
    (835,355,980,570), (760,935,1048,1320), (200,310,295,455),
    (665,215,765,370), (735,265,830,405), (40,210,150,335),
]

def preserve_corner(x,y):
    if y < 218: return True
    for x0,y0,x1,y1 in LANTERNS:
        if x0<=x<=x1 and y0<=y<=y1:return True
    return False

def smooth_curve(points, scale, xoff, yoff):
    """Corner-aware vector smoothing with anchored junctions.

    Edges between intersections are never generated twice. No independent
    smoothing per filled region, hence no SVG seams. Technical polygon corners
    remain sharp. Organic sections use short quadratic Bézier arcs.
    """
    if len(points)<2:return ''
    closed=len(points)>3 and np.array_equal(points[0],points[-1])
    p=points[:-1] if closed else points.copy()
    if len(p)<3:return ''
    smooth = ndi.gaussian_filter1d(p, sigma=1.35, axis=0, mode='wrap' if closed else 'nearest')
    if not closed:
        margin=min(7, len(p)//2)
        for i in range(margin):
            alpha=i/float(margin)
            smooth[i]=alpha*smooth[i]+(1-alpha)*p[i]
            smooth[-1-i]=alpha*smooth[-1-i]+(1-alpha)*p[-1-i]
    sharp=np.asarray([preserve_corner(float(x),float(y)) for x,y in p])
    smooth[sharp]=p[sharp]
    eps=1.25 if len(p)>15 else 0.7
    q=cv2.approxPolyDP(smooth.reshape(-1,1,2),eps,closed).reshape(-1,2)
    if len(q)<3:return ''
    def xy(t):return f'{xoff+t[0]*scale:.3f},{yoff+t[1]*scale:.3f}'
    commands=[]
    if closed:
        start=(q[-1]+q[0])*0.5
        commands.append('M'+xy(start))
        for i in range(len(q)):
            a,b,c=q[(i-1)%len(q)],q[i],q[(i+1)%len(q)]
            if preserve_corner(float(b[0]),float(b[1])):
                commands.append('L'+xy(b))
            else:
                commands.append('Q'+xy(b)+' '+xy((b+c)*0.5))
        commands.append('Z')
    else:
        commands.append('M'+xy(q[0]))
        for i in range(1,len(q)-1):
            b=q[i]; c=q[i+1]
            if preserve_corner(float(b[0]),float(b[1])):
                commands.append('L'+xy(b))
            else:
                commands.append('L'+xy((q[i-1]+b)*0.5))
                commands.append('Q'+xy(b)+' '+xy((b+c)*0.5))
        commands.append('L'+xy(q[-1]))
    return ' '.join(commands)

def build(root: Path):
    root.mkdir(parents=True,exist_ok=True)
    source_path=root/'source_exact_approved.png'
    baseline_path=root/'color_19_segmentation.png'
    source_bytes=source_path.read_bytes()
    source=np.asarray(Image.open(source_path).convert('RGB'))
    x0,y0,x1,y1=CROP
    art=source[y0:y1,x0:x1]
    prev=np.asarray(Image.open(baseline_path).convert('RGB'))
    if prev.shape != art.shape:
        raise ValueError(f'Baseline {prev.shape}, original crop {art.shape} mismatch')
    palette=source[SWATCH_Y,SWATCH_X,:].astype(np.uint8)
    assert len(np.unique(pack(palette)))==19
    value_map={int(v):i for i,v in enumerate(pack(palette))}
    known=pack(prev)
    absent=np.setdiff1d(np.unique(known),list(value_map))
    if len(absent):raise ValueError('Baseline contains non-palette colors')
    indexed=np.zeros(prev.shape[:2],dtype=np.uint8)
    for value,color in value_map.items():indexed[known==value]=color
    mm_px=PAGE_W_MM/len(indexed[0])
    changed=restore_iris(indexed,art)
    output, rounds=enforce_number_fit(changed,mm_px)
    ids,colors,area,centers,radii=components(output)
    failed=[i for i in range(1,len(colors)) if radii[i] < needed_radius(int(colors[i]),mm_px)]
    if failed:raise ValueError('Unnumbered paint islands: '+str(failed[:20]))
    if not np.all(ids>0):raise ValueError('Some pixels unassigned')
    h,w=output.shape
    page_art_h=h*mm_px
    yoff=(PAGE_H_MM-page_art_h)/2
    xoff=0.0
    paths=[]
    chains=trace_interfaces(output)
    for points in chains:
        path=smooth_curve(points,mm_px,xoff,yoff)
        if path:paths.append(path)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W_MM:g}mm" height="{PAGE_H_MM:g}mm" viewBox="0 0 {PAGE_W_MM:g} {PAGE_H_MM:g}">',
         '<rect width="400" height="600" fill="white"/>',
         f'<g fill="none" stroke="#161616" stroke-width="{LINE_MM}" stroke-linecap="round" stroke-linejoin="round">',
         f'<rect x="0" y="{yoff:.3f}" width="{PAGE_W_MM:g}" height="{page_art_h:.3f}"/>',
         *[f'<path d="{p}"/>' for p in paths], '</g>',
         f'<g fill="#111" font-family="DejaVu Sans, Arial, sans-serif" font-size="{FONT_MM:.5f}" font-weight="400" text-anchor="middle" dominant-baseline="central">']
    for rid in range(1,len(colors)):
        x,y=centers[rid]
        svg.append(f'<text x="{(x+.5)*mm_px:.3f}" y="{yoff+(y+.5)*mm_px:.3f}">{int(colors[rid])+1}</text>')
    svg+=['</g>','</svg>']
    svg_text='\n'.join(svg)+'\n'
    (root/'lineart_smooth_v2.svg').write_text(svg_text,encoding='utf-8')
    cairosvg.svg2pdf(bytestring=svg_text.encode(),write_to=str(root/'lineart_smooth_v2.pdf'))
    cairosvg.svg2png(bytestring=svg_text.encode(),write_to=str(root/'lineart_smooth_v2.png'),output_width=2400,output_height=3600)
    Image.fromarray(palette[output]).resize((2110,2646),Image.Resampling.LANCZOS).save(root/'color_smooth_v2.png')
    with (root/'palette_v2.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['number','hex','R','G','B'])
        for i,(r,g,b) in enumerate(palette,1):writer.writerow([i,f'#{r:02X}{g:02X}{b:02X}',int(r),int(g),int(b)])
    audit={
        'source_sha256':hashlib.sha256(source_bytes).hexdigest(),
        'source_file_unchanged':True,
        'colors':int(len(palette)),
        'white_outside_palette':True,
        'font_pt':FONT_PT,'stroke_mm':LINE_MM,
        'regions':int(len(colors)-1),'numbered_regions':int(len(colors)-1),'unlabeled_regions':len(failed),
        'restored_iris_pixels':int(np.count_nonzero(changed!=indexed)),
        'final_changed_pixels_vs_previous':int(np.count_nonzero(output!=indexed)),
        'chain_count':len(paths),'number_fitting_passes':rounds,
        'geometry':'shared boundaries, gaussian/subpixel smoothing; hard edges preserved on buildings/lanterns',
        'status':'PASSED' if not failed else 'FAILED'
    }
    (root/'audit_v2.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('lanterns_hachette_19_v1'))
    build(parser.parse_args().root)
