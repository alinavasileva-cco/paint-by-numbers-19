#!/usr/bin/env python3
"""Relocate actual 5.3pt SVG numbers within their OWN regions to clear lines.

No geometry/palette/number/size edits. Input is smoother shared-border SVG.
Print QA renders at 2400x3600, independent of raster fit calculation.
"""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import json, math, xml.etree.ElementTree as ET
import cv2
import cairosvg
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

BASE=Path("face_v5")
SOURCE=BASE/"lineart_review.svg"
SVG="{http://www.w3.org/2000/svg}"
RW,RH=2400,3600
PAGE_W,PAGE_H=400,600

def mask(root,kind):
    doc=deepcopy(root)
    for child in list(doc):
        if child.tag!=SVG+"g":
            doc.remove(child);continue
        contains_text=bool(list(child.iter(SVG+"text")))
        if (kind=="text" and not contains_text) or (kind=="lines" and contains_text):
            doc.remove(child)
    raw=cairosvg.svg2png(bytestring=ET.tostring(doc,encoding="utf-8"),
                            output_width=RW,output_height=RH)
    return np.asarray(Image.open(BytesIO(raw)).convert("RGBA"))[:,:,3]>80

def screen_centers(texts):
    return np.asarray([(float(t.get("x"))*RW/PAGE_W,
                        float(t.get("y"))*RH/PAGE_H) for t in texts],dtype=float)

def region_silhouette_mask(coords_yx,region,region_ids,px_per_mm):
    # Returns whether each actually rendered text ink pixel belongs to
    # its own numbered paint region in the reference-pixel segmentation.
    w=region_ids.shape[1];h=region_ids.shape[0]
    scale=min(PAGE_W/w,PAGE_H/h)
    xoff=(PAGE_W-w*scale)/2;yoff=(PAGE_H-h*scale)/2
    ys=coords_yx[:,0];xs=coords_yx[:,1]
    xsrc=np.rint((xs/px_per_mm-xoff)/scale-.5).astype(int)
    ysrc=np.rint((ys/px_per_mm-yoff)/scale-.5).astype(int)
    inside=(xsrc>=0)&(xsrc<w)&(ysrc>=0)&(ysrc<h)
    if not np.all(inside):return False
    return bool(np.all(region_ids[ysrc,xsrc]==region))

def main():
    root=ET.parse(SOURCE).getroot()
    texts=list(root.iter(SVG+"text"))
    expected=json.loads((BASE/"audit.json").read_text(encoding="utf-8"))["regions_total"]
    if len(texts)!=expected:raise RuntimeError(f"Expected {expected} labels, got {len(texts)}")
    region_ids=np.asarray(Image.open(BASE/"region_ids_16bit.png"),np.uint16)
    original_centers=screen_centers(texts)
    line_ink=mask(root,"lines")
    number_ink=mask(root,"text")
    bad_y,bad_x=np.nonzero(line_ink & number_ink)
    initial_overlap=int(len(bad_x))
    nearest=cKDTree(original_centers)
    _,nearest_id=nearest.query(np.column_stack((bad_x,bad_y)))
    bad_rids=sorted(int(i)+1 for i in np.unique(nearest_id))
    assigned_ink={}
    for rid in bad_rids:
        cx,cy=original_centers[rid-1]
        xmin=max(0,int(cx)-21);xmax=min(RW,int(cx)+22)
        ymin=max(0,int(cy)-21);ymax=min(RH,int(cy)+22)
        yy,xx=np.nonzero(number_ink[ymin:ymax,xmin:xmax])
        xy=np.column_stack((xx+xmin,yy+ymin))
        if not len(xy):continue
        # Extract ONLY this numeral from a crowded area, not its neighbors.
        _,owners=nearest.query(xy)
        xy=xy[owners==rid-1]
        if not len(xy):continue
        assigned_ink[rid]=xy
    occupied=number_ink.copy()
    modifications=[]
    uncorrected=[]
    px_per_mm=RW/PAGE_W
    for rid in sorted(bad_rids,key=lambda r:-int(np.count_nonzero(nearest_id==r-1))):
        if rid not in assigned_ink:
            uncorrected.append({"region_id":rid,"reason":"Missing original number ink pixels"})
            continue
        original_xy=assigned_ink[rid]
        ox=original_xy[:,0];oy=original_xy[:,1]
        occupied[oy,ox]=False
        options=[(dx,dy) for dx in range(-15,16) for dy in range(-15,16)]
        options.sort(key=lambda item:(item[0]*item[0]+item[1]*item[1],
                                      abs(item[0])+abs(item[1])))
        chosen=None
        best=(10**9,10**9)
        for dx,dy in options:
            nx=ox+dx;ny=oy+dy
            if min(nx)<0 or min(ny)<0 or max(nx)>=RW or max(ny)>=RH:continue
            pixels=np.column_stack((ny,nx))
            if not region_silhouette_mask(pixels,rid,region_ids,px_per_mm):
                continue
            border=int(np.count_nonzero(line_ink[ny,nx]))
            others=int(np.count_nonzero(occupied[ny,nx]))
            penalty=border*100 + others*150
            cost=(penalty,dx*dx+dy*dy)
            if cost<best:
                best=cost
                chosen=(dx,dy,border,others)
            if penalty==0:
                break
        if chosen is None:
            occupied[oy,ox]=True
            uncorrected.append({"region_id":rid,"reason":"No shape-contained text location"})
            continue
        dx,dy,border,other=chosen
        if border or other:
            occupied[oy,ox]=True
            uncorrected.append({"region_id":rid,"reason":"No non-overlapping location",
                                "best_border_overlap":border,"best_neighbor_overlap":other})
            continue
        t=texts[rid-1]
        oldx=float(t.get("x"));oldy=float(t.get("y"))
        x=oldx+dx/px_per_mm;y=oldy+dy/px_per_mm
        t.set("x",f"{x:.5f}");t.set("y",f"{y:.5f}")
        transform=t.attrib.get("transform")
        if transform:
            import re
            m=re.search(r'rotate\((-?\d+(?:\.\d+)?)\s+[\d.]+\s+[\d.]+\)',transform)
            if not m:raise RuntimeError(f"Unsupported transform {transform}")
            t.set("transform",f"rotate({m.group(1)} {x:.5f} {y:.5f})")
        occupied[oy+dy,ox+dx]=True
        modifications.append({"region_id":rid,"delta_screen_pixels":[dx,dy],
                            "delta_mm":[round(dx/px_per_mm,3),round(dy/px_per_mm,3)]})
    ET.register_namespace("","http://www.w3.org/2000/svg")
    dest=BASE/"lineart_final_review.svg"
    ET.ElementTree(root).write(dest,encoding="utf-8",xml_declaration=True)
    # Independent actual final render; if any glyph still hits a line, do
    # not call this a completed correction.
    final_line=mask(root,"lines");final_label=mask(root,"text")
    final_overlap=int(np.count_nonzero(final_line & final_label))
    centers=screen_centers(texts)
    still_outside=[]
    h,w=region_ids.shape
    scale=min(PAGE_W/w,PAGE_H/h);xoff=(PAGE_W-w*scale)/2;yoff=(PAGE_H-h*scale)/2
    for rid,center in enumerate(centers,start=1):
        xx,yy=center
        sx=round((xx/px_per_mm-xoff)/scale-.5)
        sy=round((yy/px_per_mm-yoff)/scale-.5)
        if not(0<=sx<w and 0<=sy<h and region_ids[sy,sx]==rid):
            still_outside.append(rid)
    audit={
        "base":"review_art_v4/lineart_soft_review.svg",
        "initial_print_label_outline_overlap_px":initial_overlap,
        "result_print_label_outline_overlap_px":final_overlap,
        "modified_label_count":len(modifications),
        "modifications":modifications,
        "uncorrected_labels":uncorrected,
        "centers_outside_own_region":still_outside,
        "labels_total":len(texts),
        "original_pigment_regions":expected,
        "font_size_pt":5.3,
        "line_width_mm":.2,
        "source_png_unchanged":True,
        "status":"PASS" if final_overlap==0 and not still_outside and not uncorrected else "NEEDS_ARTISTIC_REVIEW"
    }
    (BASE/"print_final_qa.json").write_text(json.dumps(audit,indent=2))
    cairosvg.svg2png(url=str(dest),write_to=str(BASE/"lineart_final_screen.png"),output_width=1200,output_height=1800)
    cairosvg.svg2png(url=str(dest),write_to=str(BASE/"lineart_final_preview.png"),output_width=2400,output_height=3600)
    print(json.dumps({k:v for k,v in audit.items() if k!="modifications"},indent=2))
if __name__=="__main__":main()
